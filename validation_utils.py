import os
import json
import torch
import torch.nn.functional as F
from detectron2.engine.hooks import HookBase
from detectron2.data import build_detection_test_loader, DatasetMapper, DatasetCatalog
import logging
import random
import copy
import numpy as np
from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T

logger = logging.getLogger(__name__)

class ValidationDatasetMapper:
    """
    A brutally minimalist mapper to extract test-size images alongside perfectly mapped
    Ground Truth BitMasks, bypassing all random training augmentations cleanly.
    """
    def __init__(self, cfg):
        self.image_format = cfg.INPUT.FORMAT
        self.augmentations = T.AugmentationList([
            T.ResizeShortestEdge(
                [cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MIN_SIZE_TEST], 
                cfg.INPUT.MAX_SIZE_TEST
            )
        ])
        
    def __call__(self, dataset_dict):
        dataset_dict = copy.deepcopy(dataset_dict)
        image = utils.read_image(dataset_dict["file_name"], format=self.image_format)
        utils.check_image_size(dataset_dict, image)
        
        orig_shape = (dataset_dict.get("height", image.shape[0]), dataset_dict.get("width", image.shape[1]))
        
        aug_input = T.AugInput(image)
        transforms = self.augmentations(aug_input)
        image = aug_input.image
        
        dataset_dict["image"] = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))

        if "annotations" in dataset_dict:
            # Bypassing the Inference resize transforms to enforce Ground Truth processing at
            # the PERFECT Native Resolution! (Mask2Former auto-upscales Predictions to Native as well)
            no_op = T.TransformList([])
            annos = [
                utils.transform_instance_annotations(obj, no_op, orig_shape)
                for obj in dataset_dict.pop("annotations")
                if obj.get("iscrowd", 0) == 0
            ]
            instances = utils.annotations_to_instances(annos, orig_shape, mask_format="bitmask")
            dataset_dict["instances"] = instances
            
        return dataset_dict

def get_boundaries(masks: torch.Tensor, dilation_kernel_size=3):
    """
    Computes Mask Boundaries using Dilation - Erosion.
    masks: [N, H, W] boolean or float masking tensor.
    """
    if masks.numel() == 0:
        return masks
        
    masks_float = masks.float().unsqueeze(1) # [N, 1, H, W]
    pad = dilation_kernel_size // 2
    
    # Dilation: max pool equivalent
    dilated = F.max_pool2d(masks_float, kernel_size=dilation_kernel_size, stride=1, padding=pad)
    # Erosion: local min equivalent -> -max_pool(-x)
    eroded = -F.max_pool2d(-masks_float, kernel_size=dilation_kernel_size, stride=1, padding=pad)
    
    boundary = (dilated - eroded) > 0.5
    return boundary.squeeze(1)

def compute_ious(preds, targets):
    if preds.shape[0] == 0 or targets.shape[0] == 0:
        return torch.zeros((preds.shape[0], targets.shape[0]), device=preds.device), \
               torch.zeros((preds.shape[0], targets.shape[0]), device=preds.device)
               
    intersection = torch.mm(preds, targets.T) # [P, G]
    p_sum = preds.sum(-1).unsqueeze(1) # [P, 1]
    g_sum = targets.sum(-1).unsqueeze(0) # [1, G]
    
    union = p_sum + g_sum - intersection
    iou = intersection / (union + 1e-6)
    dice = 2 * intersection / (p_sum + g_sum + 1e-6)
    return iou, dice

def compute_bfscore(pred_bound, gt_bound, threshold_px=2):
    """
    Native PyTorch Boundary F-Score (BFScore) Approximation.
    Measures how close predicted edges are to GT edges regardless of perfect pixel overlap.
    NOTE: MaxPool2D uses Chebyshev distance (Square Radius) rather than true Euclidean distance. 
    This is extremely fast for training monitoring, but will slightly overestimate diagonal/curved overlaps.
    """
    k = threshold_px * 2 + 1
    pad = threshold_px
    
    pb_float = pred_bound.float().unsqueeze(1)
    gb_float = gt_bound.float().unsqueeze(1)
    
    dilated_gt = F.max_pool2d(gb_float, kernel_size=k, stride=1, padding=pad) > 0.5
    dilated_pred = F.max_pool2d(pb_float, kernel_size=k, stride=1, padding=pad) > 0.5
    
    # Precision: How much of pred boundary is within D px of GT?
    matched_pred = (pred_bound & dilated_gt.squeeze(1)).sum()
    precision = matched_pred / (pred_bound.sum() + 1e-6)
    
    # Recall: How much of GT boundary is within D px of Pred?
    matched_gt = (gt_bound & dilated_pred.squeeze(1)).sum()
    recall = matched_gt / (gt_bound.sum() + 1e-6)
    
    bfscore = 2 * precision * recall / (precision + recall + 1e-6)
    return bfscore.item()

class ValidationHook(HookBase):
    TOTAL_SUBSET = 200
    CHUNK_SIZE = 50
    
    def __init__(
        self,
        cfg,
        dataset_name,
        period=2000,
        num_images=50,
        adaptive_period=True,
        flush_every_chunk=False,
    ):
        self.period = period
        self.num_images = num_images
        self.adaptive_period = adaptive_period
        self.flush_every_chunk = flush_every_chunk
        self.dataset_name = dataset_name
        self.cfg = cfg
        self.val_loader = None
        self._val_iter = None
        self._dataset_cache = []
        
        # Accumulation state — aggregate across 4 chunks before logging
        self._accum_metrics = None
        self._accum_valid = 0
        self._accum_small_obj_count = 0
        self._accum_images_logged = []
        self._accum_chunks_done = 0
        self._total_chunks = max(1, (self.TOTAL_SUBSET + self.num_images - 1) // self.num_images)
    
    def _reset_accumulator(self):
        self._accum_metrics = {
            "fg_iou": 0.0, "fg_dice": 0.0,
            "bound_iou_loose": 0.0, "bound_iou_strict": 0.0, "bfscore": 0.0,
            "inst_iou": 0.0, "inst_dice": 0.0, "fp_rate": 0.0,
            "small_obj_bound_iou": 0.0, "small_obj_bfscore": 0.0
        }
        self._accum_valid = 0
        self._accum_small_obj_count = 0
        self._accum_images_logged = []
        self._accum_chunks_done = 0

    def _init_loader(self):
        if self.val_loader is None:
            subset_name = f"{self.dataset_name}_eval_subset_200"
            subset_cache_path = os.path.join(self.cfg.OUTPUT_DIR, f"{subset_name}.json")

            
            if os.path.exists(subset_cache_path):
                logger.info(f"Loading fast subset cache from {subset_cache_path}")
                with open(subset_cache_path, "r") as f:
                    subset_data = json.load(f)
                
                # Validate cached paths still exist on disk
                missing = [d for d in subset_data if not os.path.exists(d.get("file_name", ""))]
                if missing:
                    logger.warning(f"Cache has {len(missing)} missing files — rebuilding subset")
                    os.remove(subset_cache_path)
                    subset_data = None
                    
            if not os.path.exists(subset_cache_path):
                full_dataset = DatasetCatalog.get(self.dataset_name)
                
                # Stratified Sampling to enforce rigorous metric balance
                small_objs, occlusions, normal = [], [], []
                for d in full_dataset:
                    # Skip entries where the image file doesn't exist
                    if not os.path.exists(d.get("file_name", "")):
                        continue
                        
                    anns = d.get("annotations", [])
                    
                    # Default to 1024 if somehow missing to prevent crash
                    img_area = d.get("height", 1024) * d.get("width", 1024)
                    # Ensure we catch small objects accurately
                    has_small = any(a.get("area", 0) < 0.05 * img_area for a in anns)
                    has_overlap = len(anns) > 3
                    
                    if has_small:
                        small_objs.append(d)
                    elif has_overlap:
                        occlusions.append(d)
                    else:
                        normal.append(d)
                        
                rng = random.Random(42)
                
                # Sub-sample safely incase of extreme set imbalance
                n_small = min(50, len(small_objs))
                n_occ = min(50, len(occlusions))
                n_normal = min(200 - (n_small + n_occ), len(normal))
                
                subset_data = rng.sample(small_objs, n_small) + \
                              rng.sample(occlusions, n_occ) + \
                              rng.sample(normal, n_normal)
                              
                # CRITICAL: Shuffle the combined 200 subset ONCE so rolling chunks aren't statically biased
                # This directly prevents metric oscillation on TensorBoard graphs
                rng.shuffle(subset_data)
                
                # Save out to physical cache so future reboots don't parse 60k JSON string arrays
                with open(subset_cache_path, "w") as f:
                    json.dump(subset_data, f)
            
            if subset_name in DatasetCatalog.list():
                DatasetCatalog.remove(subset_name)
                
            frozen_subset = list(subset_data)
            DatasetCatalog.register(subset_name, lambda d=frozen_subset: d)
            
            # Copy metadata so the evaluator mapping perfectly mimics the origin dataset metrics
            from detectron2.data import MetadataCatalog
            origin_meta = dict(MetadataCatalog.get(self.dataset_name).as_dict())
            if "name" in origin_meta:
                del origin_meta["name"]
            MetadataCatalog.get(subset_name).set(**origin_meta)
            
            self.val_loader = build_detection_test_loader(
                self.cfg, 
                subset_name,
                mapper=ValidationDatasetMapper(self.cfg)
            )
            self._val_iter = iter(self.val_loader)

    def _get_next_batch(self, iteration):
        # Dataloader implicitly rotates 50 images per iteration, wrapping nicely at 200
        try:
            return next(self._val_iter)
        except StopIteration:
            self._val_iter = iter(self.val_loader)
            return next(self._val_iter)

    def after_step(self):
        next_iter = self.trainer.iter + 1

        if self.adaptive_period:
            # Adaptive scheduling
            if next_iter < 20000:
                current_period = 2000
            elif next_iter < 40000:
                current_period = 1000
            else:
                current_period = 500  # Fine tuning phase
        else:
            current_period = max(1, int(self.period))

        if next_iter % current_period == 0:
            self._run_chunk(next_iter)

    def _run_chunk(self, current_iter):
        """Run one 50-image chunk. After 4 chunks (200 images), log aggregated metrics."""
        if self._accum_metrics is None:
            self._reset_accumulator()
        
        self._run_validation_chunk(current_iter)
        self._accum_chunks_done += 1

        # Short-run mode: flush after each chunk so val metrics are visible every period.
        if self.flush_every_chunk:
            self._flush_metrics(current_iter)
            self._reset_accumulator()
        elif self._accum_chunks_done >= self._total_chunks:
            self._flush_metrics(current_iter)
            self._reset_accumulator()

    def _run_validation_chunk(self, current_iter):
        """Evaluate one chunk of 50 images and accumulate into running totals."""
        logger.info(f"Running ValidationHook chunk {self._accum_chunks_done + 1}/{self._total_chunks} ({self.num_images} images)...")
        self._init_loader()
        
        self.trainer.model.eval()
        metrics = self._accum_metrics
        
        with torch.inference_mode():
            for _ in range(self.num_images):
                img_data = self._get_next_batch(current_iter)[0]
                outputs = self.trainer.model([img_data])
                output = outputs[0]
                
                if "instances" not in output or "instances" not in img_data:
                    del outputs
                    continue
                    
                pred_instances = output["instances"]
                gt_instances = img_data["instances"]
                
                all_scores = pred_instances.scores.detach().cpu()
                keep = all_scores > 0.1
                
                if keep.sum() == 0:
                    metrics["fp_rate"] += 1.0
                    self._accum_valid += 1
                    del outputs
                    continue
                
                pred_masks = pred_instances.pred_masks[keep].to("cuda", non_blocking=True)
                scores = all_scores[keep].to("cuda", non_blocking=True)
                    
                gt_masks_raw = gt_instances.gt_masks.tensor
                if not gt_masks_raw.is_cuda:
                    gt_masks_raw = gt_masks_raw.to("cuda", non_blocking=True)
                
                pred_masks = pred_masks.to(torch.float32)
                gt_masks = gt_masks_raw.to(torch.float32)

                if pred_masks.shape[-2:] != gt_masks.shape[-2:]:
                    del outputs
                    continue
                
                H, W = pred_masks.shape[-2:]
                if H == 0 or W == 0:
                    del outputs
                    continue
                    
                k_loose = max(3, int(0.01 * max(H, W)))
                if k_loose % 2 == 0: k_loose += 1
                k_strict = 3
                
                # --- 1. Foreground Aggregation Metrics ---
                pred_fg_mask = (pred_masks.sum(0) > 0)
                gt_fg_mask = (gt_masks.sum(0) > 0)
                
                pred_fg = pred_fg_mask.float().view(1, H*W)
                gt_fg = gt_fg_mask.float().view(1, H*W)
                
                fg_iou, fg_dice = compute_ious(pred_fg, gt_fg)
                
                pb_loose = torch.zeros((H, W), device=pred_masks.device)
                pb_strict = torch.zeros((H, W), device=pred_masks.device)
                gb_loose = torch.zeros((H, W), device=pred_masks.device)
                gb_strict = torch.zeros((H, W), device=pred_masks.device)
                
                if len(pred_masks) > 0:
                    pb_loose = torch.clamp(get_boundaries(pred_masks, k_loose).sum(0), 0, 1) > 0
                    pb_strict = torch.clamp(get_boundaries(pred_masks, k_strict).sum(0), 0, 1) > 0
                    
                if len(gt_masks) > 0:
                    gb_loose = torch.clamp(get_boundaries(gt_masks, k_loose).sum(0), 0, 1) > 0
                    gb_strict = torch.clamp(get_boundaries(gt_masks, k_strict).sum(0), 0, 1) > 0
                
                loose_iou, _ = compute_ious(pb_loose.view(1, -1).float(), gb_loose.view(1, -1).float())
                strict_iou, _ = compute_ious(pb_strict.view(1, -1).float(), gb_strict.view(1, -1).float())
                
                bfscore = compute_bfscore(pb_strict, gb_strict, threshold_px=3)
                
                metrics["fg_iou"] += fg_iou.item()
                metrics["fg_dice"] += fg_dice.item()
                metrics["bound_iou_loose"] += loose_iou.item()
                metrics["bound_iou_strict"] += strict_iou.item()
                metrics["bfscore"] += bfscore
                
                # --- 2. Instance Metrics & Small Object Tracking ---
                if len(pred_masks) > 0 and len(gt_masks) > 0:
                    p_flat = pred_masks.view(len(pred_masks), H*W)
                    g_flat = gt_masks.view(len(gt_masks), H*W)
                    inst_ious, inst_dices = compute_ious(p_flat, g_flat)
                    
                    best_ious_gt, best_ps_gt_img_idx = inst_ious.max(dim=0)
                    best_dices_gt, _ = inst_dices.max(dim=0)
                    
                    valid_gts_mask = best_ious_gt > 0.1
                    matched_iou = torch.where(valid_gts_mask, best_ious_gt, torch.zeros_like(best_ious_gt)).mean().item()
                    matched_dice = torch.where(valid_gts_mask, best_dices_gt, torch.zeros_like(best_dices_gt)).mean().item()
                    
                    best_ious_pred, _ = inst_ious.max(dim=1)
                    unmatched_preds_mask = best_ious_pred <= 0.1
                    fp_area = pred_masks[unmatched_preds_mask].sum().item()
                    total_area = pred_masks.sum().item()
                    fp_rate = fp_area / (total_area + 1e-6)
                    
                    metrics["inst_iou"] += matched_iou
                    metrics["inst_dice"] += matched_dice
                    metrics["fp_rate"] += fp_rate
                    
                    is_small = gt_masks.sum(dim=(1, 2)) < (H * W * 0.05)
                    small_gts = is_small.nonzero(as_tuple=True)[0]
                    
                    if len(small_gts) > 0:
                        s_gt_masks = gt_masks[small_gts]
                        pred_subset_idx = best_ps_gt_img_idx[small_gts]
                        s_pred_masks = pred_masks[pred_subset_idx]
                        
                        s_gb_strict = torch.clamp(get_boundaries(s_gt_masks, k_strict).sum(0), 0, 1) > 0
                        s_pb_strict = torch.clamp(get_boundaries(s_pred_masks, k_strict).sum(0), 0, 1) > 0
                        
                        s_bound_iou, _ = compute_ious(s_pb_strict.view(1, -1).float(), s_gb_strict.view(1, -1).float())
                        s_bfscore = compute_bfscore(s_pb_strict, s_gb_strict, threshold_px=3)
                        
                        if valid_gts_mask[small_gts].any():
                            metrics["small_obj_bound_iou"] += s_bound_iou.item()
                            metrics["small_obj_bfscore"] += s_bfscore
                            self._accum_small_obj_count += 1
                
                # --- 3. Visualization (collect up to 3 across all chunks) ---
                if len(self._accum_images_logged) < 3:
                    vis_img = img_data["image"].clone().cpu().float()
                    c_min, c_max = vis_img.min(), vis_img.max()
                    if c_max > c_min: vis_img = (vis_img - c_min) / (c_max - c_min)
                    
                    vis_h, vis_w = vis_img.shape[1], vis_img.shape[2]
                    
                    pb = F.interpolate(pb_strict.float().unsqueeze(0).unsqueeze(0), size=(vis_h, vis_w), mode="nearest").squeeze().cpu() > 0
                    gb = F.interpolate(gb_strict.float().unsqueeze(0).unsqueeze(0), size=(vis_h, vis_w), mode="nearest").squeeze().cpu() > 0
                    error_map = F.interpolate(
                        (pred_fg_mask != gt_fg_mask).float().unsqueeze(0).unsqueeze(0), size=(vis_h, vis_w), mode="nearest"
                    ).squeeze().cpu()
                    
                    vis_img[2] = torch.max(vis_img[2], error_map * 0.6)
                    vis_img[1][gb > 0] = 1.0
                    vis_img[0][pb > 0] = 1.0
                    vis_img[2][(pb > 0) | (gb > 0)] = 0.0

                    self._accum_images_logged.append(vis_img)
                    
                self._accum_valid += 1
                del outputs, img_data, pred_instances, gt_instances, pred_masks, gt_masks
                
        torch.cuda.empty_cache()
        self.trainer.model.train()
    
    def _flush_metrics(self, current_iter):
        """Log aggregated metrics over all 200 images to TensorBoard + stdout."""
        valid = self._accum_valid
        if valid == 0:
            return
        
        metrics = self._accum_metrics
        storage = self.trainer.storage
        
        for k in ["fg_iou", "fg_dice", "bound_iou_loose", "bound_iou_strict", "bfscore", "inst_iou", "inst_dice", "fp_rate"]:
            storage.put_scalar(f"val/{k}", metrics[k] / valid)
            
        if self._accum_small_obj_count > 0:
            storage.put_scalar(f"val/small_obj_bound_iou", metrics["small_obj_bound_iou"] / self._accum_small_obj_count)
            storage.put_scalar(f"val/small_obj_bfscore", metrics["small_obj_bfscore"] / self._accum_small_obj_count)
        
        logger.info(
            f"[VAL@{current_iter}] Full 200-image aggregate | "
            f"BFScore: {metrics['bfscore'] / valid:.4f} | "
            f"Bound IoU (strict): {metrics['bound_iou_strict'] / valid:.4f} | "
            f"FG IoU: {metrics['fg_iou'] / valid:.4f} | "
            f"FP Rate: {metrics['fp_rate'] / valid:.4f} | "
            f"Images: {valid}/200"
        )
        
        if self._accum_images_logged:
            grid = torch.cat(self._accum_images_logged, dim=2)
            storage.put_image("val/overlays_and_errors", grid.numpy())

    def run_validation(self, current_iter):
        """Run full 200-image evaluation in one shot (used by evaluate.py)."""
        self._reset_accumulator()
        orig_num = self.num_images
        self.num_images = self.TOTAL_SUBSET
        self._run_validation_chunk(current_iter)
        self._flush_metrics(current_iter)
        self.num_images = orig_num
        self._reset_accumulator()
