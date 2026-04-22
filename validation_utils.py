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
        
        aug_input = T.AugInput(image)
        transforms = self.augmentations(aug_input)
        image = aug_input.image
        image_shape = image.shape[:2]
        
        dataset_dict["image"] = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))

        if "annotations" in dataset_dict:
            annos = [
                utils.transform_instance_annotations(obj, transforms, image_shape)
                for obj in dataset_dict.pop("annotations")
                if obj.get("iscrowd", 0) == 0
            ]
            instances = utils.annotations_to_instances(annos, image_shape, mask_format="bitmask")
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
    def __init__(self, cfg, dataset_name, period=2000, num_images=50):
        self.period = period
        self.num_images = num_images
        self.dataset_name = dataset_name
        self.cfg = cfg
        self.val_loader = None
        self._val_iter = None
        self._dataset_cache = []

    def _init_loader(self):
        if self.val_loader is None:
            full_dataset = DatasetCatalog.get(self.dataset_name)
            
            # Stratified Sampling to enforce rigorous metric balance
            small_objs, occlusions, normal = [], [], []
            for d in full_dataset:
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
            
            subset_name = f"{self.dataset_name}_eval_subset_200"
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
        
        # Optimal adaptive scheduling
        if next_iter < 20000:
            current_period = 2000
        elif next_iter < 40000:
            current_period = 1000
        else:
            current_period = 500  # Fine tuning phase
            
        if next_iter % current_period == 0:
            self.run_validation(next_iter)

    def run_validation(self, current_iter):
        logger.info(f"Running ValidationHook on {self.num_images} images...")
        self._init_loader()
        
        self.trainer.model.eval()
        
        metrics = {
            "fg_iou": 0.0, "fg_dice": 0.0,
            "bound_iou_loose": 0.0, "bound_iou_strict": 0.0, "bfscore": 0.0,
            "inst_iou": 0.0, "inst_dice": 0.0, "fp_rate": 0.0,
            "small_obj_bound_iou": 0.0, "small_obj_bfscore": 0.0
        }
        
        images_to_log = []
        valid_images = 0
        small_obj_img_count = 0
        
        with torch.no_grad():
            for _ in range(self.num_images):
                batch = self._get_next_batch(current_iter)
                try:
                    outputs = self.trainer.model(batch)
                except Exception as e:
                    logger.warning(f"Validation inference failed: {e}")
                    continue
                
                img_data = batch[0]
                output = outputs[0]
                
                if "instances" not in output or "instances" not in img_data:
                    del outputs
                    continue
                    
                # Force tensors to CUDA strictly. Detectron2 evaluation layers natively move outputs to CPU to save 
                # VRAM, which causes indexing failures and crashes parallel multi-tensor multiplication ops.
                pred_instances = output["instances"].to("cuda")
                gt_instances = img_data["instances"].to("cuda")
                
                valid_preds = pred_instances[pred_instances.scores > 0.5]
                
                pred_masks = valid_preds.pred_masks.to(torch.float32) # [P, H, W]
                gt_masks = gt_instances.gt_masks.tensor.to(pred_masks.device).to(torch.float32) # [G, H, W]
                
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
                
                # Standard BFScore
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
                    
                    best_ious_gt, best_ps_gt_img_idx = inst_ious.max(dim=0) # [G]
                    best_dices_gt, _ = inst_dices.max(dim=0) # [G]
                    
                    valid_gts_mask = best_ious_gt > 0.1
                    matched_iou = torch.where(valid_gts_mask, best_ious_gt, torch.zeros_like(best_ious_gt)).mean().item()
                    matched_dice = torch.where(valid_gts_mask, best_dices_gt, torch.zeros_like(best_dices_gt)).mean().item()
                    
                    # FP Rate
                    best_ious_pred, _ = inst_ious.max(dim=1)
                    unmatched_preds_mask = best_ious_pred <= 0.1
                    fp_area = pred_masks[unmatched_preds_mask].sum().item()
                    total_area = pred_masks.sum().item()
                    fp_rate = fp_area / (total_area + 1e-6)
                    
                    metrics["inst_iou"] += matched_iou
                    metrics["inst_dice"] += matched_dice
                    metrics["fp_rate"] += fp_rate
                    
                    # Small Object Edge Isolation (Tracking delicate features like straps and collars < ~2-5% volume)
                    is_small = gt_masks.sum(dim=(1, 2)) < (H * W * 0.05) 
                    small_gts = is_small.nonzero(as_tuple=True)[0]
                    
                    if len(small_gts) > 0:
                        s_gt_masks = gt_masks[small_gts]
                        # Acquire Pred Masks that best matched these small objects
                        pred_subset_idx = best_ps_gt_img_idx[small_gts]
                        s_pred_masks = pred_masks[pred_subset_idx]
                        
                        s_gb_strict = torch.clamp(get_boundaries(s_gt_masks, k_strict).sum(0), 0, 1) > 0
                        s_pb_strict = torch.clamp(get_boundaries(s_pred_masks, k_strict).sum(0), 0, 1) > 0
                        
                        s_bound_iou, _ = compute_ious(s_pb_strict.view(1, -1).float(), s_gb_strict.view(1, -1).float())
                        s_bfscore = compute_bfscore(s_pb_strict, s_gb_strict, threshold_px=3)
                        
                        # Only aggregate valid small object scores if they weren't entirely failed detections
                        if valid_gts_mask[small_gts].any():
                            metrics["small_obj_bound_iou"] += s_bound_iou.item()
                            metrics["small_obj_bfscore"] += s_bfscore
                            small_obj_img_count += 1
                
                # --- 3. Visualization ---
                if valid_images < 3:
                    vis_img = img_data["image"].clone().cpu().float()
                    c_min, c_max = vis_img.min(), vis_img.max()
                    if c_max > c_min: vis_img = (vis_img - c_min) / (c_max - c_min)
                        
                    pb = pb_strict.cpu()
                    gb = gb_strict.cpu()
                    
                    error_map = (pred_fg_mask.cpu() != gt_fg_mask.cpu()).float()
                    vis_img[2] = torch.max(vis_img[2], error_map * 0.6)
                    
                    vis_img[1][gb > 0] = 1.0 # Pure GT boundaries Green
                    vis_img[0][pb > 0] = 1.0 # Pure Pred boundaries Red
                    vis_img[2][(pb > 0) | (gb > 0)] = 0.0 # Suppress blue

                    images_to_log.append(vis_img)
                    
                valid_images += 1
                del outputs, img_data, pred_instances, gt_instances, pred_masks, gt_masks
                
        # --- 4. Logs ---
        if valid_images > 0:
            storage = self.trainer.storage
            
            for k in ["fg_iou", "fg_dice", "bound_iou_loose", "bound_iou_strict", "bfscore", "inst_iou", "inst_dice", "fp_rate"]:
                storage.put_scalar(f"val/{k}", metrics[k] / valid_images)
                
            if small_obj_img_count > 0:
                storage.put_scalar(f"val/small_obj_bound_iou", metrics["small_obj_bound_iou"] / small_obj_img_count)
                storage.put_scalar(f"val/small_obj_bfscore", metrics["small_obj_bfscore"] / small_obj_img_count)
            
            # Write explicitly to console log so it skips Tensorboard parsing for fast debugging
            logger.info(f"[VAL] Strict BFScore: {metrics['bfscore'] / valid_images:.4f} | Strict Bound IoU: {metrics['bound_iou_strict'] / valid_images:.4f}")
            
            if images_to_log:
                grid = torch.cat(images_to_log, dim=2)
                storage.put_image("val/overlays_and_errors", grid.numpy())

        torch.cuda.empty_cache()
        self.trainer.model.train()
