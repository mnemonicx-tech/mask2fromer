import os
import torch
import logging

from detectron2.checkpoint import DetectionCheckpointer
from detectron2.modeling import build_model
from detectron2.utils.events import EventStorage
from detectron2.data import DatasetCatalog

from config_setup import build_cfg
from register_dataset import register_fashion_datasets, get_thing_classes
from validation_utils import ValidationHook, ValidationDatasetMapper
from detectron2.data import build_detection_test_loader

def test_hook():
    logging.basicConfig(level=logging.INFO)
    
    register_fashion_datasets()
    thing_classes = get_thing_classes()
    cfg = build_cfg(output_dir="./output_swin_fashion", resume=False, backbone="SWIN_T", num_classes=len(thing_classes))
    cfg.defrost()
    cfg.MODEL.WEIGHTS = "./output_swin_fashion/model_0027499.pth"
    cfg.freeze()

    print("[TEST] Building model...")
    model = build_model(cfg)
    model.eval()
    DetectionCheckpointer(model).resume_or_load(cfg.MODEL.WEIGHTS, resume=False)

    # ---- RAW DIAGNOSTIC: What does the model actually output? ----
    print("\n======= RAW MODEL OUTPUT DIAGNOSTIC =======")
    
    mapper = ValidationDatasetMapper(cfg)
    dataset = DatasetCatalog.get("fashion_val")
    
    for i in range(3):
        d = dataset[i]
        mapped = mapper(d)
        
        print(f"\n--- Image {i}: {os.path.basename(d['file_name'])} ---")
        print(f"  Image tensor shape: {mapped['image'].shape}")
        
        if "instances" in mapped:
            gt = mapped["instances"]
            print(f"  GT instances: {len(gt)}")
            print(f"  GT masks shape: {gt.gt_masks.tensor.shape}")
        else:
            print("  ⚠️ NO GT instances in mapped output!")
        
        with torch.inference_mode():
            outputs = model([mapped])
        
        pred = outputs[0]["instances"]
        print(f"  Pred instances: {len(pred)}")
        
        if len(pred) > 0:
            scores = pred.scores
            print(f"  Scores device: {scores.device}")
            print(f"  Scores range: [{scores.min().item():.4f}, {scores.max().item():.4f}]")
            print(f"  Scores > 0.5: {(scores > 0.5).sum().item()}")
            print(f"  Scores > 0.3: {(scores > 0.3).sum().item()}")
            print(f"  Scores > 0.1: {(scores > 0.1).sum().item()}")
            print(f"  Pred masks shape: {pred.pred_masks.shape}")
            print(f"  Pred masks device: {pred.pred_masks.device}")
            
            if "instances" in mapped:
                gt_shape = mapped["instances"].gt_masks.tensor.shape[-2:]
                pred_shape = pred.pred_masks.shape[-2:]
                print(f"  GT mask HxW: {gt_shape}")
                print(f"  Pred mask HxW: {pred_shape}")
                if gt_shape != pred_shape:
                    print(f"  ⚠️ RESOLUTION MISMATCH! GT={gt_shape} vs Pred={pred_shape}")
                else:
                    print(f"  ✅ Resolution match")
        else:
            print("  ⚠️ Zero predictions!")
        
        del outputs
    
    print("\n============================================")
    
    # ---- Now run the actual hook ----
    print("\n[TEST] Running full ValidationHook...")
    with EventStorage() as storage:
        class MockTrainer:
            def __init__(self, m, s):
                self.model = m
                self.storage = s
                self.iter = 0
        
        trainer = MockTrainer(model, storage)
        hook = ValidationHook(cfg, "fashion_val", period=2000, num_images=5)
        hook.trainer = trainer
        hook.run_validation(current_iter=27499)
        
        print("\n=============== METRIC RESULTS ===============")
        for key in ["val/bfscore", "val/bound_iou_strict", "val/bound_iou_loose", "val/fg_iou", "val/fg_dice", "val/fp_rate"]:
            val = storage.latest().get(key, (0,0))[0]
            print(f"  {key}: {val:.4f}")
        print("==============================================\n")
        print("✅ Validation hook completed!")

if __name__ == "__main__":
    test_hook()
