import os
import torch
import logging

from detectron2.checkpoint import DetectionCheckpointer
from detectron2.modeling import build_model
from detectron2.utils.events import EventStorage

# Mask2Former setup
from config_setup import build_cfg
from register_dataset import register_fashion_datasets, get_thing_classes
from validation_utils import ValidationHook

# Minimal mock class for trainer object because HookBase expects `self.trainer.iter` and `self.trainer.storage`
class MockTrainer:
    def __init__(self, model, storage):
        self.model = model
        self.storage = storage
        self.iter = 0

def test_hook():
    logging.basicConfig(level=logging.INFO)
    
    # 1. Boot up configurations safely outside training loops
    register_fashion_datasets()
    thing_classes = get_thing_classes()
    cfg = build_cfg(output_dir="./output_swin_fashion", resume=False, backbone="SWIN_T", num_classes=len(thing_classes))
    cfg.defrost()
    cfg.MODEL.WEIGHTS = "./output_swin_fashion/model_0027499.pth"
    cfg.freeze()

    # 2. Instance physical Swin model onto GPU explicitly
    print("[TEST] Building actual model to guarantee math operations compile...")
    model = build_model(cfg)
    model.eval()
    DetectionCheckpointer(model).resume_or_load(cfg.MODEL.WEIGHTS, resume=False)

    # 3. Trigger precise memory-safe storage
    with EventStorage() as storage:
        trainer = MockTrainer(model, storage)
        
        # 4. We will run it once with exactly 5 images so it evaluates fast (2 seconds)
        print("[TEST] Launching isolated ValidationHook slice...")
        hook = ValidationHook(cfg, "fashion_val", period=2000, num_images=5)
        hook.trainer = trainer
        
        hook.run_validation(current_iter=27499)
        
        print("\n\n=============== MATRIC RESULTS ===============")
        print(f"BFScore (Strict):   {storage.latest().get('val/bfscore', (0,0))[0]:.4f}")
        print(f"Bound IoU (Strict): {storage.latest().get('val/bound_iou_strict', (0,0))[0]:.4f}")
        print(f"Bound IoU (Loose):  {storage.latest().get('val/bound_iou_loose', (0,0))[0]:.4f}")
        print(f"False Positive Rate:{storage.latest().get('val/fp_rate', (0,0))[0]:.4f}")
        print("==============================================\n")
        print("✅ Validation hook completed securely without crashing!")

if __name__ == "__main__":
    test_hook()
