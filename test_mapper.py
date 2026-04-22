import os
import torch
import numpy as np

from detectron2.checkpoint import DetectionCheckpointer
from detectron2.modeling import build_model
from detectron2.utils.events import EventStorage
from detectron2.data import DatasetCatalog, DatasetMapper

from config_setup import build_cfg
from register_dataset import register_fashion_datasets, get_thing_classes

register_fashion_datasets()
thing_classes = get_thing_classes()
cfg = build_cfg(output_dir="./output_swin_fashion", resume=False, backbone="SWIN_T", num_classes=len(thing_classes))
cfg.defrost()
cfg.MODEL.WEIGHTS = "./output_swin_fashion/model_0027499.pth"
cfg.freeze()

model = build_model(cfg)
model.eval()
DetectionCheckpointer(model).resume_or_load(cfg.MODEL.WEIGHTS, resume=False)

dataset = DatasetCatalog.get("fashion_val")
# take first image
d = dataset[0]

import copy
from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T

class ValMapperWDGT(DatasetMapper):
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
            ]
            instances = utils.annotations_to_instances(annos, image_shape, mask_format="bitmask")
            dataset_dict["instances"] = instances
        return dataset_dict

mapper = ValMapperWDGT(cfg, is_train=False)
batch = [mapper(d)]

with torch.no_grad():
    outputs = model(batch)
    print("GT INSTANCES:", batch[0].get("instances"))
    print("OUTPUT INSTANCES:", outputs[0].get("instances"))

