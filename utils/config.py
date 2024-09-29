import os
from detectron2.config import get_cfg
from detectron2 import model_zoo

def setup_cfg():
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 6  # Number of classes
    cfg.MODEL.WEIGHTS = os.path.join("model", "model_final.pth")  # Model path
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    cfg.MODEL.DEVICE = "cpu"  # Use CPU
    return cfg
