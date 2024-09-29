import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.config import setup_cfg
from detectron2.engine import DefaultPredictor

class SkinModel:
    def __init__(self):
        self.cfg = setup_cfg()
        self.predictor = DefaultPredictor(self.cfg)

    def predict(self, image):
        return self.predictor(image)