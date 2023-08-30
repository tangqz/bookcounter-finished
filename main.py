from detectron2.engine import DefaultPredictor

import os
import pickle
from PIL import Image, ImageDraw, ImageFont
import numpy as np

import gradio as gr

from utils import *

cfg_save_path = "OD_cfg.pickle"

with open(cfg_save_path, 'rb') as f:
    cfg = pickle.load(f)

cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5

predictor = DefaultPredictor(cfg)

#image_path = "C:\\Users\\qizhi\\Desktop\\coding\\book_counter\\d1.jpg"
#on_image(image_path,predictor)

def predict(image):
    outputs = predictor(image)
    v = Visualizer(image[:, :, ::-1],metadata={},scale=0.5,instance_mode=ColorMode.SEGMENTATION)
    v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    return [v.get_image(),len(outputs["instances"].to("cpu"))]

#运行gradio
demo = gr.Interface(
    predict, "image", ["image","number"], 
    title="Book Counter", 
    description="从书堆侧视图检测书的数量。\n基于Detectron2。\nModel:mask_rcnn_R_50_FPN_3x\nBy qizhi7z", 
    examples=[["test.jpg","test12.jpg"]]
)
demo.launch(server_name = '0.0.0.0')