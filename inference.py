import yaml
import torch
from torch.backends import cudnn

from backbone import EfficientDetBackbone
import cv2
import numpy as np
import os

from efficientdet.utils import BBoxTransform, ClipBoxes
from utils.utils import preprocess, invert_affine, postprocess, STANDARD_COLORS, standard_to_bgr, get_index_label, plot_one_box

class Inference():
    def __init__(self, weightsPath, projectPath ):
        with open(projectPath, "r") as stream:
            self.p_data = yaml.safe_load(stream)
            
        self.use_cuda = False
        if torch.cuda.is_available():
            cudnn.fastest = True
            cudnn.benchmark = True
            self.use_cuda = True
        
        self.compound_coef = int(weightsPath[-5])
        
        force_input_size = None
        input_sizes = [512, 640, 768, 896, 1024, 1280, 1280, 1536, 1536]
        self.input_size = input_sizes[self.compound_coef] if force_input_size is None else force_input_size

        self.regressBoxes = BBoxTransform()
        self.clipBoxes = ClipBoxes()
        self.initialize_model(weightsPath)
    
    
    def forward(self, img_path, acc_threshold = 0.2, iou_threshold = 0.2):
        ori_imgs, framed_imgs, framed_metas = preprocess(img_path, max_size=self.input_size)

        if self.use_cuda:
            x = torch.stack([torch.from_numpy(fi).cuda() for fi in framed_imgs], 0)
        else:
            x = torch.stack([torch.from_numpy(fi) for fi in framed_imgs], 0)

        x = x.to(torch.float32).permute(0, 3, 1, 2)
        
        out = self.model_inference(x, acc_threshold, iou_threshold)
        out = invert_affine(framed_metas, out)
        return out[0]
        
    
    def model_inference(self, x, acc_threshold, iou_threshold):
        with torch.no_grad():
            features, regression, classification, anchors = self.model(x)

            out = postprocess(x,
                            anchors, regression, classification,
                            self.regressBoxes, self.clipBoxes,
                            acc_threshold, iou_threshold)
            
            return out

    def initialize_model(self,weightsPath):
        obj_list = self.p_data['obj_list']
        anchor_ratios = eval(self.p_data['anchors_ratios'])
        anchor_scales = eval(self.p_data['anchors_scales'])
        
        self.model = EfficientDetBackbone(compound_coef=self.compound_coef, num_classes=len(obj_list),
                                    ratios=anchor_ratios, scales=anchor_scales)
        self.model.load_state_dict(torch.load(weightsPath, map_location='cpu'))
        self.model.requires_grad_(False)
        self.model.eval()

        if self.use_cuda:
            self.model = model.cuda()
        
weightsPath = "./weights/efficientdet-d0.pth"
projectPath = "./projects/coco.yml"

infer = Inference(weightsPath, projectPath)

