# Class imports
import sys, os
import torch, detectron2
from detectron2.utils.logger import setup_logger
setup_logger()
import numpy as np
import os, json, cv2, random
#from google.colab.patches import cv2_imshow
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.utils import *
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.evaluation import *
import glob
import time

import certifi
import ssl
def create_context():
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    context.load_verify_locations(certifi.where())
    return context
ssl._create_default_https_context = create_context

# Setting up code to convert detectron output to correct input for SORT detections
instances = []
sequence = []

cfg = get_cfg()
cfg.MODEL.DEVICE ='cpu'
cfg_file_name = "Cityscapes/mask_rcnn_R_50_FPN.yaml"
cfg.merge_from_file(model_zoo.get_config_file(cfg_file_name))
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST=0.7
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(cfg_file_name)

classes = MetadataCatalog.get(cfg.DATASETS.TRAIN[0]).thing_classes

def bboxes_convert(results):
    bboxes = results.pred_boxes if results.has('pred_boxes') else None
    bboxes = Visualizer._convert_boxes(Visualizer.__init__,bboxes.to('cpu'))
    return bboxes

def scores_convert(results):
    scores = results.scores
    scores = scores.to('cpu').numpy()
    return np.asarray(scores)

def classes_convert(results):
    classes = results.pred_classes
    classes = classes.to('cpu').numpy()
    return np.asarray(classes)

def convertDetectionsSORT():
    print("Converting Detections")
    frameNum = len(instances)
    dets = []
    #print("Instances length: %s"%(frameNum))

    with open("./data/train/%s/testing/det.txt"%(sequence), 'w') as outfile:
        for data in range(len(instances)):
            #print("Converting Loop #%s"%(data))
            bboxes = bboxes_convert(instances[data])
            scores = scores_convert(instances[data])
            classes = classes_convert(instances[data])

            #print("Finished b,s,c convert")
            bboxes_filter = bboxes.tolist()
            scores_filter = scores.tolist()
            classes_filter = classes.tolist()
            pos = 0

            #print("In while loop class filter")
            #print("Class filter length = %s"%(len(classes_filter)))
            #print(classes_filter)
            while pos < len(classes_filter):
              #print("Current pos: %s"%(pos))
              if classes_filter[pos] != 0:
                    classes_filter.pop(pos)
                    bboxes_filter.pop(pos)
                    scores_filter.pop(pos)
              else:
                    pos += 1

            det_num = len(bboxes_filter)

            #print("In printing loop")
            for i in range(det_num):
                frameNum = data + 1
                cords = bboxes_filter[i]
                left = cords[0]
                top = cords[1]
                width = cords[2]-left
                height = cords[3]-top
                frameScore = scores_filter[i]
                print('%d,-1,%.3f,%.3f,%.3f,%.3f,%.6f,-1,-1,-1'%(frameNum, left, 
                                                           top, width, 
                                                           height, 
                                                           frameScore), file=outfile)
            tmp = [frameNum, -1, left, top, width, height, frameScore, -1, -1, -1]
            dets.append(tmp)

        dets = np.array(dets)
        #np.save("data/%s/det_deep")

# Running an actual video and getting original detectons from detectron2
sequence = 'PETS09-S2L1'
phase = 'train'
tot_time = 0.0
tot_frames = 0

predictor = DefaultPredictor(cfg)
# can loop seq here
OUTS = []
frmsNum = 0;
for path in os.listdir("../../Downloads/MOT15/%s/%s/img1/"%(phase,sequence)):
    frmsNum = frmsNum + 1
#frmsNum = len(glob.glob("~/Downloads/MOT15/%s/%s/img1/*.jpg"%(phase, sequence)))
print("%s has %s frames"%(sequence,frmsNum))
frm = 0
startTime = time.time()

print("Obtained Detections on %s"%(sequence))
#for f in range(frmsNum):
for f in range(frmsNum):
    f = f + 1
    print("Processing frame #%s"%(f))
    if f <= 9:
        framePath = '../../Downloads/MOT15/%s/%s/img1/00000%s.jpg'%(phase,sequence,f)
    elif f >= 10 and f <= 99:  
        framePath = '../../Downloads/MOT15/%s/%s/img1/0000%s.jpg'%(phase,sequence,f)
    elif f >= 100 and f <= 999:
        framePath = '../../Downloads/MOT15/%s/%s/img1/000%s.jpg'%(phase,sequence,f)
    else:
        framePath = '../../Downloads/MOT15/%s/%s/img1/00%s.jpg'%(phase,sequence,f)
    image = cv2.imread(framePath)
    outputs = predictor(image)
    OUTS.append(outputs['instances'])

# Run detectron output conversions
instances = OUTS
convertDetectionsSORT()
