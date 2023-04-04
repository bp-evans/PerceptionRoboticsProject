# Class imports
import sys, os
import torch, detectron2
from detectron2.utils.logger import setup_logger
setup_logger()
import numpy as np
import os, json, cv2, random
from google.colab.patches import cv2_imshow
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.utils import *
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.evaluation import *

instances = []
sequence = []
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
    frameNum = len(instances)
    dets = []

    with open("data/%s/det/det.txt"%(sequence), 'w') as outfile:
        for data in range(len(instances)):
            bboxes = bboxes_convert(instances[data])
            scores = scores_convert(instances[data])
            classes = classes_convert(instances[data])

            bboxes_filter = bboxes.tolist()
            scores_filter = scores.tolist()
            classes_filter = classes.tolist()
            pos = 0

            while pos < len(classes_filter):
              if classes_filter[pos] != 0:
                    classes_filter.pop(pos)
                    bboxes_filter.pop(pos)
                    scores_filter.pop(pos)
            else:
                pos += 1

            det_num = len(bboxes_filter)

            for i in range(det_num):
                frameNum = data + 1
                cords = bboxes_filter(i)
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


