# -*- coding: utf-8 -*-

import os
import sys
import json
import cv2
import torch
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.structures import Boxes, BoxMode 

def instances_to_coco_json(instances, img_id):
    num_instance = len(instances)
    if num_instance == 0:
        return []

    boxes = instances.pred_boxes.tensor.numpy()
    boxes = BoxMode.convert(boxes, BoxMode.XYXY_ABS, BoxMode.XYWH_ABS)
    boxes = boxes.tolist()
    scores = instances.scores.tolist()
    classes = instances.pred_classes.tolist()

    results = []
    for k in range(num_instance):
        result = {
            "image_id": img_id,
            "category_id": classes[k],
            "bbox": boxes[k],
            "score": scores[k],
        }
    
        results.append(result)
    return results


def run():
    #Путь к папке с изображениями, на которых хотим сделать предсказания. сейчас это путь к валдиационными изображениям. При загрузке вашего образа стоит использовать путь ./test/images. Папку test мы положим во время запуска контейнера.
    test_images_path = 'data/to_participants/val/images'
    #Названия файла с предсказаниями. по время локального дебага можно использовать любое имя, для сабмита в систему стоит использовать название prediction.json
    predictions_output_path = 'predict.json'
    
    
    
    
    threshold = 0.5
    model_path = "./output/model_0000999.pth"
    cpu_device = torch.device("cpu")


    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml"))
    cfg.MODEL.WEIGHTS = model_path
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = threshold   # set the testing threshold for this model
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 41
    cfg.INPUT.MIN_SIZE_TEST= 300
    cfg.INPUT.MAX_SIZE_TEST = 300
    cfg.INPUT.FORMAT = 'BGR'
    
    #cfg.MODEL.DEVICE = "cpu"
    predictor = DefaultPredictor(cfg)
    results = []
    del cfg
    for i in os.listdir(test_images_path):
        img_path =test_images_path + "/" +str(i)
        im = cv2.imread(img_path)
        outputs = predictor(im)
        instances = outputs["instances"].to(cpu_device)
        fname = int(i.split('.')[0])
        result = instances_to_coco_json(instances,fname)
        if(len(result)!=0):
            for ele in result:
                results.append(ele)    

    fp = open(predictions_output_path, "w")
    fp.write(json.dumps(results))
    fp.close()

if __name__ == "__main__":
    run()
