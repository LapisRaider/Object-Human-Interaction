import yaml
import os
import cv2
import argparse
import torch
from ultralytics import YOLO
from videoObjDetector import VideoObjDetector

yoloModel = None
availableObjs = None
yoloClassNameIndexMap = None
configs = None


def main(_videoPath):
    yoloModel = YOLO(configs["yolo_params"]["checkpoint_file"])

    objDetector = VideoObjDetector(_videoPath, [0])
    objDetector.DetectObjs(yoloModel, configs["yolo_params"], configs["deepsort_params"])



def loadYolo(_yoloParams):
    yoloModel = YOLO(_yoloParams["checkpoint_file"])

    with open(_yoloParams["dataset_file"], 'r', encoding='utf-8') as file:
        yoloDataSetConfigs = yaml.safe_load(file)

    yoloClassNameIndexMap = {name: idx for idx, name in yoloDataSetConfigs["names"].items()}

    classesIdToDetect = [yoloClassNameIndexMap.get(key) for key in availableObjs.keys() if key in yoloClassNameIndexMap]
    classesIdToDetect.append(yoloClassNameIndexMap.get('person'))

    results = yoloModel.track(source="https://youtu.be/UeiNdPaQ1IA?si=U3WrVcY4OXfPdnVt", 
                              show=True, 
                              conf=_yoloParams["confidence_score"],
                              iou=_yoloParams["intersection_over_union"],
                              device=_yoloParams["device"],
                              tracker="Data/bytetrack.yaml",
                              classes=classesIdToDetect)


def drawBoundary():
    print("DRAW")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Your application's description")
    parser.add_argument("--input", default='Input/video1.mp4', type=str, help="File path for video")
    parser.add_argument("--config", default='Configs/config.yaml', type=str, help="File path for config file")

    arguments = parser.parse_args()
    
    with open(arguments.config) as f:
        configs = yaml.safe_load(f)

    availableObjs = configs.get("interactable_objs", {})
    
    # loadYolo(configs["yolo_params"])
    main(arguments.input)
