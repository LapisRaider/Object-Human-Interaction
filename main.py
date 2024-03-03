import yaml
import os
import argparse
import torch
from ultralytics import YOLO

yoloModel = None
availableObjs = None
yoloClassNameIndexMap = None


def main(_videoPath):
    print("hello")

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
                              classes=classesIdToDetect)


def drawBoundary():
    print("DRAW")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Your application's description")
    parser.add_argument("--input", type=str, help="File path for video")
    parser.add_argument("--config", type=str, help="File path for config file")

    arguments = parser.parse_args()
    
    config = None
    with open(arguments.config) as f:
        config = yaml.safe_load(f)

    availableObjs = config.get("interactable_objs", {})
    

    loadYolo(config["yolo_params"])
    main(arguments.input)
