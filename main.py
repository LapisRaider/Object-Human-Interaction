import yaml
import os
import argparse
from ultralytics import YOLO

yoloModel = None
availableObjs = None

def main(_videoPath):
    print("hello world")

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

    yoloModel = YOLO(config["yolo_params"]["checkpoint_file"])
    availableObjs = config.get("interactable_objs", {})

    results = yoloModel.track(source="https://youtu.be/UeiNdPaQ1IA?si=U3WrVcY4OXfPdnVt", show=True)

    main(arguments.input)
