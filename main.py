import yaml
import os
import cv2
import argparse
import torch
from ultralytics import YOLO
from videoObjDetector import VideoObjDetector
from videoDrawer import VideoDrawer
from videoEntityCollisionDetector import VideoEntityCollisionDetector

from utils import CreateVideo

import sys
sys.path.insert(0, 'Rendering')
from vidSMPLParamCreator import PreProcessPersonData, VidSMPLParamCreator


yoloModel = None
availableObjs = None
yoloClassNameIndexMap = None
configs = None


def main(_videoPath):
    yoloModel = YOLO(configs["yolo_params"]["checkpoint_file"])

    videoDrawer = VideoDrawer(_videoPath)
    objDetectionClip = videoDrawer.CreateNewClip(configs["output_folder_dir_path"], "objDetection")
    collisionClip = videoDrawer.CreateNewClip(configs["output_folder_dir_path"], "collision")

    objDetector = VideoObjDetector(configs["deepsort_params"], [0, 32])
    objsInFrames = {}

    objCollisionChecker = VideoEntityCollisionDetector([32])
    objCollisions = {}

    currFrame = 0
    while True:
        hasFrames, vidFrameData = videoDrawer.video.read() # gives in BGR format
        if not hasFrames:
            break

        # detect + track objs
        objsInFrames[currFrame] = objDetector.DetectObjs(vidFrameData, yoloModel, configs["yolo_params"])
        # newFrame = vidFrameData.copy()
        # objDetector.Draw(newFrame, objsInFrames[currFrame])
        # objDetectionClip.write(newFrame)

        # check collision between objs and human
        # objCollisions[currFrame] = objCollisionChecker.CheckCollision(objsInFrames[currFrame])
        # newFrame = vidFrameData.copy()
        # objCollisionChecker.Draw(newFrame, objCollisions[currFrame])
        # collisionClip.write(newFrame)      

        #get proper positions of the objects against the humans

        currFrame += 1

    videoDrawer.StopVideo()
    objDetectionClip.release()
    collisionClip.release()

    # create SMPL parameters
    humans = {}
    for frameNo, objInFrame in objsInFrames.items():
        for obj in objInFrame:
            if obj.className == 0:
                if obj.id not in humans:
                    humans[obj.id] = PreProcessPersonData([], None, [], obj.id)
                
                humans[obj.id].bboxes.append(obj.bbox)
                # humans[obj.id].joints2D.append(obj.bbox)
                humans[obj.id].frames.append(frameNo)
    
    print(_videoPath)
    smplParamCreator = VidSMPLParamCreator(_videoPath, configs["vibe_params"])
    smplParamCreator.processPeopleInVid(humans.values(), videoDrawer.GetFilePath(configs["output_folder_dir_path"]))

    del smplParamCreator
    del humans




        



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
    parser.add_argument("--input", default='Input/clip_1.mp4', type=str, help="File path for video")
    parser.add_argument("--config", default='Configs/config.yaml', type=str, help="File path for config file")

    arguments = parser.parse_args()
    
    with open(arguments.config) as f:
        configs = yaml.safe_load(f)

    availableObjs = configs.get("interactable_objs", {})
    
    # loadYolo(configs["yolo_params"])
    main(arguments.input)
