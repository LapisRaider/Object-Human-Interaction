import yaml
import os
import cv2
import argparse
import torch
from ultralytics import YOLO
from videoObjDetector import VideoObjDetector
from videoDrawer import VideoDrawer
from videoEntityCollisionDetector import VideoEntityCollisionDetector
import colorsys
import numpy as np
import pickle
import joblib

from utils import CreateVideo

import sys
sys.path.insert(0, 'Rendering')
from vidSMPLParamCreator import PreProcessPersonData, VidSMPLParamCreator
from lib.utils.renderer import Renderer
from lib.utils.demo_utils import (
    prepare_rendering_results,
)

yoloModel = None
availableObjs = None
yoloClassNameIndexMap = None
configs = None

def main(args):
    videoDrawer = VideoDrawer(args.input, configs["output_folder_dir_path"])

    objsInFrames = {}
    objDetector = None
    objDetectionClip = None
    collisionClip = None

    if args.detectionPKL == '':
        yoloModel = YOLO(configs["yolo_params"]["checkpoint_file"])
        objDetectionClip = videoDrawer.CreateNewClip("objDetection")
        objDetector = VideoObjDetector(configs["deepsort_params"], [0, 32])
    else:
        with open(args.detectionPKL, 'rb') as f:
            objsInFrames = pickle.load(f)

    objCollisions = {}
    objCollisionChecker = None
    if args.collisionDetectionPKL == '':
        collisionClip = videoDrawer.CreateNewClip("collision")
        objCollisionChecker = VideoEntityCollisionDetector([32])
    else:
        with open(args.collisionDetectionPKL, 'rb') as f:
            objCollisions = pickle.load(f)

    print("Pre-process: Detect objects and possible collision between objects and humans")
    currFrame = 0
    while True:
        hasFrames, vidFrameData = videoDrawer.video.read() # gives in BGR format
        if not hasFrames:
            break

        # detect + track objs
        if objDetector != None:
            objsInFrames[currFrame] = objDetector.DetectObjs(vidFrameData, yoloModel, configs["yolo_params"])
            newFrame = vidFrameData.copy()
            objDetector.Draw(newFrame, objsInFrames[currFrame])
            objDetectionClip.write(newFrame)

        # check collision between objs and human
        if objCollisionChecker != None:
            objCollisions[currFrame] = objCollisionChecker.CheckCollision(objsInFrames[currFrame])
            newFrame = vidFrameData.copy()
            objCollisionChecker.Draw(newFrame, objCollisions[currFrame])
            collisionClip.write(newFrame)

        #get proper positions of the objects against the humans
        currFrame += 1
        print(f"processed frame {currFrame}/{videoDrawer.videoTotalFrames}")

    if objDetector != None:
        joblib.dump(objsInFrames, os.path.join(videoDrawer.outputPath, "detected.pkl"))
        objDetectionClip.release()

    if objCollisionChecker != None:
        joblib.dump(objCollisions, os.path.join(videoDrawer.outputPath, "object_collisions.pkl"))
        collisionClip.release()

    videoDrawer.StopVideo()
    

    # create SMPL parameters
    print("Pre-process stage done, videos stored and checkpoint files created")
    print("Creating SMPL parameters from humans in video for every frame")
    humanRenderData = None
    if args.smplPKL == '':
        humans = {}
        for frameNo, objInFrame in objsInFrames.items():
            for obj in objInFrame:
                if obj.className == 0:
                    if obj.id not in humans:
                        humans[obj.id] = PreProcessPersonData([], None, [], obj.id)
                    
                    humans[obj.id].bboxes.append(obj.ConvertBboxToCenterWidthHeight())
                    # humans[obj.id].joints2D.append(obj.bbox)
                    humans[obj.id].frames.append(frameNo)
        
        smplParamCreator = VidSMPLParamCreator(args.input, configs["vibe_params"])
        humanRenderData = smplParamCreator.processPeopleInVid(humans.values(), videoDrawer.outputPath)

        del smplParamCreator
        del humans
        print("PKL file created in output folder")
    else:
        with open(args.smplPKL, 'rb') as f:
            humanRenderData = pickle.load(f)

    # read the parameters of each human
    # for the objects find the nearest point to attach to or render

    # render the objects and humans
    print("Render Objects and humans")
    objData = {key: [obj for obj in objs if obj.className != 0] for key, objs in objsInFrames.items()}
    print(videoDrawer)
    print(objData)

    render(videoDrawer, humanRenderData, objData)
    print("Render done")


'''
    arguments:
        _humanRenderData: [{
            bboxes = [[cx, cy, h, w] ...]
            joints2D = []
            frames = [frame number the person appears in]
            id = person identification number
        }...] Array of people objects

        _objRenderData: [
        

        ]
'''
def render(_videoInfo, _humanRenderData = None, _objRenderData = None):
    # for loop the objs in frame, render it
    renderer = Renderer(resolution=(_videoInfo.videoWidth, _videoInfo.videoHeight), orig_img=True, wireframe=False, renderOnWhite=True)

    # dictionary, {frameNo: {humanId: verts, cam, joints3D, pose} }
    frame_results = prepare_rendering_results(_humanRenderData, _videoInfo.videoTotalFrames)
    mesh_color = {k: colorsys.hsv_to_rgb(np.random.rand(), 0.5, 1.0) for k in _humanRenderData.keys()}

    renderClip = _videoInfo.CreateNewClip("render")

    for frameIndex in range(0, _videoInfo.videoTotalFrames):
        img = None

        # render people in video 
        for person_id, person_data in frame_results[frameIndex].items():
            frame_verts = person_data['verts']
            frame_cam = person_data['cam']
            # [VIBE-Object Start]
            frame_joints3d = person_data['joints3d']
            frame_pose = person_data['pose']
            # [VIBE-Object End]

            mc = mesh_color[person_id]
            renderer.push_cam(frame_cam) # Add human camera to scene.
            renderer.push_human(verts=frame_verts, color=mc) # Add human to scene.

            img = renderer.pop_and_render(img) # append human into img

        # obj to render
        for obj in _objRenderData[frameIndex]:
            objCxCy = obj.ConvertBboxToCenterWidthHeight()
            
            renderer.push_default_cam()
            print(objCxCy)
            location = renderer.screenspace_to_worldspace(objCxCy[0], objCxCy[1])

            renderer.push_obj(
                '3D_Models/sphere.obj',
                translation= [location[0], location[1], 1],
                angle=0,
                axis=[0,0,0],
                scale=[0.2, 0.2, 0.2],
                color=[1.0, 0.0, 0.0],
            )

            img = renderer.pop_and_render(img) # append obj to img

        renderClip.write(img)
        print(f"processed render for frame {frameIndex}/{_videoInfo.videoTotalFrames}")

    renderClip.release()

        
def render_obj(_videoInfo):
    renderer = Renderer(resolution=(_videoInfo.videoWidth, _videoInfo.videoHeight), orig_img=True, wireframe=False, renderOnWhite=True)

    renderer.push_default_cam()
    location = renderer.screenspace_to_worldspace(0, 0)

    print("World space location")
    print(location)
    renderer.push_obj(
        '3D_Models/sphere.obj',
        translation= [location[0], location[1], 1],
        angle=0,
        axis=[0,0,0],
        scale=[0.2, 0.2, 0.2],
        color=[1.0, 0.0, 0.0],
    )

    img = renderer.pop_and_render()
    cv2.imshow('Image', img)
    cv2.waitKey(0)



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
    parser.add_argument("--input", default='Input/video11.mp4', type=str, help="File path for video")
    parser.add_argument("--config", default='Configs/config.yaml', type=str, help="File path for config file")
    parser.add_argument("--smplPKL", default='Output/video11/vibe_output.pkl', type=str, help="Pre-processed Pkl file containing smpl data of the video")
    parser.add_argument("--detectionPKL", default='Output/video11/detected.pkl', type=str, help="Pre-processed Pkl file containing smpl data of the video")
    parser.add_argument("--collisionDetectionPKL", default='Output/video11/object_collisions.pkl', type=str, help="Pre-processed Pkl file containing smpl data of the video")


    arguments = parser.parse_args()
    
    with open(arguments.config) as f:
        configs = yaml.safe_load(f)

    availableObjs = configs.get("interactable_objs", {})
    
    # loadYolo(configs["yolo_params"])
    main(arguments)
    # videoDrawer = VideoDrawer("Input/video11.mp4", configs["output_folder_dir_path"])
    # render_obj(videoDrawer)
    

