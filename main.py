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
from detectedObj import HumanInteractableObject

from utils import CreateVideo, DrawSkeleton, DistBetweenPoints, DrawLineBetweenPoints, FindIndexOfValueFromSortedArray

import sys
sys.path.insert(0, 'Rendering')
from vidSMPLParamCreator import PreProcessPersonData, VidSMPLParamCreator
from lib.utils.renderer import Renderer
from lib.utils.demo_utils import (
    prepare_rendering_results,
)

# [VIBE-Object Start]
import math
import lib.vibe_obj.utils as vibe_obj
import resize
# [VIBE-Object End]

yoloModel = None
availableObjs = None
yoloClassNameIndexMap = None
configs = None

def main(args):
    videoDrawer = VideoDrawer(args.input, configs["output_folder_dir_path"])

    # Load things needed for object detection
    objsInFrames = {}
    objDetector = None
    objDetectionClip = None
    if args.detectionPKL == '':
        yoloModel = YOLO(configs["yolo_params"]["checkpoint_file"])
        objDetectionClip = videoDrawer.CreateNewClip("objDetection")
        objDetector = VideoObjDetector(configs["deepsort_params"], [0, 32])
    else:
        print("Read data from existing detection data from PKL file")
        with open(args.detectionPKL, 'rb') as f:
            objsInFrames = joblib.load(f)

    # Load things needed to check for object collision
    objCollisions = {}
    objCollisionChecker = None
    collisionClip = None
    if args.collisionDetectionPKL == '':
        collisionClip = videoDrawer.CreateNewClip("collision")
        objCollisionChecker = VideoEntityCollisionDetector([32])
    else:
        print("Read data from existing collision data from PKL file")
        with open(args.collisionDetectionPKL, 'rb') as f:
            objCollisions = joblib.load(f)

    # start detecting objects and checking for collisions in every frame
    print("Pre-process: Detect objects and possible collision between objects and humans")
    currFrame = 0
    if objDetector != None and objCollisionChecker != None:
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

            currFrame += 1
            print(f"processed frame {currFrame}/{videoDrawer.videoTotalFrames}")

    if objDetector != None:
        joblib.dump(objsInFrames, os.path.join(videoDrawer.outputPath, "detected.pkl"))
        objDetectionClip.release()
        del objDetector

    if objCollisionChecker != None:
        joblib.dump(objCollisions, os.path.join(videoDrawer.outputPath, "object_collisions.pkl"))
        collisionClip.release()
        del objCollisionChecker
    

    # create SMPL parameters
    print("Pre-process stage done, videos stored and checkpoint files created")
    humanRenderData = None
    if args.smplPKL == '':
        print("Creating SMPL parameters from humans in video for every frame")
        humans = {}

        # extract out info to feed into VIBE
        for frameNo, objInFrame in objsInFrames.items():
            for obj in objInFrame:
                if obj.className == 0:
                    if obj.id not in humans:
                        humans[obj.id] = PreProcessPersonData([], None, [], obj.id)
                    
                    humans[obj.id].bboxes.append(obj.ConvertBboxToCenterWidthHeight())
                    # humans[obj.id].joints2D.append(obj.bbox)
                    humans[obj.id].frames.append(frameNo)
        
        # Run VIBE
        smplParamCreator = VidSMPLParamCreator(args.input, configs["vibe_params"])
        humanRenderData = smplParamCreator.processPeopleInVid(humans.values(), videoDrawer.outputPath)

        del smplParamCreator
        del humans
        del objsInFrames
        print("PKL file created in output folder")
    else:
        print("Read data from existing PKL file")
        with open(args.smplPKL, 'rb') as f:
            humanRenderData = joblib.load(f)
        
        del objsInFrames


    # for the objects find the nearest point to attach to or render
    print("Computing whether object is to be attached to a person or not")
    ATTACHABLE_KEYPOINTS = set(configs["obj_keypoint_attachment_params"]["attachable_keypoints"])
    MAX_DIST_FROM_KEYPOINT = configs["obj_keypoint_attachment_params"]["max_distance"]
    objFrameAppearances = {} # {obj id: [frame number it appears in]}
    objsData = {} # {frameId: [objs that appear]}

    objAttachmentClip = videoDrawer.CreateNewClip("attachment")
    videoDrawer.ResetVideo()
    currFrame = 0
    while True:
        hasFrames, vidFrameData = videoDrawer.video.read() # gives in BGR format
        if not hasFrames:
            break

        newFrame = vidFrameData.copy()
        objsData[currFrame] = []
        for obj, objsCollidedWith in objCollisions[currFrame].items():
            shortestDist = float('inf')
            
            # check nearest distance
            for otherObj in objsCollidedWith:
                interactableObj = HumanInteractableObject.from_parent(obj)
                objsData[currFrame].append(interactableObj)
                if obj.id not in objFrameAppearances:
                    objFrameAppearances[obj.id] = []

                objFrameAppearances[obj.id].append(currFrame)
    
                # compare with humans keypoints to see whether to attach
                frameIndex = FindIndexOfValueFromSortedArray(humanRenderData[otherObj.id]["frame_ids"], currFrame) # the fact that the obj had AABB collision with the human means the human exists in this frame
                for keypt in ATTACHABLE_KEYPOINTS:
                    keyPtPos = humanRenderData[otherObj.id]["joints2d_img_coord"][frameIndex][keypt]
                    c_x, c_y, w, h = obj.ConvertBboxToCenterWidthHeight()
                    currDist = DistBetweenPoints((c_x, c_y), keyPtPos)

                    isPotentialAttachment = False
                    # TODO, HAVE TO CHECK SIZE OF BALL VIA BOUNDING BOX SIZE / 2 THEN + THRESHOLD FOR COMPARISON
                    if currDist < shortestDist and currDist <= MAX_DIST_FROM_KEYPOINT:
                        shortestDist = currDist
                        interactableObj.Attach((keyPtPos[0] - c_x, keyPtPos[1] - c_y), otherObj.id, keypt)
                        isPotentialAttachment = True

                    lineColor = (0, 255, 0) if isPotentialAttachment else (0, 0, 255)
                    DrawLineBetweenPoints(newFrame, (int(c_x), int(c_y)), (int(keyPtPos[0]), int(keyPtPos[1])), f'{currDist}', lineColor, 1)
                
                DrawSkeleton(newFrame, humanRenderData[otherObj.id]["joints2d_img_coord"][frameIndex], ATTACHABLE_KEYPOINTS)
                
        objAttachmentClip.write(newFrame)
        currFrame += 1
        print(f"processing frame {currFrame} / {videoDrawer.videoTotalFrames}")

    objAttachmentClip.release()
    del objCollisions
    print("Computation for object's attachment is completed")


    # TODO: If object has missing frames, clean up data
    # check missing frames objFrameAppearances
    # MAX_MISSING_FRAMES = configs["obj_smoothing"]["max_missing_frames"]
    # for objId, frameAppearances in objFrameAppearances.items():

    #     for i in range(1, len(frameAppearances)):
    #         frameDiff = frameAppearances[i] - frameAppearances[i - 1]

    #         # no missing frames
    #         if frameDiff == 0:
    #             continue

    #         # went over threshold, most likely left the frame
    #         if frameDiff > MAX_MISSING_FRAMES:
    #             continue

    #         lastFrame = frameAppearances[i - 1]

    '''
        Various different states:
            both no grab: interpolate positions via kinematic equation
            both grab + same person id: frames in between we will assume that it is still attached to same person
            both grab + diff person id: idk fuckin pray??
                the ball left first person temporarily and second person grab
                first person passed to second person

            1 grab, the other does not: have to interpolate between frames and based on dist, predict when its still attached and when its not

        Based on the states, add the missing data

    '''
    
    #for r in range (frameDiff):




    # render the objects and humans
    print("Render Objects and humans")
    # objData = {key: [obj for obj in objs if obj.className != 0] for key, objs in objsInFrames.items()}
    # print(videoDrawer)
    # print(objData)

    render(videoDrawer, humanRenderData, objsData)
    videoDrawer.StopVideo()
    print("Render done")


def TEST_PKL(args):
    with open(args.smplPKL, 'rb') as f:
        humanRenderData = joblib.load(f)

    whiteBackground = np.full((1500, 2000, 3), 255, dtype=np.uint8)
    DrawSkeleton(whiteBackground, humanRenderData[1]["joints2d_img_coord"][0])

    cv2.imshow('Image', whiteBackground)
    cv2.waitKey(0)

'''
    arguments:
        _humanRenderData: [{
            bboxes = [[cx, cy, h, w] ...]
            joints2D = []
            frames = [frame number the person appears in]
            id = person identification number
        }...] Array of people objects

        _objRenderData: [
            HumanInteractableObject
        ]
'''
def render(_videoInfo, _humanRenderData = None, _objRenderData = None):
    # for loop the objs in frame, render it
    renderer = Renderer(resolution=(_videoInfo.videoWidth, _videoInfo.videoHeight), orig_img=True, wireframe=False, renderOnWhite=True)

    # dictionary, {frameNo: {humanId: verts, cam, joints3D, pose} }
    frame_results = prepare_rendering_results(_humanRenderData, _videoInfo.videoTotalFrames)
    mesh_color = {k: colorsys.hsv_to_rgb(np.random.rand(), 0.5, 1.0) for k in _humanRenderData.keys()}

    renderClip = _videoInfo.CreateNewClip("render")
    _videoInfo.ResetVideo()

    fov = math.radians(60)
    pos_x = 0.0
    pos_y = 0.0
    pos_z = 0.0
    aspect_ratio = _videoInfo.videoWidth/_videoInfo.videoHeight
    frameIndex = 0
    while True:
        hasFrames, img = _videoInfo.video.read() # gives in BGR format
        if not hasFrames:
            break

        renderer.push_persp_cam(fov)

        # # render people in video
        for person_id, person_data in frame_results[frameIndex].items():
            frame_verts = person_data['verts']
            frame_cam = person_data['cam']
            # [VIBE-Object Start]
            frame_joints3d = person_data['joints3d']
            frame_pose = person_data['pose']
            # [VIBE-Object End]

            tan_half_fov = math.tan(fov * 0.5)
            sx, sy, tx, ty = frame_cam
            pos_z = -1.0 / (sy * tan_half_fov)
            pos_y = -ty
            pos_x = tx
            
            mc = mesh_color[person_id]
            renderer.push_human(verts=frame_verts, # Add human to scene.
                                color=mc,
                                translation=[pos_x, pos_y, pos_z])

        for obj in _objRenderData[frameIndex]:
            objCxCy = obj.ConvertBboxToCenterWidthHeight()
            obj_scale = resize.get_world_height(objCxCy[3], _videoInfo.videoHeight, fov, pos_z)
            
            # Map the screen coordinate to NDC, which is [-1, 1].
            screen_x = -(objCxCy[0] / _videoInfo.videoWidth * 2.0 - 1.0)
            screen_y = objCxCy[1] / _videoInfo.videoHeight * 2.0 - 1.0

            # Convert from NDC to world coordinate.
            obj_x = screen_x * pos_z * math.tan(0.5 * fov) * aspect_ratio
            obj_y = screen_y * pos_z * math.tan(0.5 * fov)
            obj_z = pos_z

            renderer.push_obj(
                '3D_Models/sphere.obj',
                translation=[obj_x, obj_y, obj_z],
                angle=0,
                axis=[0,0,0],
                scale=[obj_scale, obj_scale, obj_scale],
                color=[0.05, 1.0, 1.0],
            )

        img = renderer.pop_and_render(img) # append human into img
        frameIndex += 1
        renderClip.write(img)
        print(f"processed render for frame {frameIndex}/{_videoInfo.videoTotalFrames}")

    renderClip.release()


'''     
def TEST_render_obj(IMAGE_FRAME, X, Y):
    renderer = Renderer(resolution=(IMAGE_FRAME.videoWidth, IMAGE_FRAME.videoHeight), orig_img=True, wireframe=False, renderOnWhite=True)

    renderer.push_default_cam()
    location = renderer.screenspace_to_worldspace(X, Y)

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
'''

if __name__ == "__main__":
    refresh = False
    video_name = "PassBallTwoHands"

    parser = argparse.ArgumentParser(description="Your application's description")
    if refresh == True:
        parser.add_argument("--input", default='Input/'+ video_name + '.mp4', type=str, help="File path for video")
        parser.add_argument("--config", default='Configs/config.yaml', type=str, help="File path for config file")
        parser.add_argument("--smplPKL", default='', type=str, help="Pre-processed Pkl file containing smpl data of the video")
        parser.add_argument("--detectionPKL", default='', type=str, help="Pre-processed Pkl file containing smpl data of the video")
        parser.add_argument("--collisionDetectionPKL", default='', type=str, help="Pre-processed Pkl file containing smpl data of the video")
    else:
        parser.add_argument("--input", default='Input/'+ video_name + '.mp4', type=str, help="File path for video")
        parser.add_argument("--config", default='Configs/config.yaml', type=str, help="File path for config file")
        parser.add_argument("--smplPKL", default='Output/' + video_name + '/vibe_output.pkl', type=str, help="Pre-processed Pkl file containing smpl data of the video")
        parser.add_argument("--detectionPKL", default='Output/' + video_name + '/detected.pkl', type=str, help="Pre-processed Pkl file containing smpl data of the video")
        parser.add_argument("--collisionDetectionPKL", default='Output/' + video_name + '/object_collisions.pkl', type=str, help="Pre-processed Pkl file containing smpl data of the video")

    arguments = parser.parse_args()
    
    with open(arguments.config) as f:
        configs = yaml.safe_load(f)

    availableObjs = configs.get("interactable_objs", {})
    
    main(arguments)
    #TEST_PKL(arguments)
    # videoDrawer = VideoDrawer("Input/video11.mp4", configs["output_folder_dir_path"])
    # TEST_render_obj(videoDrawer, 0, 0)