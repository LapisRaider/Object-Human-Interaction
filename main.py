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
from detectedObj import HumanInteractableObject, Coordinates

from utils import DrawSkeleton, DistBetweenPoints, DrawLineBetweenPoints, FindIndexOfValueFromSortedArray, ConvertBboxToCenterWidthHeight, DrawTextOnTopRight

import sys
sys.path.insert(0, 'Rendering')
from vidSMPLParamCreator import PreProcessPersonData, VidSMPLParamCreator
from lib.utils.renderer import Renderer
from lib.utils.demo_utils import (
    prepare_rendering_results,
)
from lib.data_utils.kp_utils import map_spin_to_smpl
from lib.vibe_obj.utils import get_rotation

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

    # start detecting objects and checking for collisions in every frame
    print("Pre-process: Detect objects and remove possible false positives")
    currFrame = 0
    if objDetector != None:
        objDetectedFrames = {}

        while True:
            hasFrames, vidFrameData = videoDrawer.video.read() # gives in BGR format
            if not hasFrames:
                break

            # detect + track objs
            objsInFrames[currFrame] = objDetector.DetectObjs(vidFrameData, yoloModel, configs["yolo_params"])
            
            # update the number of frames the obj appeared in
            for obj in objsInFrames[currFrame]:
                if obj.id not in objDetectedFrames:
                    objDetectedFrames[obj.id] = 0

                objDetectedFrames[obj.id] = objDetectedFrames[obj.id] + 1

            currFrame += 1
            print(f"Object detection: frame {currFrame}/{videoDrawer.videoTotalFrames}")

        # for objs that have very little frame
        print(f'{objDetectedFrames} object frame counts')
        invalidObjIds = set()
        for objId, frameCount in objDetectedFrames.items():
            if frameCount < configs["objs_data"]["min_frame_appearances"]:
                invalidObjIds.add(objId)

        print(f'{invalidObjIds} are removed as they appear in too little frames')
        
        # filter out the invalid objects and draw the detected information
        currFrame = 0
        videoDrawer.ResetVideo()
        while True:
            hasFrames, vidFrameData = videoDrawer.video.read() # gives in BGR format
            if not hasFrames:
                break
        
            delObjs = [obj for obj in objsInFrames[currFrame] if obj.id in invalidObjIds]
            objsInFrames[currFrame] = [obj for obj in objsInFrames[currFrame] if obj.id not in invalidObjIds]

            # draw the detected information
            newFrame = vidFrameData.copy()
            objDetector.Draw(newFrame, objsInFrames[currFrame])
            # objDetector.Draw(newFrame, delObjs, False)
            DrawTextOnTopRight(newFrame, f"{currFrame}/{videoDrawer.videoTotalFrames}",  videoDrawer.videoWidth)
            objDetectionClip.write(newFrame)

            currFrame += 1

        joblib.dump(objsInFrames, os.path.join(videoDrawer.outputPath, "detected.pkl"))
        objDetectionClip.release()
        del objDetector
        del invalidObjIds
        del objDetectedFrames

    print("Finish detection of objects + post processing")

   
    # check for collisions between objects
    print("Check collision between humans and objects")
    videoDrawer.ResetVideo()
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
    
    if objCollisionChecker != None:
        currFrame = 0
        while True:
            hasFrames, vidFrameData = videoDrawer.video.read() # gives in BGR format
            if not hasFrames:
                break
    
            objCollisions[currFrame] = objCollisionChecker.CheckCollision(objsInFrames[currFrame])
            newFrame = vidFrameData.copy()
            objCollisionChecker.Draw(newFrame, objCollisions[currFrame])
            DrawTextOnTopRight(newFrame, f"{currFrame}/{videoDrawer.videoTotalFrames}",  videoDrawer.videoWidth)
            collisionClip.write(newFrame)

            currFrame += 1

        # dump info
        joblib.dump(objCollisions, os.path.join(videoDrawer.outputPath, "object_collisions.pkl"))
        collisionClip.release()
        del objCollisionChecker
    
    print("Collision detection completed")
    

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
                    
                    # we use the originalBbox as the sizes are much more accurate
                    humans[obj.id].bboxes.append(ConvertBboxToCenterWidthHeight(obj.originalBbox))
                    # humans[obj.id].joints2D.append(obj.originalBbox)
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

    if args.objKeyPtRawData == '':
        objsData = {} # {frameId: [objs that appear]}
        ATTACHABLE_KEYPOINTS = set(configs["obj_keypoint_attachment_params"]["attachable_keypoints"])
        MAX_DIST_FROM_KEYPOINT = configs["obj_keypoint_attachment_params"]["max_distance"]

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
                interactableObj = HumanInteractableObject.from_parent(obj)
                objsData[currFrame].append(interactableObj)
                
                # check nearest distance
                for otherObj in objsCollidedWith:

                    # compare with humans keypoints to see whether to attach
                    frameIndex = FindIndexOfValueFromSortedArray(humanRenderData[otherObj.id]["frame_ids"], currFrame) # the fact that the obj had AABB collision with the human means the human exists in this frame
                    for keypt in ATTACHABLE_KEYPOINTS:
                        keyPtPos = humanRenderData[otherObj.id]["joints2d_img_coord"][frameIndex][keypt]
                        currDist = DistBetweenPoints((interactableObj.renderPoint[0], interactableObj.renderPoint[1]), keyPtPos)

                        isPotentialAttachment = False
                        # TODO, HAVE TO CHECK SIZE OF BALL VIA BOUNDING BOX SIZE / 2 THEN + THRESHOLD FOR COMPARISON
                        # need to take note for objects that does not have the same width and height
                        if currDist < shortestDist and currDist <= MAX_DIST_FROM_KEYPOINT:
                            shortestDist = currDist
                            interactableObj.Attach(otherObj.id, keypt, (interactableObj.renderPoint[0] - keyPtPos[0], interactableObj.renderPoint[1] - keyPtPos[1]))
                            isPotentialAttachment = True

                        lineColor = (0, 255, 0) if isPotentialAttachment else (0, 0, 255)
                        DrawLineBetweenPoints(newFrame, (int(interactableObj.renderPoint[0]), int(interactableObj.renderPoint[1])), (int(keyPtPos[0]), int(keyPtPos[1])), f'{currDist}', lineColor, 1)
                    
                    DrawSkeleton(newFrame, humanRenderData[otherObj.id]["joints2d_img_coord"][frameIndex], ATTACHABLE_KEYPOINTS)
            
            DrawTextOnTopRight(newFrame, f"{currFrame}/{videoDrawer.videoTotalFrames}",  videoDrawer.videoWidth)
            objAttachmentClip.write(newFrame)
            currFrame += 1
            print(f"processing frame {currFrame} / {videoDrawer.videoTotalFrames}")

        objAttachmentClip.release()
        joblib.dump(objsData, os.path.join(videoDrawer.outputPath, "obj_kpt_attachment.pkl"))
    else:
        print("Read data from existing PKL file")
        with open(args.objKeyPtRawData, 'rb') as f:
            objsData = joblib.load(f)
    
    del objCollisions
    
    print("Computation for object's attachment is completed")


    # render the objects and humans
    print("Render Objects and humans")
    render(videoDrawer, humanRenderData, objsData, configs["render_output"])
    videoDrawer.StopVideo()
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
            HumanInteractableObject
        ]
'''
def render(_videoInfo, _humanRenderData, _objRenderData, _renderConfigs):
    # for loop the objs in frame, render it
    renderer = Renderer(resolution=(_videoInfo.videoWidth, _videoInfo.videoHeight), orig_img=_renderConfigs["renderOnOriVid"], wireframe=_renderConfigs["isWireframe"], renderOnWhite=True)

    # dictionary, {frameNo: {humanId: verts, cam, joints3D, pose} }
    frame_results = prepare_rendering_results(_humanRenderData, _videoInfo.videoTotalFrames)
    mesh_color = {k: colorsys.hsv_to_rgb(np.random.rand(), 0.5, 1.0) for k in _humanRenderData.keys()}

    renderClip = _videoInfo.CreateNewClip("render")
    _videoInfo.ResetVideo()

    fov = math.radians(60)
    frameIndex = 0
    objCoordinates = {} # key: obj Id, value: Coordinates object
    while True:
        hasFrames, img = _videoInfo.video.read() # gives in BGR format
        if not hasFrames:
            break

        renderer.push_persp_cam(fov)

        # render people in video
        peopleCoordinates = {}
        for person_id, person_data in frame_results[frameIndex].items():
            frame_verts = person_data['verts']
            frame_cam = person_data['cam']
            # [VIBE-Object Start]
            frame_joints3d = person_data['joints3d']
            # [VIBE-Object End]

            tan_half_fov = math.tan(fov * 0.5)
            sx, sy, tx, ty = frame_cam
            pos_z = -1.0 / (sy * tan_half_fov)
            pos_y = -ty
            pos_x = tx

            peopleCoordinates[person_id] = Coordinates(pos_x, pos_y, pos_z)
            
            mc = mesh_color[person_id]
            renderer.push_human(verts=frame_verts, # Add human to scene.
                                color=mc,
                                translation=[pos_x, pos_y, pos_z])

        # render objs in video
        for obj in _objRenderData[frameIndex]:
            pos_z = 1.0 if obj.id not in objCoordinates else objCoordinates[obj.id].z
            obj_screenX = obj.renderPoint[0]
            obj_screenY = obj.renderPoint[1]
            offset = [0, 0 , 0]
            axis = [0, 0, 0]
            angle = 0

            # if object is attached to a person initialize values from parenting
            if obj.isAttached and obj.attachedToObjId in peopleCoordinates:
                pos_z = peopleCoordinates[obj.attachedToObjId].z

                smpl_pose = frame_results[frameIndex][obj.attachedToObjId]['pose']
                axis_angle = get_rotation(smpl_pose, map_spin_to_smpl()[obj.boneAttached]) 
                axis = axis_angle[1:4]
                angle = axis_angle[0] * (180.0/math.pi)

                # overall screen x and y coordinates are the same, but the transformation order is different
                offset = [obj.offset[0], obj.offset[1], 0]
                # TODO: now using 2d keypoint position instead of smpl 3d position, might want to change
                humanFrameIndex = FindIndexOfValueFromSortedArray(_humanRenderData[obj.attachedToObjId]["frame_ids"], frameIndex)
                joint2DCoords = _humanRenderData[obj.attachedToObjId]["joints2d_img_coord"][humanFrameIndex][obj.boneAttached]
                jointWorldCoords = renderer.screenspace_to_worldspace(joint2DCoords[0], joint2DCoords[1], pos_z)
                objWorldCoords = renderer.screenspace_to_worldspace(obj_screenX, obj_screenY, pos_z)
                
                offset = [objWorldCoords[0] - jointWorldCoords[0], objWorldCoords[1] - jointWorldCoords[1], 0]
                obj_screenX, obj_screenY = joint2DCoords
                

            obj_scale = resize.get_world_height(obj.width, _videoInfo.videoHeight, fov, pos_z)
            obj_x, obj_y, obj_z = renderer.screenspace_to_worldspace(obj_screenX, obj_screenY, pos_z)

            objCoordinates[obj.id] = Coordinates(obj_x, obj_y, obj_z)

            renderer.push_obj(
                '3D_Models/sphere.obj',
                translation_offset=offset,
                translation=[obj_x, obj_y, obj_z],
                angle=angle,
                axis=axis,
                scale=[obj_scale, obj_scale, obj_scale],
                color=[0.05, 1.0, 1.0],
            )
        
        del peopleCoordinates

        img = renderer.pop_and_render(img) # append human into img
        frameIndex += 1
        DrawTextOnTopRight(img, f"{frameIndex}/{_videoInfo.videoTotalFrames}",  _videoInfo.videoWidth)
        renderClip.write(img)
        print(f"processed render for frame {frameIndex}/{_videoInfo.videoTotalFrames}")

    renderClip.release()


if __name__ == "__main__":
    refresh = True
    video_name = "PassBallTwoHands"

    parser = argparse.ArgumentParser(description="Your application's description")
    if refresh == True:
        parser.add_argument("--input", default='Input/'+ video_name + '.mp4', type=str, help="File path for video")
        parser.add_argument("--config", default='Configs/config.yaml', type=str, help="File path for config file")
        parser.add_argument("--smplPKL", default='', type=str, help="Pre-processed Pkl file containing smpl data of the video")
        parser.add_argument("--detectionPKL", default='', type=str, help="Pre-processed Pkl file containing smpl data of the video")
        parser.add_argument("--collisionDetectionPKL", default='', type=str, help="Pre-processed Pkl file containing smpl data of the video")
        parser.add_argument("--objKeyPtRawData", default='', type=str, help="Pre-processed Pkl file containing smpl data of the video")
    else:
        parser.add_argument("--input", default='Input/'+ video_name + '.mp4', type=str, help="File path for video")
        parser.add_argument("--config", default='Configs/config.yaml', type=str, help="File path for config file")
        parser.add_argument("--smplPKL", default='Output/' + video_name + '/vibe_output.pkl', type=str, help="Pre-processed Pkl file containing smpl data of the video")
        parser.add_argument("--detectionPKL", default='Output/' + video_name + '/detected.pkl', type=str, help="Pre-processed Pkl file containing smpl data of the video")
        parser.add_argument("--collisionDetectionPKL", default='Output/' + video_name + '/object_collisions.pkl', type=str, help="Pre-processed Pkl file containing smpl data of the video")
        parser.add_argument("--objKeyPtRawData", default='Output/' + video_name + '/obj_kpt_attachment.pkl', type=str, help="Pre-processed Pkl file containing smpl data of the video")

    arguments = parser.parse_args()
    
    with open(arguments.config) as f:
        configs = yaml.safe_load(f)

    availableObjs = configs.get("interactable_objs", {})
    
    main(arguments)