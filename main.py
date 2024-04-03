import math
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
from tqdm import tqdm
import pickle
import joblib
from detectedObj import HumanInteractableObject, ObjectTransformation, DetectedObj
from typing import Dict, List

from utils import DrawSkeleton, DistBetweenPoints, DrawLineBetweenPoints, FindIndexOfValueFromSortedArray, ConvertBboxToCenterWidthHeight, DrawTextOnTopRight

import sys
sys.path.insert(0, 'Rendering')
from vidSMPLParamCreator import PreProcessPersonData, VidSMPLParamCreator
from lib.utils.demo_utils import (
    prepare_rendering_results,
)
from lib.utils.pose_tracker import run_posetracker

import resize_util
import skeleton_util
from renderer import Renderer

import maths_util
import mtx_util
from vec3 import Vec3
from mtx import Mtx
from quat import Quat

yoloModel = None
availableObjs = None
yoloClassNameIndexMap = None
configs = None

def main(args):
    videoDrawer = VideoDrawer(args.input, configs["output_folder_dir_path"])

    # Load things needed for object detection
    objsInFrames: Dict[int, Dict[int, DetectedObj]] = {}
    objDetector = None
    objDetectionClip = None
    if args.detectionPKL == '':
        yoloModel = YOLO(configs["yolo_params"]["checkpoint_file"])
        objDetectionClip = videoDrawer.CreateNewClip("objDetection")
        objsId = [configs["human_id"]] + list(configs["interactable_objs"].keys())
        objDetector = VideoObjDetector(configs["tracking_params"], objsId)
    else:
        print("Read data from existing detection data from PKL file")
        with open(args.detectionPKL, 'rb') as f:
            objsInFrames = joblib.load(f)

    # start detecting objects and checking for collisions in every frame
    print("Pre-process: Detect objects and remove possible false positives")
    currFrame = 0
    if objDetector != None:
        objDetectedFrames : Dict[int, List[int]] = {}

        while True:
            hasFrames, vidFrameData = videoDrawer.video.read() # gives in BGR format
            if not hasFrames:
                break

            # detect + track objs
            objsInFrames[currFrame] = objDetector.DetectObjs(vidFrameData, yoloModel, configs["yolo_params"])
            
            # update the number of frames the obj appeared in
            for obj in objsInFrames[currFrame].values():
                if obj.id not in objDetectedFrames:
                    objDetectedFrames[obj.id] = []

                objDetectedFrames[obj.id].append(currFrame)

            currFrame += 1
            print(f"Object detection: frame {currFrame}/{videoDrawer.videoTotalFrames}")

        # for objs that have very little frame
        invalidObjIds = set()
        for objId, frameAppearances in objDetectedFrames.items():
            if len(frameAppearances) < configs["objs_data"]["min_frame_appearances"]:
                invalidObjIds.add(objId)

        print(f'{invalidObjIds} are removed as they appear in only {len(frameAppearances)} frames')

        # Linear interpolation
        if configs["tracking_params"]["useLinearInterpolation"]:
            for objId, objFrameAppearances in objDetectedFrames.items():

                for index in range(1, len(objFrameAppearances)):
                    prevFrameIndex = objFrameAppearances[index - 1]
                    currFrameIndex = objFrameAppearances[index]

                    frameOffset: int = currFrameIndex - prevFrameIndex
                    # if it is missing a frame and number of missing frames is less than a certain amt
                    if frameOffset > 1 and frameOffset <= configs["tracking_params"]["max_missing_frames"]:
                        objPrevBBox = objsInFrames[prevFrameIndex][objId].bbox
                        objCurrBBox = objsInFrames[currFrameIndex][objId].bbox

                        objPrevPos = ConvertBboxToCenterWidthHeight(objPrevBBox)[:2]
                        objCurrPos =  ConvertBboxToCenterWidthHeight(objCurrBBox)[:2]

                        # get speed per frame via displacement moved / frames between displacement
                        dir = [(objCurrPos[0] - objPrevPos[0]) / frameOffset, (objCurrPos[1] - objPrevPos[1]) / frameOffset]

                        for frameNumber in range(objFrameAppearances[index - 1] + 1, objFrameAppearances[index]):
                            cloneObj = DetectedObj.clone(objsInFrames[frameNumber - 1][objId])
                            cloneObj.applyOffset(dir[0], dir[1])
                            objsInFrames[frameNumber][objId] = cloneObj

        # filter out the invalid objects and draw the detected information
        currFrame = 0
        videoDrawer.ResetVideo()
        while True:
            hasFrames, vidFrameData = videoDrawer.video.read() # gives in BGR format
            if not hasFrames:
                break
        
            delObjs = {objId: obj for objId, obj in objsInFrames[currFrame].items() if objId in invalidObjIds}
            objsInFrames[currFrame] = {objId: obj for objId, obj in objsInFrames[currFrame].items() if objId not in invalidObjIds}

            # draw the detected information
            newFrame = vidFrameData.copy()
            objDetector.Draw(newFrame, objsInFrames[currFrame].values())
            objDetector.Draw(newFrame, delObjs.values(), False)
            DrawTextOnTopRight(newFrame, f"{currFrame}/{videoDrawer.videoTotalFrames}",  videoDrawer.videoWidth)
            objDetectionClip.write(newFrame)

            currFrame += 1

        joblib.dump(objsInFrames, os.path.join(videoDrawer.outputPath, "detected.pkl"))
        objDetectionClip.release()
        del invalidObjIds
        del objDetector
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
        objCollisionChecker = VideoEntityCollisionDetector(list(configs["interactable_objs"].keys()))
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
    
            objCollisions[currFrame] = objCollisionChecker.CheckCollision(objsInFrames[currFrame].values())
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
        tracking_results = None

        # extract out info to feed into VIBE
        if configs["vibe_params"]["trackViaPose"]:
            if not os.path.isabs(args.input):
                absFilePathVid = os.path.join(os.getcwd(), args.input)

            absStafFolderPath = os.path.join(os.getcwd(), "openposetrack")
            absVidOutput = os.path.join(os.getcwd(), videoDrawer.outputPath)
            tracking_results = run_posetracker(os.path.normpath(absFilePathVid), staf_folder=absStafFolderPath, posetrack_output_folder=os.path.normpath(absVidOutput), display=True)
            
            for person_id in list(tracking_results.keys()):
                if tracking_results[person_id]['frames'].shape[0] < configs["obj_smoothing"]["max_missing_frames"]:
                    del tracking_results[person_id]

            print(tracking_results.keys())
            
            # format data
            for person_id in tqdm(list(tracking_results.keys())):
                humans[person_id] = PreProcessPersonData(None, tracking_results[person_id]['joints2d'], tracking_results[person_id]['frames'], person_id)
        else:
            # format object bounding box data detected from yolo to format for vibe
            for frameNo, objInFrame in objsInFrames.items():
                for obj in objInFrame.values():
                    if obj.className == 0:
                        if obj.id not in humans:
                            humans[obj.id] = PreProcessPersonData([], None, [], obj.id)
                        
                        # we use the originalBbox as the sizes are much more accurate
                        humans[obj.id].bboxes.append(ConvertBboxToCenterWidthHeight(obj.originalBbox))
                        # humans[obj.id].joints2D.append(obj.originalBbox)
                        humans[obj.id].frames.append(frameNo)

        # Run VIBE
        print("GUMAN")
        print(tracking_results)
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

    print(humanRenderData)

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
                '''
                    HACK: 
                    i am too tired to re-write my code cause i realize the openpose code works differently from mmpose
                    I can't do the spend 5 hours on documentation anymore
                    Basically if use object detection, use AABB collision to find which human has collision with the object to match keypoint
                    Else if use pose detection, we will just iterate through all humans and check for keypoint match
                '''
                objsCollidedWithID = [obj.id for obj in objsCollidedWith] if not configs["vibe_params"]["trackViaPose"] else humanRenderData.keys()
                for otherObjId in objsCollidedWithID:

                    # compare with humans keypoints to see whether to attach
                    frameIndex = FindIndexOfValueFromSortedArray(humanRenderData[otherObjId]["frame_ids"], currFrame) # the fact that the obj had AABB collision with the human means the human exists in this frame
                    for keypt in ATTACHABLE_KEYPOINTS:
                        keyPtPos = humanRenderData[otherObjId]["joints2d_img_coord"][frameIndex][keypt]
                        currDist = DistBetweenPoints((interactableObj.renderPoint[0], interactableObj.renderPoint[1]), keyPtPos)

                        isPotentialAttachment = False
                        # TODO, HAVE TO CHECK SIZE OF BALL VIA BOUNDING BOX SIZE / 2 THEN + THRESHOLD FOR COMPARISON
                        # need to take note for objects that does not have the same width and height
                        if currDist < shortestDist and currDist <= MAX_DIST_FROM_KEYPOINT:
                            shortestDist = currDist
                            interactableObj.Attach(otherObjId, keypt, (interactableObj.renderPoint[0] - keyPtPos[0], interactableObj.renderPoint[1] - keyPtPos[1]))
                            isPotentialAttachment = True

                        lineColor = (0, 255, 0) if isPotentialAttachment else (0, 0, 255)
                        DrawLineBetweenPoints(newFrame, (int(interactableObj.renderPoint[0]), int(interactableObj.renderPoint[1])), (int(keyPtPos[0]), int(keyPtPos[1])), f'{currDist}', lineColor, 1)
                    
                    DrawSkeleton(newFrame, humanRenderData[otherObjId]["joints2d_img_coord"][frameIndex], ATTACHABLE_KEYPOINTS)
            
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
    renderer = Renderer(resolution=(_videoInfo.videoWidth, _videoInfo.videoHeight), wireframe=_renderConfigs["isWireframe"])

    # dictionary, {frameNo: {humanId: verts, cam, joints3D, pose} }
    frame_results = prepare_rendering_results(_humanRenderData, _videoInfo.videoTotalFrames)
    mesh_color = {k: colorsys.hsv_to_rgb(np.random.rand(), 0.5, 1.0) for k in _humanRenderData.keys()}

    renderClip = _videoInfo.CreateNewClip("render")
    _videoInfo.ResetVideo()

    fov = math.radians(60)
    tan_half_fov = math.tan(fov * 0.5)
    aspect_ratio = float(_videoInfo.videoWidth)/float(_videoInfo.videoHeight)

    # Last known object transformations.
    obj_transformations : Dict[int, ObjectTransformation] = {}

    # Camera rotation.
    cam_angular_velocity = 1.0
    cam_pivot_pos_z = -2.5
    current_cam_angle = 0.0

    frameIndex = 0
    while True:
        hasFrames, img = _videoInfo.video.read() # gives in BGR format
        if not hasFrames:
            break

        # Calculate Rotating Camera View Matrix
        # 1. Move objects to origin.
        # 2. Rotate objects.
        # 3. Push objects out.
        if _renderConfigs["rotateCamera"]:
            view_matrix = mtx_util.translation_matrix(Vec3(0.0, 0.0, 1.5 * -cam_pivot_pos_z)) 
            view_matrix = mtx_util.rotation_matrix_y(current_cam_angle * maths_util.deg2rad) * view_matrix 
            view_matrix = mtx_util.translation_matrix(Vec3(0.0, 0.0, cam_pivot_pos_z)) * view_matrix
        else:
            view_matrix = Mtx.identity(4)

        # Convert Mtx to numpy matrix.
        cam_pose = np.eye(4)
        for col in range(4):
            for row in range(4):
                cam_pose[row, col] = view_matrix[col, row]

        # Push camera.
        renderer.push_persp_cam(fov, cam_pose)

        # Render people.
        ws_person_pos = {} # World Space Person Positions (Vec3)
        joints3d = {}
        for person_id, person_data in frame_results[frameIndex].items():
            frame_verts = person_data['verts']
            frame_cam = person_data['cam']
            frame_joints3d = person_data['joints3d']
            frame_pose = person_data['pose']

            # Store joints.
            joints3d[person_id] = frame_joints3d

            sx, sy, tx, ty = frame_cam
            ws_pos = Vec3(tx, # World Space Position X
                          -ty, # World Space Position Y
                          -1.0 / (sy * tan_half_fov)) # World Space Position Z
            ws_person_pos[person_id] = ws_pos

            # Render
            renderer.push_human(verts=frame_verts, # Add human to scene.
                                color=mesh_color[person_id],
                                translation=[ws_pos.x, ws_pos.y, ws_pos.z])

        # Render Objects.
        '''
        !!!!!!!!SUPER IMPORTANT!!!!!!!!
        1. Because we have no way of knowing how big our object's model is, without building some sort of bounding box when loading the 3D model,
           we assume that all our 3D models are of size 1. For example, a ball of diameter 1, or a cube of WxLxH = 1x1x1.
           That way they'll be appropriately sized after scaling in code.

        2. The 3D model provided by VIBE is actually upside down, and what VIBE does is that it hardcodes a 180 degree rotation on the X axis just before rendering.
           So when doing our calculations we need to account for this 180 degree X rotation.
        '''
        for obj in _objRenderData[frameIndex]:
            ss_height = obj.height # Screen Space Height
            ss_pos_x = obj.renderPoint[0] # Screen Space Position X
            ss_pos_y = obj.renderPoint[1] # Screen Space Position Y

            if obj.id not in obj_transformations:
                obj_transformations[obj.id] = ObjectTransformation()

            # Case 1: Object is attached to person.
            # We know the Z value of the object. (e.g. From the person's hand.)
            # Using that Z value, we can find the scale of the object.
            if obj.isAttached and obj.attachedToObjId in ws_person_pos:
                # Find the joint's world position.
                ls_joint = skeleton_util.get_bone_position(joints3d[obj.attachedToObjId], obj.boneAttached)
                ls_joint = Quat.rotate_via_axis_angle(ls_joint, Vec3.x_axis(), maths_util.pi) # 180 degree X rotation.
                ws_joint = ls_joint + ws_person_pos[obj.attachedToObjId]

                # Find object's world position via screen space position.
                ws_pos = resize_util.screen_to_world_xy(fov, _videoInfo.videoWidth, _videoInfo.videoHeight, ss_pos_x, ss_pos_y, ws_joint.z)

                # Find joint's current rotation.
                smpl_pose = frame_results[frameIndex][obj.attachedToObjId]['pose']
                bone_id = skeleton_util.map_spin_to_smpl()[obj.boneAttached]
                joint_rotation = skeleton_util.get_bone_rotation(smpl_pose, bone_id)
                joint_rotation = joint_rotation * Quat.from_axis_angle(Vec3.x_axis(), maths_util.pi) # 180 degree X rotation.

                if obj_transformations[obj.id].currAttachedObjId != obj.attachedToObjId or obj_transformations[obj.id].currJointId != obj.boneAttached:
                    obj_transformations[obj.id].currJointId = obj.boneAttached
                    obj_transformations[obj.id].currAttachedObjId = obj.attachedToObjId
                    obj_transformations[obj.id].initialAttachOffset = ws_pos - ws_joint # find offset
                    obj_transformations[obj.id].currJointRot = joint_rotation

                # get the rotation difference from prev rotation and curr rotation
                delta_jointRotation = joint_rotation * obj_transformations[obj.id].currJointRot.inversed()
                obj_transformations[obj.id].currJointRot = joint_rotation

                offset = obj_transformations[obj.id].initialAttachOffset

                # update values
                obj_transformations[obj.id].currRot = obj_transformations[obj.id].currRot * delta_jointRotation
                obj_transformations[obj.id].currPos = ws_pos

                axis_angle = obj_transformations[obj.id].currRot.to_axis_angle()
                axis = axis_angle[0]
                angle = axis_angle[1]


                KEYPOINT_COLOR = (255, 255, 255)
                x_point = Quat.rotate_via_quaternion(Vec3.x_axis() * 0.1, joint_rotation) + ws_joint
                x_point = resize_util.world_to_screen(fov, _videoInfo.videoWidth, _videoInfo.videoHeight, x_point.x, x_point.y, x_point.z)
                y_point = Quat.rotate_via_quaternion(Vec3.y_axis() * 0.1, joint_rotation) + ws_joint
                y_point = resize_util.world_to_screen(fov, _videoInfo.videoWidth, _videoInfo.videoHeight, y_point.x, y_point.y, y_point.z)
                z_point = Quat.rotate_via_quaternion(Vec3.z_axis()* 0.1, joint_rotation) + ws_joint
                z_point = resize_util.world_to_screen(fov, _videoInfo.videoWidth, _videoInfo.videoHeight, z_point.x, z_point.y, z_point.z)

                jointRenderPosition = resize_util.world_to_screen(fov, _videoInfo.videoWidth, _videoInfo.videoHeight, ws_joint.x, ws_joint.y, ws_joint.z)
                
                cv2.circle(img, (int(jointRenderPosition.x), int(jointRenderPosition.y)), 8, KEYPOINT_COLOR, 3)
                cv2.line(img, (int(jointRenderPosition.x), int(jointRenderPosition.y)), (int(x_point.x), int(x_point.y)), (0,0,255), 2)
                cv2.line(img, (int(jointRenderPosition.x), int(jointRenderPosition.y)), (int(y_point.x), int(y_point.y)), (0,255,0), 2)
                cv2.line(img, (int(jointRenderPosition.x), int(jointRenderPosition.y)), (int(z_point.x), int(z_point.y)), (255,0,0), 2)


                KEYPOINT_COLOR = (0, 255, 255)
                est = resize_util.world_to_screen(fov, _videoInfo.videoWidth, _videoInfo.videoHeight, ws_person_pos[obj.attachedToObjId].x, ws_person_pos[obj.attachedToObjId].y, ws_person_pos[obj.attachedToObjId].z)
                cv2.circle(img, (int(est.x), int(est.y)), 8, KEYPOINT_COLOR, 3)                

                # Find scale.
                # Use the screen space height to estimate the world space height.
                # Prefer height over width as height does not have to deal with aspect ratio in all circumstances.
                ws_scale_y = resize_util.screen_to_world_height(ss_height, _videoInfo.videoHeight, fov, ws_pos.z)
                obj_transformations[obj.id].currScaleY = ws_scale_y

                # Render.
                if _renderConfigs["objAttachToJoint"]:
                    # attach via offset to make it follow hand
                    renderer.push_obj(
                    configs["interactable_objs"][obj.className],
                    translation_offset=[offset.x, offset.y, 0.0],
                    translation=[ws_joint.x, ws_joint.y, ws_joint.z],
                    angle=angle,
                    axis=[axis.x, axis.y, axis.z],
                    scale=[ws_scale_y, ws_scale_y, ws_scale_y],
                    color=[0.05, 1.0, 1.0])
                else:
                    # follow image position
                    renderer.push_obj(
                        configs["interactable_objs"][obj.className],
                        translation_offset=[0.0,0.0, 0.0],
                        translation=[ws_pos.x, ws_pos.y, ws_pos.z],
                        angle=angle,
                        axis=[axis.x, axis.y, axis.z],
                        scale=[ws_scale_y, ws_scale_y, ws_scale_y],
                        color=[0.05, 1.0, 1.0])


            # Case 2: Object is not attached to person.
            # We don't know the Z value of the object.
            # Using the scale of the object (we assume the object is already correctly scaled), find the Z value of the object.
            else:
                obj_transformations[obj.id].currAttachedObjId = -1 # not attach to anything

                # Get the last known world space scale of the object.
                ws_scale_y = obj_transformations[obj.id].currScaleY

                # Get the NDC transforms of the object. NDC is range [-1, 1].
                ndc_scale_y = 2.0 * ss_height / _videoInfo.videoHeight
                ndc_pos_x = resize_util.screen_to_ndc_x(_videoInfo.videoWidth, ss_pos_x)
                ndc_pos_y = resize_util.screen_to_ndc_y(_videoInfo.videoHeight, ss_pos_y)

                # How far must the object be in world space, such that it's NDC scale is correct?
                ws_pos_z = -ws_scale_y / (ndc_scale_y * tan_half_fov) # Negate because the camera is looking down the -Z axis.
                ws_pos = Vec3(tan_half_fov * ws_pos_z * ndc_pos_x * aspect_ratio,
                              tan_half_fov * ws_pos_z * ndc_pos_y,
                              ws_pos_z)
                obj_transformations[obj.id].currPos = ws_pos

                # Get the last known rotation. We do not change rotation here, hard to predict without a reference
                # TODO: make sure obj is inside first
                rotation = obj_transformations[obj.id].currRot
                axis_angle = rotation.to_axis_angle()
                axis = axis_angle[0]
                angle = axis_angle[1]

                # Render.
                renderer.push_obj(
                    configs["interactable_objs"][obj.className],
                    translation_offset=[0.0, 0.0, 0.0],
                    translation=[ws_pos.x, ws_pos.y, ws_pos.z],
                    angle=angle,
                    axis=[axis.x, axis.y, axis.z],
                    scale=[ws_scale_y, ws_scale_y, ws_scale_y],
                    color=[0.0, 1.0, 1.0])

        # Clear ws_person_pos for next frame.
        del ws_person_pos

        img = renderer.pop_and_render(img, _renderConfigs["renderOnWhite"])
        frameIndex += 1
        current_cam_angle += cam_angular_velocity
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
        parser.add_argument("--detectionPKL",  default='Output/' + video_name + '/detected.pkl', type=str, help="Pre-processed Pkl file containing smpl data of the video")
        parser.add_argument("--collisionDetectionPKL", default='Output/' + video_name + '/object_collisions.pkl', type=str, help="Pre-processed Pkl file containing smpl data of the video")
        parser.add_argument("--smplPKL", default='', type=str, help="Pre-processed Pkl file containing smpl data of the video")
        parser.add_argument("--objKeyPtRawData", default='', type=str, help="Pre-processed Pkl file containing smpl data of the video")
    else:
        parser.add_argument("--input", default='Input/'+ video_name + '.mp4', type=str, help="File path for video")
        parser.add_argument("--config", default='Configs/config.yaml', type=str, help="File path for config file")
        parser.add_argument("--detectionPKL", default='Output/' + video_name + '/detected.pkl', type=str, help="Pre-processed Pkl file containing smpl data of the video")
        parser.add_argument("--collisionDetectionPKL", default='Output/' + video_name + '/object_collisions.pkl', type=str, help="Pre-processed Pkl file containing smpl data of the video")
        parser.add_argument("--smplPKL", default='Output/' + video_name + '/vibe_output.pkl', type=str, help="Pre-processed Pkl file containing smpl data of the video")
        parser.add_argument("--objKeyPtRawData", default='Output/' + video_name + '/obj_kpt_attachment.pkl', type=str, help="Pre-processed Pkl file containing smpl data of the video")

    arguments = parser.parse_args()
    
    with open(arguments.config) as f:
        configs = yaml.safe_load(f)

    availableObjs = configs.get("interactable_objs", {})
    
    main(arguments)