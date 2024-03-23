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
import lib.vibe_obj.utils as vibe_obj
from vidSMPLParamCreator import PreProcessPersonData, VidSMPLParamCreator
from lib.utils.renderer import Renderer
from lib.utils.demo_utils import (
    prepare_rendering_results,
)
from lib.data_utils.kp_utils import map_spin_to_smpl
from lib.vibe_obj.utils import get_rotation

# [VIBE-Object Start]
import math
import resize

import maths_util
import mtx_util
from vec3 import Vec3
from mtx import Mtx
from quat import Quat
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
    tan_half_fov = math.tan(fov * 0.5)
    aspect_ratio = float(_videoInfo.videoWidth)/float(_videoInfo.videoHeight)

    # Last known object transformations.
    ws_obj_pos = {} # Key: Obj Id, Value: Object World Space Position (Coordinates)
    ws_obj_scale_y = {} # Key: Obj Id, Value: Object World Space Scale Y (float)
    ws_obj_rot = {}

    # Camera rotation.
    cam_pivot_angle = 0.0
    cam_angular_velocity = 1.0
    cam_pivot_pos_z = -2.5

    frameIndex = 0
    while True:
        hasFrames, img = _videoInfo.video.read() # gives in BGR format
        if not hasFrames:
            break

        # Calculate Rotating Camera View Matrix
        view_matrix = Mtx.identity(4)
        view_matrix = mtx_util.translation_matrix(Vec3(0.0, 0.0, 1.5 * -cam_pivot_pos_z)) * view_matrix # 1. Move objects to origin.
        view_matrix = mtx_util.rotation_matrix_y(cam_pivot_angle * maths_util.deg2rad) * view_matrix # 2. Rotate objects.
        view_matrix = mtx_util.translation_matrix(Vec3(0.0, 0.0, cam_pivot_pos_z)) * view_matrix # 3. Push objects out. Or something, I don't fucking know anymore and frankly, I don't care either.
 
        # Convert Mtx to numpy matrix.
        cam_pose = np.eye(4)
        for col in range(4):
            for row in range(4):
                cam_pose[row, col] = view_matrix[col, row]

        # Push camera.
        renderer.push_persp_cam(fov, cam_pose)

        # Render people.
        ws_person_pos = {} # World Space Person Positions (Coordinates)
        ws_hand_pos = {} # World Space Hand Positions (Coordinates)
        ws_hand_rot = {}
        for person_id, person_data in frame_results[frameIndex].items():
            frame_verts = person_data['verts']
            frame_cam = person_data['cam']
            frame_joints3d = person_data['joints3d']
            frame_pose = person_data['pose']

            sx, sy, tx, ty = frame_cam
            ws_pos_x = tx # World Space Position X
            ws_pos_y = -ty # World Space Position Y
            ws_pos_z = -1.0 / (sy * tan_half_fov) # World Space Position Z
            ws_person_pos[person_id] = Coordinates(ws_pos_x, ws_pos_y, ws_pos_z) # World Space Position

            # Render
            renderer.push_human(verts=frame_verts, # Add human to scene.
                                color=mesh_color[person_id],
                                translation=[ws_pos_x, ws_pos_y, ws_pos_z])

            # Get hand position.
            # We have to negate the Y & Z, because the 3D model provided by VIBE is actually upside down,
            # and what VIBE does is that it hardcodes a 180 degree rotation on the X axis just before rendering.
            # So we simulate that 180 degree rotation on the X axis by just negating the Y & Z.
            hand_pos = vibe_obj.get_left_wrist_translation(frame_joints3d)
            ws_hand_pos[person_id] = Coordinates(hand_pos[0] + ws_pos_x,
                                                 -hand_pos[1] + ws_pos_y,
                                                 -hand_pos[2] + ws_pos_z)
            
            # Get hand rotation.
            # TODO: Find rotation. Don't forget about the 3D model being upside down.
            ws_hand_rot[person_id] = None

        # Render Objects.
        '''
        !!!!!!!!SUPER IMPORTANT!!!!!!!!
        Because we have no way of knowing how big our object's model is, without building some sort of bounding box when loading the 3D model,
        we assume that all our 3D models are of size 1. For example, a ball of diameter 1, or a cube of WxLxH = 1x1x1.
        That way they'll be appropriately sized after scaling in code.
        '''
        for obj in _objRenderData[frameIndex]:
            ss_height = obj.height # Screen Space Height
            ss_pos_x = obj.renderPoint[0] # Screen Space Position X
            ss_pos_y = obj.renderPoint[1] # Screen Space Position Y

            # Case 1: Object is attached to person.
            # We know the Z value of the object. (e.g. From the person's hand.)
            # Using that Z value, we can find the scale of the object.
            if obj.isAttached and obj.attachedToObjId in ws_person_pos:
                '''
                # Find world position via screen space position.
                ws_pos_z = ws_person_pos[obj.attachedToObjId].z # World Space Position Z
                ws_pos_x, ws_pos_y, ws_pos_z = renderer.screenspace_to_worldspace(ss_pos_x, ss_pos_y, ws_pos_z) # World Space Position

                # Update last known object position.
                ws_obj_pos[obj.id] = Coordinates(ws_pos_x, ws_pos_y, ws_pos_z)
                '''

                # What is the offset between the object and the hand?
                ws_offset_x = 0.0 # TODO: Find the Offset
                ws_offset_y = 0.0
                ws_offset_z = 0.0

                # Find world space position via attached hand.
                ws_pos_x = ws_hand_pos[obj.attachedToObjId].x
                ws_pos_y = ws_hand_pos[obj.attachedToObjId].y
                ws_pos_z = ws_hand_pos[obj.attachedToObjId].z

                # Update last known object position.
                # TODO: Account for offset too? Or maybe not?
                ws_obj_pos[obj.id] = Coordinates(ws_pos_x, ws_pos_y, ws_pos_z)
                
                # What is the hand rotation?
                # TODO: Find rotation.
                ws_angle = 0.0
                ws_axis = [1.0, 0.0, 0.0]

                # Update last known object rotation.
                ws_obj_rot[obj.id] = None

                # Find Scale
                ws_scale_y = resize.get_world_height(ss_height, _videoInfo.videoHeight, fov, ws_pos_z) # Use the screen space height to estimate the world space height. Prefer height over width as height does not have to deal with aspect ratio in all circumstances.

                # Update last known object scale.
                # ws_obj_scale_y[obj.id] = ws_scale_y
                ws_obj_scale_y[obj.id] = ws_scale_y

                renderer.push_obj(
                    '3D_Models/sphere.obj',
                    translation_offset=[ws_offset_x, ws_offset_y, ws_offset_z],
                    translation=[ws_pos_x + ws_offset_x, ws_pos_y + ws_offset_y, ws_pos_z + ws_offset_z],
                    angle=ws_angle,
                    axis=ws_axis,
                    scale=[ws_scale_y, ws_scale_y, ws_scale_y],
                    color=[0.05, 1.0, 1.0])

                '''
                smpl_pose = frame_results[frameIndex][obj.attachedToObjId]['pose']
                axis_angle = get_rotation(smpl_pose, map_spin_to_smpl()[obj.boneAttached]) 
                axis = axis_angle[1:4]
                angle = axis_angle[0] * (180.0/math.pi)

                
                # overall screen x and y coordinates are the same, but the transformation order is different
                # TODO: now using 2d keypoint position instead of smpl 3d position, might want to change
                humanFrameIndex = FindIndexOfValueFromSortedArray(_humanRenderData[obj.attachedToObjId]["frame_ids"], frameIndex)
                joint2DCoords = _humanRenderData[obj.attachedToObjId]["joints2d_img_coord"][humanFrameIndex][obj.boneAttached]
                jointWorldCoords = renderer.screenspace_to_worldspace(joint2DCoords[0], joint2DCoords[1], pos_z)
                objWorldCoords = renderer.screenspace_to_worldspace(ss_pos_x, ss_pos_y, pos_z)
                
                offset = [objWorldCoords[0] - jointWorldCoords[0], objWorldCoords[1] - jointWorldCoords[1], 0]
                ss_pos_x, ss_pos_y = joint2DCoords

                obj_scale = resize.get_world_height(obj.width, _videoInfo.videoHeight, fov, pos_z)
                obj_x, obj_y, obj_z = renderer.screenspace_to_worldspace(ss_pos_x, ss_pos_y, pos_z)
                ws_obj_pos[obj.id] = Coordinates(obj_x, obj_y, obj_z)
                '''

            # Case 2: Object is not attached to person.
            # We don't know the Z value of the object.
            # Using the scale of the object (we assume the object is already correctly scaled), find the Z value of the object.
            else:
                # Get the last known world space scale of the object.
                ws_scale_y = 1.0 if obj.id not in ws_obj_scale_y else ws_obj_scale_y[obj.id]

                # Get the NDC transforms of the object. NDC is range [-1, 1].
                ndc_scale_y = 2.0 * ss_height / _videoInfo.videoHeight
                ndc_pos_x = -(ss_pos_x / _videoInfo.videoWidth * 2.0 - 1.0)
                ndc_pos_y = ss_pos_y / _videoInfo.videoHeight * 2.0 - 1.0

                # How far must the object be in world space, such that it's NDC scale is correct?
                ws_pos_z = -ws_scale_y / (ndc_scale_y * tan_half_fov) # Negate because the camera is looking down the -Z axis.
                ws_pos_y = tan_half_fov * ws_pos_z * ndc_pos_y
                ws_pos_x = tan_half_fov * ws_pos_z * ndc_pos_x * aspect_ratio
                ws_obj_pos[obj.id] = Coordinates(ws_pos_x, ws_pos_y, ws_pos_z)

                # print("Object Position: (" + str(ws_pos_z) + ", " + str(ws_pos_y) + ", " + str(ws_pos_z) + ")")

                # Get the last known world space rotation of the object.
                # TODO: Find rotation.
                # ws_rot = (0.0, (1.0, 0.0, 0.0)) if obj.id not in ws_obj_rot else ws_obj_rot[obj.id]
                ws_angle = 0.0
                ws_axis = [0.0, 0.0, 0.0]

                renderer.push_obj(
                    '3D_Models/sphere.obj',
                    translation_offset=[0.0, 0.0, 0.0],
                    translation=[ws_pos_x, ws_pos_y, ws_pos_z],
                    angle=ws_angle,
                    axis=ws_axis,
                    scale=[ws_scale_y, ws_scale_y, ws_scale_y],
                    color=[0.0, 1.0, 1.0])

        # Clear ws_person_pos for next frame.
        del ws_person_pos

        img = renderer.pop_and_render(img)
        frameIndex += 1
        cam_pivot_angle += cam_angular_velocity
        DrawTextOnTopRight(img, f"{frameIndex}/{_videoInfo.videoTotalFrames}",  _videoInfo.videoWidth)
        renderClip.write(img)
        print(f"processed render for frame {frameIndex}/{_videoInfo.videoTotalFrames}")

    renderClip.release()


if __name__ == "__main__":
    refresh = False
    video_name = "ThrowBall_2People"

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