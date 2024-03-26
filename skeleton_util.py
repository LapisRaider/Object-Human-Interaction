import math

import maths_util
import mtx_util
from vec3 import Vec3
from mtx import Mtx
from quat import Quat

# Get joint index from kp_utils.get_spin_joint_names().
def get_bone_position(frame_joints3d, joint_id):
    return Vec3(frame_joints3d[joint_id][0], frame_joints3d[joint_id][1], frame_joints3d[joint_id][2])

# Get bone index from fbx_output.bone_name_from_index. This is SMPL parameters kp_utils.get_smpl_joint_names().
RIGHT_HAND_INHERITANCE = [0, 3, 6, 9, 14, 17, 19, 21, 23]
LEFT_HAND_INHERITANCE = [0, 3, 6, 9, 13, 16, 18, 20, 22]

LEFT_LEG_INHERITANCE = [0, 1, 4, 7, 10]
RIGHT_LEG_INHERITANCE = [0, 2, 5, 8, 11]

HEAD_INHERITANCE = [0, 3, 6, 9, 12, 15]

'''
    Taking the smpl pose data and bone index
    Returns a quaternion 
'''
def _local_rotation(frame_pose, index):
    axis_angle = Vec3(frame_pose[index * 3], frame_pose[index * 3 + 1], frame_pose[index * 3 + 2])
    axis = axis_angle.normalised()
    angle = axis_angle.length()
    return Quat.from_axis_angle(axis, angle)
    
   
'''
    Parameter:
        smpl_joints: get information of 3D joints in axis angle format
        bone_id: the bone we want the rotation of
        inheritance: part of the body and the bones in that part that the boneId belongs to
    Return:
        Quaternion matrix stating the rotation of the bone
'''
def _accumulated_rotation(smpl_joints, bone_id, inheritance):
    rot = Quat.identity()
    for parent_id in inheritance:
        rot = rot * _local_rotation(smpl_joints, parent_id)
        if parent_id == bone_id:
            break
    return rot

'''
    Parameter:
        smpl_joints: get information of 3D joints in axis angle format
        bone_id: the bone we want the rotation of
    Return:
        Quaternion matrix stating the rotation of the bone
'''
def get_bone_rotation(smpl_joints, bone_id):
    # find the parents of boneId and which part of the body it belongs to
    inheritance = RIGHT_HAND_INHERITANCE
    if bone_id in LEFT_HAND_INHERITANCE:
        inheritance = LEFT_HAND_INHERITANCE
    elif bone_id in LEFT_LEG_INHERITANCE:
        inheritance = LEFT_LEG_INHERITANCE
    elif bone_id in RIGHT_LEG_INHERITANCE:
        inheritance = RIGHT_LEG_INHERITANCE
    elif bone_id in HEAD_INHERITANCE:
        inheritance = HEAD_INHERITANCE    
    return _accumulated_rotation(smpl_joints, bone_id, inheritance)

'''
    Notes:
        SMPL have some things that cannot be mapped
            - spine
            - spine1
            - leftHandIndex1
            - rightHandIndex1
'''
def map_spin_to_smpl():
    return {
        8 : 0, # OP MidHip to hips
        1: 9, # OP Neck to spine2
        0: 12, # OP Nose to neck
        12: 1, # OP LHip to leftUpLeg
        13: 4, # OP LKnee to leftLeg
        14: 7,  # OP LAnkle to leftFoot
        19: 10, # OP LBigToe to leftToeBase
        9: 2, # OP RHip to rightUpLeg
        10:5, # OP RKnee to rightLeg
        11:8, # OP RAnkle to rightFoot
        22:11, # OP RBigToe to rightToeBase
        2: 17, # OP RShoulder to rightArm
        3: 19, # OP RElbow to rightForeArm
        4: 21, # OP RWrist to rightHand
        5: 16, # OP LShoulder to leftArm
        6:18, # OP LElbow to leftForeArm
        7: 20 # OP LWrist to leftHand
    }