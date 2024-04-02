import os
import cv2
import math
from Rendering.lib.data_utils.kp_utils import get_spin_skeleton, get_spin_joint_names
import numpy as np

import resize_util
from vec3 import Vec3
from mtx import Mtx
from quat import Quat

# Font settings
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.2
FONT_COLOR = (0, 0, 0)  # White color in BGR format
FONT_THICKNESS = 1

def CreateVideo(_fileDirPath, _nameOfVid, _fps, _width, _height):
    if not os.path.exists(_fileDirPath):
        os.makedirs(_fileDirPath, exist_ok=True)

    filePath = f"{_fileDirPath}/{_nameOfVid}"
    return  cv2.VideoWriter(filePath, cv2.VideoWriter_fourcc(*'mp4v'), _fps, (_width, _height))

# box is to be given in xyxy format
def DrawBox(_frame, _bbox, _color=[0, 255,0], _lineThickness=1):
        x1, y1, x2, y2 = map(int, _bbox[:4])

        #draw bounding box
        cv2.rectangle(_frame, (x1, y1), (x2, y2), _color, _lineThickness)


'''
    joints2d:
        array of [x, y] coordinates in SPIN format
    _specialKeypoints:
        a hash set of keypoints' indexes
'''
def DrawSkeleton(_frame, _joints2d, _specialKeypoints = set()):
    # draw joints
    for index, point in enumerate(_joints2d[:24]):
        text = f'{get_spin_joint_names()[index]}'
        pos = (int(point[0]), int(point[1]))

        KEYPOINT_COLOR = (0, 255, 0)
        if index in _specialKeypoints:
            KEYPOINT_COLOR = (0, 255, 255) 

        cv2.circle(_frame, pos, 8, KEYPOINT_COLOR, 3)
        cv2.putText(_frame, text, (pos[0], pos[1] - 5), FONT, FONT_SCALE, FONT_COLOR, FONT_THICKNESS)

    # draw connections
    for connection in get_spin_skeleton():
        startPointJointIndex = connection[0]
        endPointJointIndex = connection[1]

        startPoint = (int(_joints2d[startPointJointIndex][0]), int(_joints2d[startPointJointIndex][1]))
        endPoint = (int(_joints2d[endPointJointIndex][0]), int(_joints2d[endPointJointIndex][1]))
        
        cv2.line(_frame, startPoint, endPoint, (255,0,0), 2)

    return _frame


def DrawLineBetweenPoints(_frame, _startPt, _endPt, _text = "", _lineColor = (255,0,0), _lineThickness = 2):
    cv2.line(_frame, _startPt, _endPt, _lineColor, _lineThickness)

    midPt = (int((_startPt[0] + _endPt[0]) / 2), int((_startPt[1] + _endPt[1]) / 2))
    cv2.putText(_frame, _text, (midPt[0], midPt[1] - 5), FONT, FONT_SCALE, FONT_COLOR, FONT_THICKNESS)
    return _frame

def DrawTextOnTopRight(_frame, _text, _imgWidth, _height = 20):
    textSize = cv2.getTextSize(_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
    sceneInfoX = (int)(_imgWidth - textSize[0])

    cv2.putText(_frame, _text, (sceneInfoX, _height), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 150,0), 2)

'''
    Arguments:
        ws_pt: point in world space
        rotation: rotation of this point
'''
def DrawAxisAtPoint(_frame, _ws_pt, _rotation, _imgWidth, _height, _fov, _length = 0.2):
    KEYPOINT_COLOR = (255, 255, 255)
    x_point = Quat.rotate_via_quaternion(Vec3.x_axis() * _length, _rotation) + _ws_pt
    x_point = resize_util.world_to_screen(_fov, _imgWidth, _height, x_point.x, x_point.y, x_point.z)
    
    y_point = Quat.rotate_via_quaternion(Vec3.y_axis() * _length, _rotation) + _ws_pt
    y_point = resize_util.world_to_screen(_fov, _imgWidth, _height, y_point.x, y_point.y, y_point.z)
    
    z_point = Quat.rotate_via_quaternion(Vec3.z_axis()* _length, _rotation) + _ws_pt
    z_point = resize_util.world_to_screen(_fov, _imgWidth, _height, z_point.x, z_point.y, z_point.z)

    screenSpacePt = resize_util.world_to_screen(_fov, _imgWidth, _height, _ws_pt.x, _ws_pt.y, _ws_pt.z)
                
    cv2.circle(_frame, (int(screenSpacePt.x), int(screenSpacePt.y)), 8, KEYPOINT_COLOR, 3)
    cv2.line(_frame, (int(screenSpacePt.x), int(screenSpacePt.y)), (int(x_point.x), int(x_point.y)), (0,0,255), 2)
    cv2.line(_frame, (int(screenSpacePt.x), int(screenSpacePt.y)), (int(y_point.x), int(y_point.y)), (0,255,0), 2)
    cv2.line(_frame, (int(screenSpacePt.x), int(screenSpacePt.y)), (int(z_point.x), int(z_point.y)), (255,0,0), 2)



"""
    Calculate the Euclidean distance between two points in a 2-dimensional plane.
    
    Args:
        point1 (tuple): Coordinates of the first point (x1, y1).
        point2 (tuple): Coordinates of the second point (x2, y2).
    
    Returns:
        float: The Euclidean distance between the two points.
"""
def DistBetweenPoints(point1, point2):
    x1, y1 = point1
    x2, y2 = point2
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)


"""
    Find the index of the target_id in the sorted frame_ids array using binary search.

    Args:
        frame_ids (list): A sorted list of frame IDs.
        target_id: The ID to search for.

    Returns:
        int: The index of the target_id in the frame_ids array, or -1 if not found.
"""
def FindIndexOfValueFromSortedArray(sortedArray, targetValue):
    left, right = 0, len(sortedArray) - 1

    while left <= right:
        mid = left + (right - left) // 2
        if sortedArray[mid] == targetValue:
            return mid
        elif sortedArray[mid] < targetValue:
            left = mid + 1
        else:
            right = mid - 1

    return -1

def ConvertBboxToCenterWidthHeight(_bbox):
        # Extract coordinates from bbox
        minX, minY, maxX, maxY = _bbox

        # Calculate height h
        h = maxY - minY
        w = maxX - minX

        # Calculate center coordinates (c_x, c_y)
        c_x = minX + w/2
        c_y = minY + h/2

        w = h = np.where(w / h > 1, w, h)

        # Return as numpy array in shape (4,)
        return [c_x, c_y, w, h]