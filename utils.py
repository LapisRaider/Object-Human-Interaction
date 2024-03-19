import os
import cv2
import math
from Rendering.lib.data_utils.kp_utils import get_spin_skeleton, get_spin_joint_names

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