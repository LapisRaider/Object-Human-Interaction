import os
import cv2
from Rendering.lib.data_utils.kp_utils import get_spin_skeleton, get_spin_joint_names

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
        array of [x, y] coordinates
'''
def DrawSkeleton(_frame, _joints2d):
    # Font settings
    FONT = cv2.FONT_HERSHEY_SIMPLEX
    FONT_SCALE = 0.2
    FONT_COLOR = (0, 0, 0)  # White color in BGR format
    FONT_THICKNESS = 1


    # draw joints
    for index, point in enumerate(_joints2d):
        text = f'{get_spin_joint_names()[index]}'
        pos = (int(point[0]), int(point[1]))

        cv2.circle(_frame, pos, 8, (0,255,0), 3)
        cv2.putText(_frame, text, (pos[0], pos[1] - 5), FONT, FONT_SCALE, FONT_COLOR, FONT_THICKNESS)

    # draw connections
    for connection in get_spin_skeleton():
        startPointJointIndex = connection[0]
        endPointJointIndex = connection[1]

        startPoint = (int(_joints2d[startPointJointIndex][0]), int(_joints2d[startPointJointIndex][1]))
        endPoint = (int(_joints2d[endPointJointIndex][0]), int(_joints2d[endPointJointIndex][1]))
        
        cv2.line(_frame, startPoint, endPoint, (255,0,0), 2)

    return _frame