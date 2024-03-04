import os
import cv2

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