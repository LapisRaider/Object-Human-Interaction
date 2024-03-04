import os
import cv2

def CreateVideo(_fileDirPath, _nameOfVid, _fps, _width, _height):
    if not os.path.exists(_fileDirPath):
        os.makedirs(_fileDirPath, exist_ok=True)

    filePath = f"{_fileDirPath}/{_nameOfVid}"
    return  cv2.VideoWriter(filePath, cv2.VideoWriter_fourcc(*'mp4v'), _fps, (_width, _height))
