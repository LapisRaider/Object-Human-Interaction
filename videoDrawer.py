
import os
import cv2
from utils import CreateVideo

class VideoDrawer:
    # Create new video clip to draw on
    def __init__(self, _videoPath):
        self.video = cv2.VideoCapture(_videoPath)
        self.videoFps = int(self.video.get(cv2.CAP_PROP_FPS))
        self.videoWidth = int(self.video.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.videoHeight = int(self.video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.videoTotalFrames = int(self.video.get(cv2.CAP_PROP_FRAME_COUNT))
        self.videoPath = _videoPath
        

    def CreateNewClip(self, _outputFolder = "", _vidName = ""):
        videoFileName = os.path.basename(self.videoPath)
        vidName, vidFileExtension = os.path.splitext(videoFileName)

        return CreateVideo(f"{_outputFolder}/{vidName}" , f"{_vidName}.mp4", self.videoFps, self.videoWidth, self.videoHeight)

    def GetFilePath(self, _outputFolder = ""):
        videoFileName = os.path.basename(self.videoPath)
        vidName, vidFileExtension = os.path.splitext(videoFileName)

        return f"{_outputFolder}/{vidName}"

    def StopVideo(self):
        self.video.release()
