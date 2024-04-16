
import os
import cv2
from utils import CreateVideo

class VideoDrawer:
    # Create new video clip to draw on
    def __init__(self, _videoPath, _outputFolderPath):
        self.video = cv2.VideoCapture(_videoPath)
        self.videoFps = int(self.video.get(cv2.CAP_PROP_FPS))
        self.videoWidth = int(self.video.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.videoHeight = int(self.video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.videoTotalFrames = int(self.video.get(cv2.CAP_PROP_FRAME_COUNT))
        self.videoPath = _videoPath 

        videoFileName = os.path.basename(self.videoPath)
        vidName, vidFileExtension = os.path.splitext(videoFileName)
        self.outputPath = f"{_outputFolderPath}/{vidName}"  

    def CreateNewClip(self, _vidName = "", _outputFolder = "", videoWidth = -1, videoHeight = -1):
        if _outputFolder == "":
            _outputFolder = self.outputPath

        videoFileName = os.path.basename(self.videoPath)
        vidName, vidFileExtension = os.path.splitext(videoFileName)

        return CreateVideo(_outputFolder, f"{_vidName}.mp4", self.videoFps, self.videoWidth if videoWidth == -1 else videoWidth,
                            self.videoHeight if videoHeight == -1 else videoHeight)

    def StopVideo(self):
        self.video.release()
    
    def ResetVideo(self):
        self.video.set(cv2.CAP_PROP_POS_FRAMES, 0)
