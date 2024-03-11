
from abc import ABC, abstractmethod
import os
import cv2
from utils import CreateVideo

class VideoDrawer(ABC):
    # Create new video clip to draw on
    def StartVideo(self, _videoPath, _createNewClip = True, _outputFilePath = ""):
        self.video = cv2.VideoCapture(_videoPath)
        self.videoFps = int(self.video.get(cv2.CAP_PROP_FPS))
        self.videoWidth = int(self.video.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.videoHeight = int(self.video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.videoTotalFrames = int(self.video.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if _createNewClip:
            outputFolder = _outputFilePath
            videoFileName = os.path.basename(_videoPath)
            vidName, vidFileExtension = os.path.splitext(videoFileName)
            self.currentClip = CreateVideo(f"{outputFolder}/{vidName}" , "FullClip.mp4", self.videoFps, self.videoWidth, self.videoHeight)

    def StopVideo(self):
        self.video.release()

    @abstractmethod
    def Draw(self, _frame, _frameIndex = 0):
        pass

    def DrawOnClip(self, _vidFrame, _frameIndex = 0):
        self.Draw(_vidFrame, _frameIndex)
        self.currentClip.write(_vidFrame)

    def ReleaseNewClipRecording(self):
        self.currentClip.release()
