import cv2

# deep sort
from deep_sort.application_util import preprocessing
from deep_sort.deep_sort import nn_matching
from deep_sort.deep_sort.detection import Detection
from deep_sort.deep_sort.tracker import Tracker as DeepsortTracker
from deep_sort.deep_sort.track import TrackState
from deep_sort.tools import generate_detections as gdet

from utils import DrawBox
from videoDrawer import VideoDrawer
from detectedObj import DetectedObj


# Detect and track certain class objs throughout the frames
class VideoObjDetector():
    def __init__(self, _deepSortConfigs, _classIds = None):
        self.classIds = _classIds
        self.objsInFrames = [] # a double array of Detected Objs in each frame

        # DeepSORT -> Initializing tracker.
        max_cosine_distance = _deepSortConfigs["max_distance"]
        distanceMode = 'cosine' if _deepSortConfigs["use_cosine_distance"] else 'euclidean'
        nn_budget = None
        model_filename = _deepSortConfigs["checkpoint_file"]
        self.encoder = gdet.create_box_encoder(model_filename, batch_size=1)
        metric = nn_matching.NearestNeighborDistanceMetric(distanceMode, max_cosine_distance, nn_budget)
        self.deepsortTracker = DeepsortTracker(
            metric,
            _deepSortConfigs["max_iou_distance"],
            _deepSortConfigs["max_age"],
            _deepSortConfigs["n_init"],
            )
        
    def DetectObjs(self, _vidFrameData, _yoloModel, _yoloConfigs):
        objs = []

        results = _yoloModel.predict(
            _vidFrameData, 
            conf = _yoloConfigs["confidence_score"],
            iou = _yoloConfigs["intersection_over_union"],
            classes = self.classIds
            )
        
        bboxes_xywh = []
        scores = []
        labels = []
        # results is frames, since we only have 1 frame, it won't matter
        for detected in results[0].boxes.cpu():
            x1, y1, x2, y2 = detected.xyxy[0]
            bbox_left = min([x1, x2])
            bbox_top = min([y1, y2])
            bbox_w = abs(x1 - x2)
            bbox_h = abs(y1 - y2)
            box = [bbox_left, bbox_top, bbox_w, bbox_h] 
            bboxes_xywh.append(box)
            scores.append(detected.conf[0])
            labels.append(int(detected.cls[0]))
        
        # deepsort, track people movements across frames
        features = self.encoder(_vidFrameData, bboxes_xywh) # get appearance features of obj
        detections = [Detection(bbox, score, feature, label) for bbox, score, feature, label in zip(bboxes_xywh, scores, features, labels)]
        self.deepsortTracker.predict()
        self.deepsortTracker.update(detections)
        self.objsInFrames.append([])

        for track in self.deepsortTracker.tracks:
            # track is inactive
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            detectedObj = DetectedObj(
                track.track_id, 
                track.to_tlbr(), 
                track.initialDetectionData["bbox"], 
                track.initialDetectionData["score"], 
                track.initialDetectionData["class"]
                )
            objs.append(detectedObj)

        return objs;


    def Draw(self, _frame, _objs):
        for obj in _objs:
            DrawBox(_frame, obj.bbox, [0, 255, 0], 2)
            DrawBox(_frame, obj.originalBbox, [255, 0, 0], 1)
            cv2.putText(_frame, f'id: {obj.id}', (int(obj.bbox[0]), int(obj.bbox[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.putText(_frame, f'c: {obj.className}', (int(obj.bbox[0]), int(obj.bbox[1]) + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.putText(_frame, f's: {obj.conf}', (int(obj.bbox[0]), int(obj.bbox[1]) + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)





            
