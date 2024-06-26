from utils import ConvertBboxToCenterWidthHeight
from vec3 import Vec3
from quat import Quat


class DetectedObj:
    def __init__(self, _id, _bbox, _oriBBox, _conf, _className):
        self.bbox = _bbox # in xyxy format
        self.originalBbox = _oriBBox
        self.conf = _conf
        self.className = _className
        self.id = _id

    @classmethod
    def clone(cls, obj):
        _id = obj.id
        _bbox = obj.bbox
        _oriBBox = obj.originalBbox
        _conf = obj.conf
        _className = obj.className

        return cls(_id, _bbox, _oriBBox, _conf, _className)

    '''
        Will move the box by a certain offset given a x and y coordinates
    '''
    def applyOffset(self, _offsetX, _offsetY):
        self.bbox = [self.bbox[0] + _offsetX, self.bbox[1] + _offsetY, self.bbox[2] + _offsetX, self.bbox[3] + _offsetY]

    def collidesWith(self, _other):
        minX, minY, maxX, maxY = self.bbox
        otherMinX, otherMinY, otherMaxX, otherMaxY = _other.bbox
        x_intersect = (minX < otherMaxX) and (otherMinX < maxX)
        y_intersect = (minY < otherMaxY) and (otherMinY < maxY)

        return x_intersect and y_intersect

'''
    Objects that a human can interact with
'''
class HumanInteractableObject(DetectedObj):
    
    def __init__(self, _id, _bbox, _oriBBox, _conf, _className):
        super().__init__(_id, _bbox, _oriBBox, _conf, _className)

        self.isAttached = False
        self.attachedToObjId = -1 # id of object it is attached to
        self.boneAttached = -1 # which bone is it attached to

        predictionBboxConverstion = ConvertBboxToCenterWidthHeight(_bbox)
        originalBboxConverstion = ConvertBboxToCenterWidthHeight(_oriBBox)

        self.renderPoint = [predictionBboxConverstion[0], predictionBboxConverstion[1]]
        self.width = originalBboxConverstion[2]
        self.height = originalBboxConverstion[3]
        self.offset = (0 , 0) # offset away from interactable point

    @classmethod
    def from_parent(cls, parent_obj):
        _id = parent_obj.id
        _bbox = parent_obj.bbox
        _oriBBox = parent_obj.originalBbox
        _conf = parent_obj.conf
        _className = parent_obj.className

        interactable_obj = cls(_id, _bbox, _oriBBox, _conf, _className)

        return interactable_obj
    
    def Attach(self, _otherObjId, _boneId, _offset):
        self.offset = _offset
        self.attachedToObjId = _otherObjId
        self.boneAttached = _boneId
        self.isAttached = True

class Coordinates:
    def __init__(self, x = 0, y = 0, z = 0):
        self.x = x
        self.y = y
        self.z = z

class ObjectTransformation:
    def __init__(self):
        self.currPos : Vec3 = Vec3.zero()
        self.currRot : Quat = Quat.identity()
        self.currScaleY : float = 1.0

        # for attachment
        self.currJointId : int = -1
        self.currAttachedObjId : int = -1; # curr obj it is attached to
        self.currJointRot : Quat = Quat.identity()
        self.initialAttachOffset: Vec3 = Vec3.zero()