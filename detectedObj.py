import numpy as np

class DetectedObj:
    def __init__(self, _id, _bbox, _oriBBox, _conf, _className):
        self.bbox = _bbox # in xyxy format
        self.originalBbox = _oriBBox
        self.conf = _conf
        self.className = _className
        self.id = _id

    def collidesWith(self, _other):
        minX, minY, maxX, maxY = self.bbox
        otherMinX, otherMinY, otherMaxX, otherMaxY = _other.bbox
        x_intersect = (minX < otherMaxX) and (otherMinX < maxX)
        y_intersect = (minY < otherMaxY) and (otherMinY < maxY)

        return x_intersect and y_intersect
    
    def ConvertBboxToCenterWidthHeight(self):
        # Extract coordinates from bbox
        minX, minY, maxX, maxY = self.bbox

        # Calculate height h
        h = maxY - minY
        w = maxX - minX

        # Calculate center coordinates (c_x, c_y)
        c_x = minX + w/2
        c_y = minY + h/2

        w = h = np.where(w / h > 1, w, h)

        # Return as numpy array in shape (4,)
        return [c_x, c_y, w, h]

'''
    Objects that a human can interact with
'''
class HumanInteractableObject(DetectedObj):
    
    def __init__(self, _id, _bbox, _oriBBox, _conf, _className):
        super().__init__(_id, _bbox, _oriBBox, _conf, _className)

        self.isAttached = False
        self.attachedToObjId = -1 # id of object it is attached to
        self.boneAttached = -1 # which bone is it attached to

        c_x, c_y, w, h = self.ConvertBboxToCenterWidthHeight()
        self.renderPoint = [c_x, c_y]
        self.width = w
        self.heigh = h
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