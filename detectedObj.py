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
