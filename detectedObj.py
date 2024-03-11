class DetectedObj:
    def __init__(self, _id, _bbox, _oriBBox, _conf, _className):
        self.bbox = _bbox # in xyxy format
        self.originalBbox = _oriBBox
        self.conf = _conf
        self.className = _className
        self.id = _id

    def collidesWith(self, other):
        x_intersect = (self.minX < other.maxX) and (other.minX < self.maxX)
        y_intersect = (self.minY < other.maxY) and (other.minY < self.maxY)

        return x_intersect and y_intersect
