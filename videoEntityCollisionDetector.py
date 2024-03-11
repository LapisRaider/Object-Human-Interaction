from utils import DrawBox
import random

class VideoEntityCollisionDetector:

    # entities to check with all other entities
    def __init__(self, _entityClasses):
        self.entityClasses = _entityClasses;

    """
        Parameters
        _allEntities: array of detected obj DetectedObj
    """
    def CheckCollision(self, _allEntities):
        mainEntities = []
        otherEntities = []

        collision = {}

        for entity in _allEntities:
            if entity.className in self.entityClasses:
                mainEntities.append(entity)
            else:
                otherEntities.append(entity)

        for mainEntity in mainEntities:
            collision[mainEntity] = []
            for other in otherEntities:
                if mainEntity.collidesWith(other):
                    collision[mainEntity].append(other)

        return collision


    def Draw(self, _frame, _collisionEntities):
        for entity, others in _collisionEntities.items():
            random.seed(entity.id)
            color = [random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)]

            DrawBox(_frame, entity.bbox, color, 2)
            for other in others:
                DrawBox(_frame, other.bbox, color, 2)


    
        



    