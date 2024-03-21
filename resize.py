import math

def get_world_height(screenspace_height, resolution_y, fov, pos_z):
    # In NDC, the Y axis range is [-1, 1].
    ndc_height = 2.0 * screenspace_height / resolution_y # Normalise the size of the object into the [-1, 1] range.
    return -(ndc_height * pos_z) * math.tan(fov*0.5)