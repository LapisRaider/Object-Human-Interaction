import math

from vec3 import Vec3

def screen_to_world_height(screenspace_height : float, resolution_y : float, fov : float, ws_pos_z : float):
    # In NDC, the Y axis range is [-1, 1].
    ndc_height = 2.0 * screenspace_height / resolution_y # Normalise the size of the object into the [-1, 1] range.
    return -(ndc_height * ws_pos_z) * math.tan(fov*0.5)

def screen_to_world_xy(fov : float, resolution_x : float, resolution_y : float, ss_pos_x : float, ss_pos_y : float, ws_pos_z : float):
    # Map the screen coordinate to NDC, which is [-1, 1].
    aspect_ratio = float(resolution_x)/float(resolution_y)

    ndc_x = -(ss_pos_x / float(resolution_x) * 2.0 - 1.0)
    ndc_y = ss_pos_y / float(resolution_y) * 2.0 - 1.0

    # Convert from NDC to world coordinate.
    ws_pos_x = ndc_x * ws_pos_z * math.tan(0.5 * fov) * aspect_ratio
    ws_pos_y = ndc_y * ws_pos_z * math.tan(0.5 * fov)

    return Vec3(ws_pos_x, ws_pos_y, ws_pos_z)

def world_to_screen(fov : float, resolution_x : float, resolution_y : float, world_pos_x : float, world_pos_y : float, world_pos_z : float):
    aspect_ratio = float(resolution_x)/float(resolution_y)

    # convert world to NDC
    ndc_x = world_to_ndc_x(fov, aspect_ratio, world_pos_x, world_pos_z)
    ndc_y = world_to_ndc_y(fov, world_pos_y, world_pos_z)

    # convert ndc to screen
    ss_pos_x = ((ndc_x + 1.0) / 2.0) * resolution_x
    ss_pos_y = ((ndc_y + 1.0) / 2.0) * resolution_y
    
    return Vec3(ss_pos_x, ss_pos_y, 0.0)


def screen_to_ndc_x(resolution_x : float, ss_pos_x : float):
    return -(ss_pos_x / resolution_x * 2.0 - 1.0)

def screen_to_ndc_y(resolution_y : float, ss_pos_y : float):
    return ss_pos_y / resolution_y * 2.0 - 1.0

def world_to_ndc_x(fov : float, aspect_ratio : float, ws_pos_x : float, ws_pos_z : float):
    return -ws_pos_x / (ws_pos_z * math.tan(fov * 0.5) * aspect_ratio)

def world_to_ndc_y(fov : float, ws_pos_y : float, ws_pos_z : float):
    return ws_pos_y / (ws_pos_z * math.tan(fov * 0.5))