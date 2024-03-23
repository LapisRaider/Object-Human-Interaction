import math

from vec3 import Vec3
from mtx import Mtx

def scale_matrix(scale=Vec3(1.0, 1.0, 1.0)):
    mtx = Mtx(4, 4)
    mtx[0, 0] = scale.x
    mtx[1, 1] = scale.y
    mtx[2, 2] = scale.z
    mtx[3, 3] = 1.0
    return mtx

def translation_matrix(translation=Vec3(0.0, 0.0, 0.0)):
    '''
    Translation Matrix
    |   1   0   0   x   |
    |   0   1   0   y   |
    |   0   0   1   z   |
    |   0   0   0   1   |
    '''
    mtx = Mtx.identity(4)
    mtx[3, 0] = translation.x
    mtx[3, 1] = translation.y
    mtx[3, 2] = translation.z
    return mtx

def rotation_matrix_x(angle=0.0):
    '''
    Rotation Matrix on X-axis
    |   1   0   0   0   |
    |   0  cos -sin 0   |
    |   0  sin  cos 0   |
    |   0   0   0   1   |
    '''
    mtx = Mtx.identity(4)
    mtx[1, 1] = math.cos(angle)
    mtx[2, 2] = mtx[1, 1]
    mtx[1, 2] = math.sin(angle)
    mtx[2, 1] = -mtx[1, 2]
    return mtx

def rotation_matrix_y(angle=0.0):
    '''
    Rotation Matrix on Y-axis
    |  cos  0  sin  0   |
    |   0   1   0   0   |
    | -sin  0  cos  0   |
    |   0   0   0   1   |
    '''
    mtx = Mtx.identity(4)
    mtx[0, 0] = math.cos(angle)
    mtx[2, 2] = mtx[0, 0]
    mtx[2, 0] = math.sin(angle)
    mtx[0, 2] = -mtx[2,0]
    return mtx

# Angle in radians.
def rotation_matrix_z(angle=0.0):
    '''
    Rotation Matrix on Y-axis
    |  cos -sin 0   0   |
    |  sin  cos 0   0   |
    |   0   0   1   0   |
    |   0   0   0   1   |
    '''
    mtx = Mtx.identity(4)
    mtx[0, 0] = math.cos(angle)
    mtx[1, 1] = mtx[0, 0]
    mtx[0, 1] = math.sin(angle)
    mtx[1, 0] = -mtx[0, 1]
    return mtx

# Rotation Order: Z-Y-X
def rotation_matrix(euler_angles=Vec3(0.0, 0.0, 0.0)):
    return rotation_matrix_x(euler_angles.x) * rotation_matrix_y(euler_angles.y) * rotation_matrix_z(euler_angles.z)

def view_matrix(position=Vec3(0.0, 0.0, 0.0), forward=Vec3(0.0, 0.0, 1.0), up=Vec3(0.0, 1.0, 0.0)):
    b = -forward.normalised() # Backward
    r = forward.cross(up).normalised() # Right
    u = b.cross(r) # Up
    
    mtx = Mtx.identity(4)
    mtx[0, 0] = r.x
    mtx[1, 0] = r.y
    mtx[2, 0] = r.z

    mtx[0, 1] = u.x
    mtx[1, 1] = u.y
    mtx[2, 1] = u.z

    mtx[0, 2] = b.x
    mtx[1, 2] = b.y
    mtx[2, 2] = b.z

    mtx[3, 0] = -position.dot(r)
    mtx[3, 1] = -position.dot(u)
    mtx[3, 2] = -position.dot(b)
    return mtx

def perspective_matrix(aspect_ratio : float, fov : float, near : float, far : float):
    '''
    | 1/(tan(FOV/2) * AR)         0                0              0       |
    |          0            1/tan(FOV/2)           0              0       |
    |          0                  0           (N+F)/(N-F)    (2*N*F)(N-F) |
    |          0                  0               -1              0       |
    '''
    tan_fov = math.tan(fov * 0.5)

    mtx = Mtx(4, 4)
    mtx[0, 0] = 1.0 / (aspect_ratio * tan_fov)
    mtx[1, 1] = 1.0 / tan_fov
    mtx[2, 2] = (near + far) / (near - far)
    mtx[3, 2] = (2.0 * near * far) / (near - far)
    mtx[2, 3] = -1.0
    return mtx

# TODO: Matrix Inverse