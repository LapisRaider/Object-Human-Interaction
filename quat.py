from __future__ import annotations

import math

import maths_util
from vec3 import Vec3
from mtx import Mtx

class Quat:
    @classmethod
    def from_axis_angle(cls, axis=Vec3(1.0, 0.0, 0.0), angle=0.0):
        w = math.cos(angle * 0.5)
        x = axis.x * math.sin(angle * 0.5)
        y = axis.y * math.sin(angle * 0.5)
        z = axis.z * math.sin(angle * 0.5)
        return cls(w, x, y, z)

    def rotate_via_quaternion(point : Vec3, rotation : Quat) -> Vec3:
        result = rotation * Quat(0.0, point.x, point.y, point.z) * rotation.conjugated()
        return Vec3(result.x, result.y, result.z)
    
    def rotate_via_axis_angle(point : Vec3, axis : Vec3, angle : float) -> Vec3:
        return Quat.rotate_via_quaternion(point, Quat.from_axis_angle(axis, angle))

    def __init__(self, w, x, y, z):
        self.w = w # Scalar component.
        self.x = x # Vector component X.
        self.y = y # Vector component Y.
        self.z = z # Vector component Z.

    def is_unit(self):
        return maths_util.approx_equal(1.0, self.length_squared())
    
    def is_zero(self):
        return maths_util.approx_equal(0.0, self.length_squared())

    def dot(self, rhs : Quat) -> float:
        return (self.w * rhs.w) + (self.x * rhs.x) + (self.y * rhs.y) + (self.z * rhs.z)

    def length_squared(self) -> float:
        return (self.w * self.w) + (self.x * self.x) + (self.y * self.y) + (self.z * self.z)
    
    def length(self) -> float:
        return math.sqrt(self.length_squared())

    def normalised(self) -> Quat:
        length = self.length()
        if maths_util.approx_equal(length, 0.0):
            return Quat(0.0, 0.0, 0.0, 0.0)
        return Quat(self.w / length, self.x / length, self.y / length, self.z / length)

    def conjugated(self) -> Quat:
        return Quat(self.w, -self.x, -self.y, -self.z)
    
    def inversed(self) -> Quat:
        return self.conjugated() * (1.0 / self.length_squared())

    def to_axis_angle(self) -> tuple[Vec3, float]:
        xyz = Vec3(self.x, self.y, self.z)
        if xyz.iszero():
            axis = Vec3(1.0, 0.0, 0.0)
            angle = 0.0
            return (axis, angle)

        axis = xyz.normalised()
        angle = math.acos(self.w) * 2.0
        return (axis, angle)
    
    def to_rotation_matrix(self) -> Mtx:
        mtx = Mtx(4, 4)
        mtx[0, 0] = 1.0 - 2.0 * self.y * self.y - 2.0 * self.z * self.z
        mtx[0, 1] = 2.0 * self.x * self.y + 2.0 * self.w * self.z
        mtx[0, 2] = 2.0 * self.x * self.z - 2.0 * self.w * self.y
        mtx[0, 3] = 0.0

        mtx[1, 0] = 2.0 * self.x * self.y - 2.0 * self.w * self.z
        mtx[1, 1] = 1.0 - 2.0 * self.x * self.x - 2.0 * self.z * self.z
        mtx[1, 2] = 2.0 * self.y * self.z + 2.0 * self.w * self.x
        mtx[1, 3] = 0.0
        
        mtx[2, 0] = 2.0 * self.x * self.z + 2.0 * self.w * self.y
        mtx[2, 1] = 2.0 * self.y * self.z - 2.0 * self.w * self.x
        mtx[2, 2] = 1.0 - 2.0 * self.x * self.x - 2.0 * self.y * self.y
        mtx[2, 3] = 0.0

        mtx[3, 0] = 0.0
        mtx[3, 1] = 0.0
        mtx[3, 2] = 0.0
        mtx[3, 3] = 1.0
        return mtx
    
    # Operator overloads.
    def __mul__(self, rhs) -> Quat:
        # Quaternion multiplication.
        if isinstance(rhs, self.__class__):
            lhs_vec = Vec3(self.x, self.y, self.z)
            rhs_vec = Vec3(rhs.x, rhs.y, rhs.z)
            w = self.w * rhs.w - lhs_vec.dot(rhs_vec)
            xyz = self.w * rhs_vec + rhs.w * lhs_vec + lhs_vec.cross(rhs_vec)
            return Quat(w, xyz.x, xyz.y, xyz.z)
        # Scalar multiplication.
        else:
            return Quat(self.w * rhs, self.x * rhs, self.y * rhs, self.z * rhs)
        
    # Scalar multiplication.
    def __rmul__(self, lhs) -> Quat:
        return Quat(self.w * lhs, self.x * lhs, self.y * lhs, self.z * lhs)
    
    def __eq__(self, rhs) -> bool:
        return maths_util.approx_equal(self.w, rhs.w) and maths_util.approx_equal(self.x, rhs.x) and maths_util.approx_equal(self.y, rhs.y) and maths_util.approx_equal(self.z, rhs.z)
    
    def __ne__(self, rhs) -> bool:
        return not (self == rhs)
    
    def __repr__(self):
        return "(" + str(self.w) + ", " + str(self.x) + ", " + str(self.y) + ", " + str(self.z) + ")"