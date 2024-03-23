from __future__ import annotations

import math

import maths_util

class Vec3:
    def zero() -> Vec3:
        return Vec3(0.0, 0.0, 0.0)
    
    def forward() -> Vec3:
        return Vec3(0.0, 0.0, 1.0)
    
    def backward() -> Vec3:
        return Vec3(0.0, 0.0, -1.0)
    
    def left() -> Vec3:
        return Vec3(1.0, 0.0, 0.0)
    
    def right() -> Vec3:
        return Vec3(-1.0, 0.0, 0.0)
    
    def up() -> Vec3:
        return Vec3(0.0, 1.0, 0.0)
    
    def down() -> Vec3:
        return Vec3(0.0, 1.0, 0.0)
    
    def x_axis() -> Vec3:
        return Vec3(1.0, 0.0, 0.0)
    
    def y_axis() -> Vec3:
        return Vec3(0.0, 1.0, 0.0)
    
    def z_axis() -> Vec3:
        return Vec3(0.0, 0.0, 1.0)

    def __init__(self, x : float, y : float, z : float):
        self.x = x
        self.y = y
        self.z = z

    def iszero(self) -> bool:
        return self.x == 0.0 and self.y == 0.0 and self.z == 0.0

    def dot(self, rhs : Vec3) -> float:
        return self.x * rhs.x + self.y * rhs.y + self.z * rhs.z

    def cross(self, rhs : Vec3) -> Vec3:
        x = self.y * rhs.z - self.z * rhs.y
        y = self.z * rhs.x - self.x * rhs.z
        z = self.x * rhs.y - self.y * rhs.x
        return Vec3(x, y, z)
    
    def length(self) -> float:
        return math.sqrt(self.length_squared())
    
    def length_squared(self) -> float:
        return (self.x * self.x) + (self.y * self.y) + (self.z * self.z)

    def normalised(self) -> Vec3:
        length = self.length()
        return Vec3(self.x / length, self.y / length, self.z / length)

    # Scalar multiplication.
    def __mul__(self, rhs : float) -> Vec3:
        return Vec3(self.x * rhs, self.y * rhs, self.z * rhs)
    
    # Scalar multiplication.
    __rmul__ = __mul__
    
    # Scalar multiplication.
    def __imul__(self, rhs : float):
        self.x = self.x * rhs
        self.y = self.y * rhs
        self.z = self.z * rhs
        return self
    
    def __add__(self, rhs : Vec3) -> Vec3:
        return Vec3(self.x + rhs.x, self.y + rhs.y, self.z + rhs.z)
    
    def __iadd__(self, rhs : Vec3) -> Vec3:
        self.x = self.x + rhs.x
        self.y = self.y + rhs.y
        self.z = self.z + rhs.z
        return self
    
    def __sub__(self, rhs : Vec3) -> Vec3:
        return Vec3(self.x - rhs.x, self.y - rhs.y, self.z - rhs.z)
    
    def __isub__(self, rhs : Vec3) -> Vec3:
        self.x = self.x - rhs.x
        self.y = self.y - rhs.y
        self.z = self.z - rhs.z
        return self
    
    def __neg__(self) -> Vec3:
        return Vec3(-self.x, -self.y, -self.z)

    def __eq__(self, rhs : Vec3) -> bool:
        return maths_util.approx_equal(self.x, rhs.x) and maths_util.approx_equal(self.y, rhs.y) and maths_util.approx_equal(self.z, rhs.z)
    
    def __ne__(self, rhs : Vec3) -> bool:
        return not (self == rhs)
    
    def __repr__(self):
        return "(" + str(self.x) + ", " + str(self.y) + ", " + str(self.z) + ")"