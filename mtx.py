from __future__ import annotations

import math

import maths_util

# !!!!!!!!!!!!!!!!SUPER IMPORTANT!!!!!!!!!!!!!!!!
# Column-major Matrix
# NOT ROW-MAJOR! OPPOSITE OF NUMPY!
class Mtx:
    @classmethod
    def identity(cls, cols : int) -> Mtx:
        mat = cls(cols, cols)
        for i in range(cols):
            mat[i, i] = 1
        return mat
    
    @classmethod
    def from_array(cls, cols : int, rows : int, array : list[int]) -> Mtx:
        mat = cls(cols, rows)
        for i in range(cols * rows):
            mat._val[i] = array[i]
        return mat

    def __init__(self, cols : int, rows : int):
        self._cols = cols
        self._rows = rows
        self._val = [0] * (cols * rows)

    def transposed(self) -> Mtx:
        mtx = Mtx(self._rows, self._cols)
        for i in range(self._cols):
            for j in range(self._rows):
                mtx[j, i] = self[i, j]
        return mtx

    # Operator overloads.
    def __getitem__(self, key : int) -> float:
        col = key[0]
        row = key[1]
        return self._val[col * self._rows + row]
    
    def __setitem__(self, key : int, value : float):
        col = key[0]
        row = key[1]
        self._val[col * self._rows + row] = value

    def __add__(self, rhs : Mtx) -> Mtx:
        mtx = Mtx(self._cols, self._rows)
        for i in range(self._cols * self._rows):
            mtx._val[i] = self._val[i] + rhs._val[i]
        return mtx
    
    def __iadd__(self, rhs : Mtx) -> Mtx:
        for i in range(self._cols * self._rows):
            self._val[i] = self._val[i] + rhs._val[i]
        return self

    def __sub__(self, rhs : Mtx) -> Mtx:
        mtx = Mtx(self._cols, self._rows)
        for i in range(self._cols * self._rows):
            mtx._val[i] = self._val[i] - rhs._val[i]
        return mtx
    
    def __isub__(self, rhs : Mtx) -> Mtx:
        for i in range(self._cols * self._rows):
            self._val[i] = self._val[i] - rhs._val[i]
        return self

    def __mul__(self, rhs : Mtx) -> Mtx:
        # Matrix multiplication. Fucking Python is disgusting.
        # Why can't we have types like a damn sane language?!
        if isinstance(rhs, self.__class__):
            mtx = Mtx(rhs._cols, self._rows)
            for i in range(rhs._cols):
                for j in range(self._rows):
                    for k in range(self._cols):
                        mtx[i, j] = mtx[i, j] + self[k, j] * rhs[i, k]
        # Scalar multiplication.
        else:
            mtx = Mtx(self._cols, self._rows)
            for i in range(self._cols * self._rows):
                mtx._val[i] = self._val[i] * rhs
        return mtx
    
    def __eq__(self, rhs : Mtx) -> bool:
        if self._cols != rhs._cols:
            return False
        if self._rows != rhs._rows:
            return False
        
        for i in range(self._cols * self._rows):
            if not maths_util.approx_equal(self._val[i], rhs._val[i]):
                return False
        return True
    
    def __ne__(self, rhs : Mtx) -> bool:
        return not (self == rhs)