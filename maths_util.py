epsilon = 0.00001
pi = 3.141592653589793
rad2deg = 180.0 / pi
deg2rad = pi / 180.0

def approx_equal(a, b):
    return abs(a - b) < epsilon