from scipy.optimize import fsolve
import numpy as np


class Point(object):
    def __init__(self, x, y):
        self.x = x
        self.y = y


def get_radian(x, y):
    v_arctan = np.arctan(np.divide(y, x))
    if x >= 0 and y >= 0:
        return v_arctan
    elif x >= 0 and y < 0:
        return 2 * np.pi + v_arctan
    else:
        return np.pi + v_arctan


def solve_rotating_parabola(*points):
    p1, p2, p3 = points
    # check p1-p2-p3 is anticlockwise
    cross_product = (p2.x - p1.x) * (p3.y - p2.y) - (p2.y - p1.y) * (p3.x - p2.x)
    assert cross_product > 0, "we need p1-p2-p3 is anti-clockwise"

    # get angle of p3-p2
    theta = get_radian(p3.x - p1.x, p3.y - p1.y)
