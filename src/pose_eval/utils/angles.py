import math
from typing import Iterable


def normalize_angle(angle: float) -> float:
    """Normalizes an angle into the range [-pi, pi)"""
    return (angle + math.pi) % (2 * math.pi) - math.pi

def angular_distance(a: float, b: float) -> float:
    return abs((a - b + math.pi) % (2 * math.pi) - math.pi)

def signed_angular_distance(a: float, b: float) -> float:
    return (a - b + math.pi) % (2 * math.pi) - math.pi

def average_angle(angles: Iterable[float]) -> float:
    sum_sin = sum(math.sin(h) for h in angles)
    sum_cos = sum(math.cos(h) for h in angles)
    return math.atan2(sum_sin, sum_cos)

