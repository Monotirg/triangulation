import numpy as np 

from .triangle import Triangle
from .utils import (
    is_clockwise_orientation,
    angle_vec
)

def ear_clipping_triangulation(polygon: list[np.ndarray]) -> list[Triangle]:
    polygon = polygon.copy()
    clockwise = is_clockwise_orientation(polygon)
    triangulation = []
    current = 0

    while len(polygon) > 2:
        prev = (current - 1) % len(polygon)
        next = (current + 1) % len(polygon)
        vec1 = polygon[prev] - polygon[current]
        vec2 = polygon[next] - polygon[current]

        if clockwise:
            angle = angle_vec(vec1, vec2)
        else:
            angle = angle_vec(vec2, vec1)

        if angle >= np.pi:
            current = (current + 1) % len(polygon)
            continue

        triangle = Triangle([polygon[prev], polygon[current], polygon[next]])
        has_another_point = False
        
        for ind, point in enumerate(polygon):
            if ind in [prev, current, next]:
                continue
            if triangle.has_point(point):
                has_another_point = True
                break
        
        if not has_another_point:
            triangulation.append(triangle)
            del polygon[current]

        current = (current + 1) % len(polygon)

    return triangulation