from __future__ import annotations

import numpy as np

from .utils import (
    angle_vec,
    is_clockwise_orientation
)

class Triangle:
    clockwise = False

    def __init__(self, points) -> None:
        self.points = points.copy()
        clockwise = is_clockwise_orientation(points)

        if clockwise != self.clockwise:
            self.points.reverse()

    def replace_vertex(self, vertex, new_vertex):
        for ind, point in enumerate(self.points):
            if np.array_equal(point, vertex):
                self.points[ind] = new_vertex
                break
    
    def has_point(self, point: np.ndarray) -> bool:
        sign = 0

        for i in range(3):
            vec1 = self.points[(i + 1) % 3] - self.points[i]
            vec2 = point - self.points[i]
            product = np.cross(vec1, vec2)

            if np.isclose(product, 0):
                continue
            elif sign == 0:
                sign = np.sign(product)
            elif sign != np.sign(product):
                return False
        
        return True

    def has_vertex(self, vertex: np.ndarray) -> bool:
        for point in self.points:
            if np.array_equal(point, vertex):
                return True
        
        return False

    def get_angle(self, vertex: np.ndarray) -> float:
        for ind, point in enumerate(self.points):
            if np.array_equal(point, vertex):
                v1 = self.points[(ind - 1) % 3]
                v2 = self.points[(ind + 1) % 3]
                vec1 = v1 - vertex
                vec2 = v2 - vertex

                if self.clockwise:
                    return angle_vec(vec1, vec2)

                return angle_vec(vec2, vec1)

    def get_common_edge(self, triangle: Triangle) -> list[np.ndarray]:
        edge = [
            p1 for p1 in self.points
            for p2 in triangle.points
            if np.array_equal(p1, p2)
        ]
        
        if len(edge) == 2:
            return edge
        
        return None

    def get_height(self, vertex: np.ndarray) -> float:
        for ind, point in enumerate(self.points):
            if np.array_equal(point, vertex):
                v1 = self.points[(ind - 1) % 3]
                v2 = self.points[(ind + 1) % 3]
                vec1 = v1 - vertex
                vec2 = v2 - vertex

                if self.clockwise:
                    det = np.cross(vec1, vec2)
                else:
                    det = np.cross(vec2, vec1)
                
                return det/np.linalg.norm(v1 - v2)

    def get_vertex_by_edge(self, edge: list[np.ndarray]):
        for point in self.points:
            if (
                not ( 
                    np.array_equal(point, edge[0])
                    or np.array_equal(point, edge[1])
                )
            ): 
                return point

    def get_adjacent_vertices(self, vertex: np.ndarray) -> float:
        for ind, point in enumerate(self.points):
            if np.array_equal(point, vertex):
                return self.points[(ind - 1) % 3], self.points[(ind + 1) % 3] 