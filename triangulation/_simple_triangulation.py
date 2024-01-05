from __future__ import annotations

import numpy as np

from .triangle import Triangle
from .utils import (
    min_angle,
    is_clockwise_orientation,
    is_convex,
    ray_line_segment_intersection,
    line_segment_intersection
)


class Triangulation:
    def __init__(self, polygon, clockwise_orientation) -> None:
        self.polygon = polygon
        self.clockwise_orientation = clockwise_orientation

    @classmethod
    def init_polygon(cls, polygon):
        clockwise_orientation = is_clockwise_orientation(polygon)
        return cls(polygon, clockwise_orientation)
    
    def _dist_point_to_polygon(self, point: np.ndarray, direction: np.ndarray, polygon: list[np.ndarray]):
        size = len(polygon)
        min_dist = float("inf")
        
        for i in range(len(polygon)):
            edge = [polygon[i], polygon[(i + 1) % size]]
            intersection_point = ray_line_segment_intersection(point, direction, edge)
            
            if intersection_point is None:
                continue
            
            dist = np.linalg.norm(intersection_point - point)

            if np.isclose(dist, 0):
                continue
            elif dist < min_dist:
                min_dist = dist 
        
        return min_dist
        
    def _has_line(self, line: np.ndarray, polygon: list[np.ndarray]):
        size = len(polygon)
        intersection_cnt = 0

        for i in range(len(polygon)):
            edge = polygon[i], polygon[(i + 1) % size]
            point = line_segment_intersection(line, edge)
            if point is not None:
                intersection_cnt += 1
                if intersection_cnt > 2:
                    return False
                
        return True

    def _new_triangles(
        self, 
        prev: np.ndarray, 
        current: np.ndarray, 
        next: np.ndarray, 
        polygon: list[np.ndarray],
        factor: float 
    ):
        vec1 = (prev - current)/np.linalg.norm(prev - current)
        vec2 = (next - current)/np.linalg.norm(next - current)
        bisection = 0.5*(vec1 + vec2)
        bisection = bisection/np.linalg.norm(bisection)
        
        dist = self._dist_point_to_polygon(current, bisection, polygon)
        new_point = current + bisection*dist/factor
        step = (dist/factor)/100

        while not (
            self._has_line([prev, new_point], polygon) 
            and self._has_line([next, new_point], polygon)
        ):
            new_point = new_point - bisection*step 
            
        triangle1 = Triangle([current, new_point, prev])
        triangle2 = Triangle([current, new_point, next])

        return triangle1, triangle2, new_point

    def init_triangulation(self):
        self.triangles = []
        self.inner_points = []

        polygon = self.polygon.copy()

        while len(polygon) > 2:
            point, angle = min_angle(polygon, self.clockwise_orientation)
            prev = polygon[(point - 1) % len(polygon)]
            current = polygon[point]
            next = polygon[(point + 1) % len(polygon)]
            
            triangle = Triangle([prev, current, next])
            adj_angle1 = triangle.get_angle(prev)
            adj_angle2 = triangle.get_angle(next)

            if (
                (angle < 5*np.pi/12)
                or (
                    5*np.pi/12 <= angle < np.pi/2
                    and adj_angle1 >= np.pi/6
                    and adj_angle2 >= np.pi/6
                )
            ):
                self.triangles.append(triangle)
                del polygon[point]
            else:
                t1, t2, new_point = self._new_triangles(prev, current, next, polygon, 2)
                self.triangles.extend([t1, t2])
                self.inner_points.append(new_point)
                polygon[point] = new_point

    def align_inner_points(self):
        if not self.inner_points:
            return None
        
        new_triangles = self.triangles.copy()
        new_inner_points = []
        for point in self.inner_points:
            triangles = [
                triangle for triangle in self.triangles 
                if triangle.has_vertex(point)
            ]
            adj_points = [
                tuple(adj_point) for triangle in triangles 
                for adj_point in triangle.get_adjacent_vertices(point)
            ]
            adj_points = np.array(list(set(adj_points)))
            new_point = np.sum(adj_points, axis=0)/adj_points.shape[0]
            new_inner_points.append(new_point)

            for triangle in new_triangles:
                triangle.replace_vertex(point, new_point)

        self.triangles = new_triangles
        self.inner_points = new_inner_points
            
    def align_triangles(self):
        new_triangles = self.triangles.copy()

        for i in range(len(new_triangles)):
            for j in range(i + 1, len(new_triangles)):
                triangle1, triangle2 = new_triangles[i], new_triangles[j]
                edge = triangle1.get_common_edge(triangle2)

                if edge is None:
                    continue

                vertex1 = triangle1.get_vertex_by_edge(edge)
                vertex2 = triangle2.get_vertex_by_edge(edge)

                if not is_convex([vertex1, edge[0], vertex2, edge[1]]):
                    continue
                if np.isclose(np.cross(vertex1 - vertex2, edge[0] - vertex2),0):
                    continue
                if np.isclose(np.cross(vertex1 - vertex2, edge[1] - vertex2),0):
                    continue
                
                angles_prev1 = sorted([
                    triangle1.get_angle(edge[0]), 
                    triangle1.get_angle(edge[1]),
                    triangle1.get_angle(vertex1)
                ])
                angles_prev2 = sorted([
                    triangle2.get_angle(edge[0]), 
                    triangle2.get_angle(edge[1]),
                    triangle2.get_angle(vertex2)
                ])

                new_triangle1 = Triangle([vertex1, edge[0], vertex2])
                new_triangle2 = Triangle([vertex1, edge[1], vertex2])

                angles_new1 = sorted([
                    new_triangle1.get_angle(vertex1), 
                    new_triangle1.get_angle(edge[0]),
                    new_triangle1.get_angle(vertex2)
                ])
                angles_new2 = sorted([
                    new_triangle2.get_angle(vertex1), 
                    new_triangle2.get_angle(edge[1]),
                    new_triangle2.get_angle(vertex2)
                ])

    
                sum_prev = sum(angles_prev1[:2] + angles_prev2[:2])
                sum_new = sum(angles_new1[:2] + angles_new2[:2])

                if sum_new >= sum_prev:
                    new_triangles[i] = new_triangle1
                    new_triangles[j] = new_triangle2
        
        self.triangles = new_triangles

    def align_triangulation(
        self, 
        total: int = 2, 
        inner_points: int = 5,
        triangles: int = 5,
        last_inner_points: int = 5
    ):
        for _ in range(total):
            for i in range(inner_points):
                self.align_inner_points()
            for j in range(triangles):
                self.align_triangles()

        for _ in range(last_inner_points):
            self.align_inner_points()

    @classmethod
    def create(
        cls, 
        polygon: list[np.ndarray],
        total: int = 2, 
        inner_points: int = 5,
        triangles: int = 5,
        last_inner_points: int = 5
    ) -> list[Triangle]:
        triangulation = cls.init_polygon(polygon)
        triangulation.init_triangulation()
        triangulation.align_triangulation(total, inner_points, triangles, last_inner_points)

        return triangulation.triangles 