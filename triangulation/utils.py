import numpy as np 


def angle_vec(vec1: np.ndarray, vec2: np.ndarray) -> float:
    angle = np.arctan2(vec2[1], vec2[0]) - np.arctan2(vec1[1], vec1[0])
    
    if angle < 0:
        return 2*np.pi + angle
    
    return angle 

def min_angle(polygon: list[np.ndarray], clockwise: bool) -> tuple[int, float]:
    point, min_angle = 0, np.pi
    size = len(polygon)
        
    for i in range(len(polygon)):
        vec1 = polygon[(i + 1) % size] - polygon[i]
        vec2 = polygon[(i - 1) % size] - polygon[i]
        
        if clockwise:
            angle = angle_vec(vec2, vec1)
        else:
            angle = angle_vec(vec1, vec2)

        if angle < min_angle:
            min_angle = angle
            point = i 

    return point, min_angle

def is_convex(polygon: list[np.ndarray]) -> bool:
    size = len(polygon)
    sign = 0 

    for i in range(len(polygon)):
        vec1 = polygon[(i + 1) % size] - polygon[i]
        vec2 = polygon[i] - polygon[(i - 1) % size]
        product = np.cross(vec1, vec2)

        if np.isclose(product, 0):
            continue
        elif sign == 0:
            sign = np.sign(product)
        elif sign != np.sign(product):
            return False 
    
    return True

def is_clockwise_orientation(polygon: list[np.ndarray]) -> bool:
    left_point = 0
    size = len(polygon)
    
    for i in range(1, size):
        if polygon[i][0] < polygon[left_point][0]:
            left_point = i
        elif np.isclose(polygon[i][0], polygon[left_point][0]):
            if polygon[i][1] < polygon[left_point][1]:
                left_point = i
    
    vec1 = polygon[(left_point + 1) % size] - polygon[left_point]
    vec2 = polygon[(left_point - 1) % size] - polygon[left_point]
    
    if np.cross(vec1, vec2) < 0:
        return True
    
    return False

def ray_line_segment_intersection(
    point: np.ndarray, 
    direction: np.ndarray, 
    line_segment: list[np.ndarray]
) -> np.ndarray | None:
    s = np.vstack(([point, point + direction], line_segment))
    h = np.hstack((s, np.ones((4, 1))))
    line1 = np.cross(h[0], h[1])
    line2 = np.cross(h[2], h[3])
    x, y, z = np.cross(line1, line2)

    if np.isclose(z, 0):
        return None

    point = np.array([x / z, y / z])
    
    vec1 = line_segment[0] - point
    vec2 = line_segment[1] - point
    product = np.dot(vec1, vec2)

    if np.isclose(product, 0) or product < 0:
        return point
    
    return None

def line_segment_intersection(line1: np.ndarray, line2: np.ndarray) -> np.ndarray | None:
    point = ray_line_segment_intersection(
        line1[0], 
        (line1[1] - line1[0])/np.linalg.norm(line1[1] - line1[0]), 
        line2
    )

    if point is None:
        return None
    
    vec1 = line1[0] - point
    vec2 = line1[1] - point
    product = np.dot(vec1, vec2)

    if np.isclose(product, 0) or product < 0:
        return point
    
    return None