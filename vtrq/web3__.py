from typing import Tuple, List, Union

# 定义坐标类型别名，增强可读性
Point = Tuple[float, float]
Segment = Tuple[Point, Point]
Rectangle = Tuple[Point, Point]

# 浮点数精度阈值（解决浮点运算误差问题）
EPS = 1e-9


def orientation(p: Point, q: Point, r: Point) -> int:
    """
    Determine the orientation relationship of three collinear/non-collinear points.
    Calculates cross product of vectors (q-p) and (r-q) to judge relative positions:
    - 0: Collinear (points lie on a straight line)
    - 1: Clockwise rotation
    - 2: Counterclockwise rotation
    
    :param p: First point (x, y)
    :param q: Second point (x, y)
    :param r: Third point (x, y)
    :return: Orientation code (0/1/2)
    """
    # Calculate vector cross product with precision tolerance
    val = (q[1] - p[1]) * (r[0] - q[0]) - (q[0] - p[0]) * (r[1] - q[1])
    if abs(val) < EPS:
        return 0  # Collinear (handle floating point precision)
    return 1 if val > 0 else 2


def on_segment(p: Point, q: Point, r: Point) -> bool:
    """
    Check if point q lies on line segment pr (including endpoints).
    Validates if q's coordinates are within the bounding rectangle of p and r,
    and confirms collinearity (via orientation check).
    
    :param p: Start endpoint of segment (x, y)
    :param q: Target point to check (x, y)
    :param r: End endpoint of segment (x, y)
    :return: True if q is on pr, False otherwise
    """
    if (min(p[0], r[0]) - EPS <= q[0] <= max(p[0], r[0]) + EPS and
        min(p[1], r[1]) - EPS <= q[1] <= max(p[1], r[1]) + EPS):
        return orientation(p, q, r) == 0
    return False


def do_intersect(p1: Point, q1: Point, p2: Point, q2: Point) -> bool:
    """
    Check if two line segments (p1q1 and p2q2) intersect (including endpoint overlap).
    Uses orientation tests to handle both general intersection and collinear overlap cases.
    
    :param p1: Start of first segment (x, y)
    :param q1: End of first segment (x, y)
    :param p2: Start of second segment (x, y)
    :param q2: End of second segment (x, y)
    :return: True if segments intersect, False otherwise
    """
    # Calculate orientation values for all endpoint combinations
    o1 = orientation(p1, q1, p2)
    o2 = orientation(p1, q1, q2)
    o3 = orientation(p2, q2, p1)
    o4 = orientation(p2, q2, q1)

    # Case 1: General intersection (endpoints on opposite sides of each segment)
    if o1 != o2 and o3 != o4:
        return True

    # Case 2: Collinear overlap (special case handling)
    if o1 == 0 and on_segment(p1, p2, q1):
        return True
    if o2 == 0 and on_segment(p1, q2, q1):
        return True
    if o3 == 0 and on_segment(p2, p1, q2):
        return True
    if o4 == 0 and on_segment(p2, q1, q2):
        return True

    # No intersection
    return False


def segment_intersects_rect(segment: Segment, rect: Rectangle) -> bool:
    """
    Check if a line segment intersects with an axis-aligned rectangle (AABB).
    Optimized with bounding box pre-check to avoid unnecessary edge tests.
    
    :param segment: Target line segment ((x1,y1), (x2,y2))
    :param rect: Axis-aligned rectangle ((x_min,y_min), (x_max,y_max))
    :return: True if segment intersects rect (including containment), False otherwise
    """
    # Extract rectangle bounds
    x_min, y_min = rect[0]
    x_max, y_max = rect[1]
    seg_p1, seg_p2 = segment

    # Pre-check 1: Segment is entirely inside the rectangle (fast path)
    def point_in_rect(point: Point) -> bool:
        return (x_min - EPS <= point[0] <= x_max + EPS and
                y_min - EPS <= point[1] <= y_max + EPS)
    
    if point_in_rect(seg_p1) and point_in_rect(seg_p2):
        return True

    # Pre-check 2: Bounding box of segment does not overlap with rect (early exit)
    seg_x_min = min(seg_p1[0], seg_p2[0])
    seg_x_max = max(seg_p1[0], seg_p2[0])
    seg_y_min = min(seg_p1[1], seg_p2[1])
    seg_y_max = max(seg_p1[1], seg_p2[1])
    
    if (seg_x_max < x_min - EPS or seg_x_min > x_max + EPS or
        seg_y_max < y_min - EPS or seg_y_min > y_max + EPS):
        return False

    # Define rectangle edges (bottom/right/top/left)
    rect_edges: List[Segment] = [
        ((x_min, y_min), (x_max, y_min)),  # Bottom edge
        ((x_max, y_min), (x_max, y_max)),  # Right edge
        ((x_max, y_max), (x_min, y_max)),  # Top edge
        ((x_min, y_max), (x_min, y_min))   # Left edge
    ]

    # Check intersection with each rectangle edge
    for edge in rect_edges:
        if do_intersect(seg_p1, seg_p2, edge[0], edge[1]):
            return True

    return False


# ------------------------------ Test Cases ------------------------------
def test_segment_intersects_rect():
    """Comprehensive test suite for segment-rectangle intersection."""
    test_cases = [
        # Case 1: No intersection (segment above rect)
        {
            "name": "Segment above rectangle",
            "segment": ((104.05057, 30.68449), (104.04993, 30.68676)),
            "rect": ((104.05, 30.65), (104.06, 30.66)),
            "expected": False
        },
        # Case 2: Segment crosses rectangle edge
        {
            "name": "Segment crosses rectangle edge",
            "segment": ((104.04, 30.655), (104.07, 30.655)),
            "rect": ((104.05, 30.65), (104.06, 30.66)),
            "expected": True
        },
        # Case 3: Segment entirely inside rectangle
        {
            "name": "Segment inside rectangle",
            "segment": ((104.052, 30.652), (104.058, 30.658)),
            "rect": ((104.05, 30.65), (104.06, 30.66)),
            "expected": True
        },
        # Case 4: Segment touches rectangle vertex
        {
            "name": "Segment touches rectangle vertex",
            "segment": ((104.05, 30.65), (104.04, 30.64)),
            "rect": ((104.05, 30.65), (104.06, 30.66)),
            "expected": True
        },
        # Case 5: Collinear with rectangle edge (overlap)
        {
            "name": "Segment collinear with rectangle edge",
            "segment": ((104.055, 30.65), (104.058, 30.65)),
            "rect": ((104.05, 30.65), (104.06, 30.66)),
            "expected": True
        }
    ]

    # Run tests
    passed = 0
    for case in test_cases:
        result = segment_intersects_rect(case["segment"], case["rect"])
        if result == case["expected"]:
            passed += 1
            print(f"✅ {case['name']}: Passed")
        else:
            print(f"❌ {case['name']}: Failed (expected {case['expected']}, got {result})")
    
    print(f"\nTotal tests: {len(test_cases)}, Passed: {passed}, Failed: {len(test_cases)-passed}")


if __name__ == "__main__":
    test_segment_intersects_rect()
