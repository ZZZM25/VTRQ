def orientation(p, q, r):
    """
    This function is used to determine the orientation relationship of three points p, q, r.
    The relative positions of the three points are determined by calculating the vector cross product, 
    and the result can be collinear, clockwise, or counterclockwise.
    :param p: The first point in the format (x, y)
    :param q: The second point in the format (x, y)
    :param r: The third point in the format (x, y)
    :return: 0 for collinear, 1 for clockwise, 2 for counterclockwise
    """
    # Calculate the vector cross product
    val = (q[1] - p[1]) * (r[0] - q[0]) - (q[0] - p[0]) * (r[1] - q[1])
    if val == 0:
        return 0
    elif val > 0:
        return 1
    return 2


def on_segment(p, q, r):
    """
    Determine if point q lies on the line segment pr.
    Judgment is made by comparing whether the coordinates of point q are within the coordinate range of p and r.
    :param p: One endpoint of the line segment in the format (x, y)
    :param q: The point to be judged in the format (x, y)
    :param r: The other endpoint of the line segment in the format (x, y)
    :return: True if point q is on segment pr, False otherwise
    """
    return (min(p[0], r[0]) <= q[0] <= max(p[0], r[0]) and
            min(p[1], r[1]) <= q[1] <= max(p[1], r[1]))


def do_intersect(p1, q1, p2, q2):
    """
    Determine if two line segments p1q1 and p2q2 intersect.
    First, judge the orientation relationship of points through the orientation function, 
    then handle the special case of collinear and overlapping segments.
    :param p1: One endpoint of the first line segment in the format (x, y)
    :param q1: The other endpoint of the first line segment in the format (x, y)
    :param p2: One endpoint of the second line segment in the format (x, y)
    :param q2: The other endpoint of the second line segment in the format (x, y)
    :return: True if the two line segments intersect, False otherwise
    """
    # Calculate four orientation values to determine the relative positions of points
    o1 = orientation(p1, q1, p2)
    o2 = orientation(p1, q1, q2)
    o3 = orientation(p2, q2, p1)
    o4 = orientation(p2, q2, q1)

    # Normal intersection case: endpoints of each segment are on opposite sides of the other segment's line
    if o1 != o2 and o3 != o4:
        return True

    # Handle special cases of collinear and overlapping segments
    if o1 == 0 and on_segment(p1, p2, q1):
        return True
    if o2 == 0 and on_segment(p1, q2, q1):
        return True
    if o3 == 0 and on_segment(p2, p1, q2):
        return True
    if o4 == 0 and on_segment(p2, q1, q2):
        return True

    return False


def segment_intersects_rect(segment, rect):
    """
    Determine if a line segment intersects with a rectangle.
    Check intersection between the segment and each of the rectangle's four edges respectively.
    :param segment: Line segment in the format ((x1, y1), (x2, y2))
    :param rect: Rectangle in the format ((x_min, y_min), (x_max, y_max))
    :return: True if intersects, False otherwise
    """
    # Extract the coordinates of the rectangle's bottom-left and top-right corners
    x_min, y_min = rect[0]
    x_max, y_max = rect[1]
    # Extract the two endpoints of the line segment
    p1, q1 = segment
    # Define the four edges of the rectangle
    rect_edges = [
        ((x_min, y_min), (x_max, y_min)),  # Bottom edge
        ((x_max, y_min), (x_max, y_max)),  # Right edge
        ((x_max, y_max), (x_min, y_max)),  # Top edge
        ((x_min, y_max), (x_min, y_min))  # Left edge
    ]
    # Iterate through the four edges of the rectangle and check intersection with the segment
    for rect_edge in rect_edges:
        p2, q2 = rect_edge
        if do_intersect(p1, q1, p2, q2):
            return True
    return False


def test_segment_intersects_rect():
    # Test case 1: Segment intersects with rectangle
    segment1 = ((104.05057 ,30.68449 ),( 104.04993 ,30.68676))
    rect1 = ((104.05, 30.65), (104.06, 30.66))
    if segment_intersects_rect(segment1, rect1):
        print("Test passed: Segment intersects with rectangle")
    else:
        print("Test failed: Segment does not intersect with rectangle")




if __name__ == "__main__":
    test_segment_intersects_rect()
