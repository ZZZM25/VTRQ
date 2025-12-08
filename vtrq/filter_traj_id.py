import csv
import json

from vector_cross_product import segment_intersects_rect


def filter_id(result, input_file):
    # Path of the output new CSV file
    output_file = "filter_xian.csv"

    with open(input_file, "r", newline="") as infile, open(output_file, "w", newline="") as outfile:
        # Create CSV reader and writer
        reader = csv.reader(infile)
        writer = csv.writer(outfile)

        # Read the header of the input file and write it to the output file
        header = next(reader)
        writer.writerow(header)

        # Iterate over each row of the input file
        for row in reader:
            # Assume the trajectory ID is in the first column; modify the index if different in practice
            traj_id = row[0]

            # Write the row to the output file if its trajectory ID is in the target set
            if traj_id in result:
                writer.writerow(row)


import csv
import json
import time


def final_filter_id(csv_file_path, min_lng, max_lng, min_lat, max_lat, start_time, end_time):
    rect = (min_lng, min_lat, max_lng, max_lat)
    result_ids = set()
    start_time_find = time.time()
    with open(csv_file_path, 'r', newline='') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)  # Skip the header row
        for row in reader:
            traj_id = row[0]
            # Use json.loads instead of eval
            traj_data = json.loads(row[1])
            for i in range(len(traj_data)-1):
                _, lng, lat, timestamp = traj_data[i]
                # Both endpoints of the line segment fall within the query range:
                if min_lng <= lng <= max_lng and min_lat <= lat <= max_lat and min_lng <= traj_data[i+1][1] <= max_lng and min_lat <= traj_data[i+1][2] <= max_lat:
                        next_timestamp = traj_data[i+1][3]
                        if timestamp <= end_time and start_time <= next_timestamp:
                            # print(lng,lat,timestamp,traj_data[i+1][1],traj_data[i+1][2],traj_data[i+1][3])
                            result_ids.add(traj_id)
                            break
                # Not completely within the range
                else:
                    traj_start = (lng, lat, timestamp)
                    traj_end = (traj_data[i+1][1], traj_data[i+1][2], traj_data[i+1][3])
                    time_interval = get_time_interval_in_rect(traj_start, traj_end, rect)
                    if time_interval:
                        t_start, t_end = time_interval
                        if t_start <= end_time and start_time <= t_end:
                            # print("2",lng, lat, t_start, traj_data[i + 1][1], traj_data[i + 1][2], t_end)
                            result_ids.add(traj_id)
                            break
    end_time_find = time.time()
    time_all = end_time_find - start_time_find
    # print("Trajectory filtering time:", end_time_find - start_time_find)
    print(len(result_ids), "Final count")
    # print(result_ids)
    return time_all


def line_rect_intersection(line_start, line_end, rect):
    """
    Detect whether a line segment intersects a rectangle and calculate the intersection coordinates and time.

    Parameters:
    line_start: Coordinates and timestamp of the line segment's start point, formatted as (x, y, t)
    line_end: Coordinates and timestamp of the line segment's end point, formatted as (x, y, t)
    rect: Rectangle coordinates, formatted as (x_min, y_min, x_max, y_max)

    Returns:
    If intersecting, returns a list of intersection points (each containing coordinates and time); 
    if not intersecting, returns an empty list
    """
    # Extract the four vertices of the rectangle
    x_min, y_min, x_max, y_max = rect

    # Extract coordinates and timestamps of the line segment's start and end points
    x1, y1, t1 = line_start
    x2, y2, t2 = line_end

    # Four edges of the rectangle
    edges = [
        ((x_min, y_min), (x_max, y_min)),  # Bottom edge
        ((x_max, y_min), (x_max, y_max)),  # Right edge
        ((x_max, y_max), (x_min, y_max)),  # Top edge
        ((x_min, y_max), (x_min, y_min))  # Left edge
    ]

    intersections = []

    # Check if the line segment intersects each edge of the rectangle
    for edge_start, edge_end in edges:
        intersection = line_line_intersection((x1, y1), (x2, y2), edge_start, edge_end)
        if intersection:
            # Calculate the time of the intersection point
            x, y = intersection
            # Calculate parameter t
            if x2 - x1 != 0:
                t = (x - x1) / (x2 - x1)
            else:
                t = (y - y1) / (y2 - y1)
            # Calculate the time
            intersection_time = t1 + t * (t2 - t1)
            intersections.append((x, y, intersection_time))

    # Check if both endpoints of the line segment are inside the rectangle
    if is_point_in_rect((x1, y1), rect):
        intersections.append((x1, y1, t1))
    if is_point_in_rect((x2, y2), rect):
        intersections.append((x2, y2, t2))

    # Deduplicate intersection points
    unique_intersections = []
    for point in intersections:
        if point not in unique_intersections:
            unique_intersections.append(point)

    return unique_intersections


def line_line_intersection(line1_start, line1_end, line2_start, line2_end):
    """
    Calculate the intersection point of two line segments.

    Parameters:
    line1_start, line1_end: Start and end points of the first line segment
    line2_start, line2_end: Start and end points of the second line segment

    Returns:
    If intersecting, returns the intersection coordinates; if not intersecting, returns None
    """
    x1, y1 = line1_start
    x2, y2 = line1_end
    x3, y3 = line2_start
    x4, y4 = line2_end

    # Calculate the denominator
    denominator = (y4 - y3) * (x2 - x1) - (x4 - x3) * (y2 - y1)

    # If the denominator is 0, the two line segments are parallel or collinear
    if denominator == 0:
        return None

    # Calculate parameters t and s
    t = ((x4 - x3) * (y1 - y3) - (y4 - y3) * (x1 - x3)) / denominator
    s = ((x2 - x1) * (y1 - y3) - (y2 - y1) * (x1 - x3)) / denominator

    # If both t and s are within the range [0,1], the two line segments intersect
    if 0 <= t <= 1 and 0 <= s <= 1:
        # Calculate the intersection coordinates
        x = x1 + t * (x2 - x1)
        y = y1 + t * (y2 - y1)
        return (x, y)

    return None


def is_point_in_rect(point, rect):
    """
    Check if a point is inside a rectangle.

    Parameters:
    point: Coordinates of the point, formatted as (x, y)
    rect: Rectangle coordinates, formatted as (x_min, y_min, x_max, y_max)

    Returns:
    True if the point is inside the rectangle; False otherwise
    """
    x, y = point
    x_min, y_min, x_max, y_max = rect
    return x_min <= x <= x_max and y_min <= y <= y_max


def get_time_interval_in_rect(line_start, line_end, rect):
    """
    Get the time interval of a line segment within a rectangle.

    Parameters:
    line_start: Coordinates and timestamp of the line segment's start point, formatted as (x, y, t)
    line_end: Coordinates and timestamp of the line segment's end point, formatted as (x, y, t)
    rect: Rectangle coordinates, formatted as (x_min, y_min, x_max, y_max)

    Returns:
    If the line segment intersects the rectangle, returns the earliest and latest timestamps within the rectangle;
    if not intersecting, returns None
    """
    # Get all intersection points
    intersections = line_rect_intersection(line_start, line_end, rect)

    if not intersections:
        return None

    # Extract timestamps of all intersection points
    timestamps = [point[2] for point in intersections]

    # Sort the timestamps
    timestamps.sort()

    # Check if the start and end points of the line segment are inside the rectangle
    x1, y1, t1 = line_start
    x2, y2, t2 = line_end

    # Determine the part of the line segment inside the rectangle
    if is_point_in_rect((x1, y1), rect) and is_point_in_rect((x2, y2), rect):
        # The line segment is completely inside the rectangle
        return t1, t2
    elif is_point_in_rect((x1, y1), rect):
        # Start point is inside the rectangle, end point is outside
        return t1, timestamps[0]
    elif is_point_in_rect((x2, y2), rect):
        # End point is inside the rectangle, start point is outside
        return timestamps[-1], t2
    else:
        # The line segment is partially inside the rectangle
        return timestamps[0], timestamps[-1]
