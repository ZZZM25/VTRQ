import csv
import json

from vector_cross_product import segment_intersects_rect


def filter_id(result, input_file):
    """
    Filter trajectory records by ID set and save to new CSV file
    
    Args:
        result (set): Target trajectory ID set to retain
        input_file (str): Path to input CSV file containing trajectory data
    """
    # Output path for filtered CSV file
    output_file = "traj_filtered_chengdu.csv"

    with open(input_file, "r", newline="") as infile, open(output_file, "w", newline="") as outfile:
        # Create CSV reader and writer objects
        reader = csv.reader(infile)
        writer = csv.writer(outfile)

        # Read header from input file and write to output file
        header = next(reader)
        writer.writerow(header)

        # Iterate over each row in the input file
        for row in reader:
            # Assume trajectory ID is in the first column (modify index if different)
            traj_id = row[0]

            # Write row to output file if trajectory ID is in target set
            if traj_id in result:
                writer.writerow(row)


import csv
import json
import time


def final_filter_id(csv_file_path, min_lng, max_lng, min_lat, max_lat, start_time, end_time):
    """
    Spatiotemporal filtering of trajectories: retain records intersecting with target geographic rectangle 
    and time range
    
    Args:
        csv_file_path (str): Path to CSV file with trajectory data
        min_lng (float): Minimum longitude of query rectangle
        max_lng (float): Maximum longitude of query rectangle
        min_lat (float): Minimum latitude of query rectangle
        max_lat (float): Maximum latitude of query rectangle
        start_time (int): Start timestamp of query time range (Unix timestamp)
        end_time (int): End timestamp of query time range (Unix timestamp)
    
    Returns:
        float: Total execution time of filtering process (seconds)
    """
    rect = (min_lng, min_lat, max_lng, max_lat)
    result_ids = set()
    start_time_find = time.time()
    
    with open(csv_file_path, 'r', newline='') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)  # Skip header row
        
        for row in reader:
            traj_id = row[0]
            # Use json.loads instead of eval for safe deserialization
            traj_data = json.loads(row[1])
            
            for i in range(len(traj_data)-1):
                _, lng, lat, timestamp = traj_data[i]
                
                # Case 1: Both endpoints of the segment are within query rectangle
                if (min_lng <= lng <= max_lng and 
                    min_lat <= lat <= max_lat and 
                    min_lng <= traj_data[i+1][1] <= max_lng and 
                    min_lat <= traj_data[i+1][2] <= max_lat):
                    
                    next_timestamp = traj_data[i+1][3]
                    if timestamp <= end_time and start_time <= next_timestamp:
                        result_ids.add(traj_id)
                
                # Case 2: Segment partially intersects with query rectangle
                else:
                    traj_start = (lng, lat, timestamp)
                    traj_end = (traj_data[i+1][1], traj_data[i+1][2], traj_data[i+1][3])
                    time_interval = get_time_interval_in_rect(traj_start, traj_end, rect)
                    
                    if time_interval:
                        t_start, t_end = time_interval
                        if t_start <= end_time and start_time <= t_end:
                            result_ids.add(traj_id)
    
    end_time_find = time.time()
    time_all = end_time_find - start_time_find
    print(len(result_ids), "final valid trajectories")
    return time_all


def line_rect_intersection(line_start, line_end, rect):
    """
    Detect intersections between a line segment and a rectangle, calculate intersection coordinates and timestamps
    
    Args:
        line_start (tuple): Start point of line segment with timestamp (x, y, t)
        line_end (tuple): End point of line segment with timestamp (x, y, t)
        rect (tuple): Rectangle boundaries (x_min, y_min, x_max, y_max)
    
    Returns:
        list: List of intersection points (each as (x, y, timestamp)) if intersecting; empty list otherwise
    """
    # Extract rectangle boundaries
    x_min, y_min, x_max, y_max = rect

    # Extract line segment coordinates and timestamps
    x1, y1, t1 = line_start
    x2, y2, t2 = line_end

    # Define four edges of the rectangle
    edges = [
        ((x_min, y_min), (x_max, y_min)),  # Bottom edge
        ((x_max, y_min), (x_max, y_max)),  # Right edge
        ((x_max, y_max), (x_min, y_max)),  # Top edge
        ((x_min, y_max), (x_min, y_min))   # Left edge
    ]

    intersections = []

    # Check intersection with each rectangle edge
    for edge_start, edge_end in edges:
        intersection = line_line_intersection((x1, y1), (x2, y2), edge_start, edge_end)
        if intersection:
            # Calculate timestamp of intersection point
            x, y = intersection
            # Calculate interpolation parameter t
            if x2 - x1 != 0:
                t = (x - x1) / (x2 - x1)
            else:
                t = (y - y1) / (y2 - y1)
            # Compute intersection timestamp
            intersection_time = t1 + t * (t2 - t1)
            intersections.append((x, y, intersection_time))

    # Check if line segment endpoints are inside the rectangle
    if is_point_in_rect((x1, y1), rect):
        intersections.append((x1, y1, t1))
    if is_point_in_rect((x2, y2), rect):
        intersections.append((x2, y2, t2))

    # Remove duplicate intersection points
    unique_intersections = []
    for point in intersections:
        if point not in unique_intersections:
            unique_intersections.append(point)

    return unique_intersections


def line_line_intersection(line1_start, line1_end, line2_start, line2_end):
    """
    Calculate intersection point of two line segments
    
    Args:
        line1_start, line1_end (tuple): Start and end points of first line segment
        line2_start, line2_end (tuple): Start and end points of second line segment
    
    Returns:
        tuple: Intersection coordinates (x, y) if segments intersect; None otherwise
    """
    x1, y1 = line1_start
    x2, y2 = line1_end
    x3, y3 = line2_start
    x4, y4 = line2_end

    # Calculate denominator for line intersection formula
    denominator = (y4 - y3) * (x2 - x1) - (x4 - x3) * (y2 - y1)

    # Segments are parallel or collinear if denominator is zero
    if denominator == 0:
        return None

    # Calculate interpolation parameters t and s
    t = ((x4 - x3) * (y1 - y3) - (y4 - y3) * (x1 - x3)) / denominator
    s = ((x2 - x1) * (y1 - y3) - (y2 - y1) * (x1 - x3)) / denominator

    # Segments intersect if both parameters are within [0, 1]
    if 0 <= t <= 1 and 0 <= s <= 1:
        # Calculate intersection coordinates
        x = x1 + t * (x2 - x1)
        y = y1 + t * (y2 - y1)
        return (x, y)

    return None


def is_point_in_rect(point, rect):
    """
    Check if a point is inside a rectangle (including boundaries)
    
    Args:
        point (tuple): Coordinates of the point (x, y)
        rect (tuple): Rectangle boundaries (x_min, y_min, x_max, y_max)
    
    Returns:
        bool: True if point is inside rectangle; False otherwise
    """
    x, y = point
    x_min, y_min, x_max, y_max = rect
    return x_min <= x <= x_max and y_min <= y <= y_max


def get_time_interval_in_rect(line_start, line_end, rect):
    """
    Get time interval of a line segment that lies within a rectangle
    
    Args:
        line_start (tuple): Start point of line segment with timestamp (x, y, t)
        line_end (tuple): End point of line segment with timestamp (x, y, t)
        rect (tuple): Rectangle boundaries (x_min, y_min, x_max, y_max)
    
    Returns:
        tuple: (earliest_timestamp, latest_timestamp) if segment intersects rectangle; None otherwise
    """
    # Get all intersection points
    intersections = line_rect_intersection(line_start, line_end, rect)

    if not intersections:
        return None

    # Extract timestamps from intersection points
    timestamps = [point[2] for point in intersections]

    # Sort timestamps
    timestamps.sort()

    # Check if segment endpoints are inside rectangle
    x1, y1, t1 = line_start
    x2, y2, t2 = line_end

    # Determine time interval based on segment position relative to rectangle
    if is_point_in_rect((x1, y1), rect) and is_point_in_rect((x2, y2), rect):
        # Entire segment is inside rectangle
        return t1, t2
    elif is_point_in_rect((x1, y1), rect):
        # Start point inside, end point outside
        return t1, timestamps[0]
    elif is_point_in_rect((x2, y2), rect):
        # End point inside, start point outside
        return timestamps[-1], t2
    else:
        # Segment partially overlaps with rectangle
        return timestamps[0], timestamps[-1]
