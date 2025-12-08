import csv
import ast
import hashlib

from hash_collect import first_vo_hash_collect2, first_vo_hash_collect1, edge_vo_hash, second_vo_hash_collect_find, \
    no_edge_vo_hash


def orientation(p, q, r):
    """
    Determine the orientation of three points p, q, r by calculating the vector cross product.
    This function identifies whether the points are collinear, clockwise, or counterclockwise.
    
    Args:
        p (tuple): First point in the format (x, y)
        q (tuple): Second point in the format (x, y)
        r (tuple): Third point in the format (x, y)
    
    Returns:
        int: 0 for collinear, 1 for clockwise, 2 for counterclockwise
    """
    # Calculate vector cross product
    val = (q[1] - p[1]) * (r[0] - q[0]) - (q[0] - p[0]) * (r[1] - q[1])
    if val == 0:
        return 0
    elif val > 0:
        return 1
    return 2


def on_segment(p, q, r):
    """
    Check if point q lies on the line segment pr by verifying coordinate ranges.
    
    Args:
        p (tuple): One endpoint of the segment in the format (x, y)
        q (tuple): Point to be checked in the format (x, y)
        r (tuple): The other endpoint of the segment in the format (x, y)
    
    Returns:
        bool: True if q is on segment pr, False otherwise
    """
    return (min(p[0], r[0]) <= q[0] <= max(p[0], r[0]) and
            min(p[1], r[1]) <= q[1] <= max(p[1], r[1]))


def do_intersect(p1, q1, p2, q2):
    """
    Check if two line segments p1q1 and p2q2 intersect.
    First determine orientation relationships, then handle special collinear/overlapping cases.
    
    Args:
        p1 (tuple): One endpoint of the first segment in the format (x, y)
        q1 (tuple): The other endpoint of the first segment in the format (x, y)
        p2 (tuple): One endpoint of the second segment in the format (x, y)
        q2 (tuple): The other endpoint of the second segment in the format (x, y)
    
    Returns:
        bool: True if segments intersect, False otherwise
    """
    # Calculate four orientation values to determine relative positions
    o1 = orientation(p1, q1, p2)
    o2 = orientation(p1, q1, q2)
    o3 = orientation(p2, q2, p1)
    o4 = orientation(p2, q2, q1)

    # General intersection case: endpoints of each segment are on opposite sides of the other segment
    if o1 != o2 and o3 != o4:
        return True

    # Handle special collinear and overlapping cases
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
    Check if a line segment intersects with a rectangle by testing intersection with all four rectangle edges.
    
    Args:
        segment (tuple): Line segment in the format ((x1, y1), (x2, y2))
        rect (tuple): Rectangle in the format ((x_min, y_min), (x_max, y_max))
    
    Returns:
        bool: True if segment intersects the rectangle, False otherwise
    """
    # Extract bottom-left and top-right coordinates of the rectangle
    x_min, y_min = rect[0]
    x_max, y_max = rect[1]
    # Extract endpoints of the segment
    p1, q1 = segment
    # Define four edges of the rectangle
    rect_edges = [
        ((x_min, y_min), (x_max, y_min)),  # Bottom edge
        ((x_max, y_min), (x_max, y_max)),  # Right edge
        ((x_max, y_max), (x_min, y_max)),  # Top edge
        ((x_min, y_max), (x_min, y_min))   # Left edge
    ]
    # Check intersection with each rectangle edge
    for rect_edge in rect_edges:
        p2, q2 = rect_edge
        if do_intersect(p1, q1, p2, q2):
            return True
    return False


# Check if two longitude/latitude rectangles intersect
def rectangles_intersect(rect1, rect2):
    """
    Check intersection between two geographic rectangles (longitude/latitude bounds).
    
    Args:
        rect1 (list): First rectangle in the format [[min_lng, max_lng], [min_lat, max_lat]] (empty list if no edges)
        rect2 (list): Second rectangle in the format [[min_lng, max_lng], [min_lat, max_lat]]
    
    Returns:
        bool: True if rectangles intersect, False otherwise
    """
    if rect1 != []:  # Skip nodes with no edges (non-leaf nodes)
        min_lat1, max_lat1 = rect1[1]
        min_lng1, max_lng1 = rect1[0]
        min_lat2, max_lat2 = rect2[1]
        min_lng2, max_lng2 = rect2[0]
        # Non-intersection conditions (axis-aligned bounding box test)
        return not (max_lat1 <= min_lat2 or min_lat1 >= max_lat2 or max_lng1 <= min_lng2 or min_lng1 >= max_lng2)
    else:
        return False


def range_query(root, query_lng_range, query_lat_range, query_time_start, query_time_end):
    """
    Perform spatiotemporal range query on RP-Tree with Verification Object (VO) generation for hash verification.
    Collects trajectory IDs matching the query range and writes VO data to CSV for subsequent verification.
    
    Args:
        root (RPTreeNode): Root node of the RP-Tree
        query_lng_range (tuple): Longitude range of the query (min_lng, max_lng)
        query_lat_range (tuple): Latitude range of the query (min_lat, max_lat)
        query_time_start (int): Start timestamp of the query (Unix timestamp)
        query_time_end (int): End timestamp of the query (Unix timestamp)
    
    Returns:
        list: List of trajectory IDs matching the spatiotemporal query range
    """
    # Store matching trajectory IDs
    traj_set = []
    # Track traversed RP-Tree nodes
    rp_path = []
    # Temporary list for node collection
    a = []
    # Query rectangle (longitude/latitude bounds)
    query_rect = [query_lng_range, query_lat_range]
    # Query time range
    query_time = [query_time_start, query_time_end]

    # Open CSV file in append mode to write VO data
    with open('vo.csv', 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)

        def write_to_csv(ver, data=None):
            """
            Helper function to write verification data to CSV.
            
            Args:
                ver (list): Verification data components
                data (list, optional): Supplementary data (e.g., coordinates, timestamps)
            """
            if data is not None:
                writer.writerow([ver, data])
            else:
                writer.writerow([ver])

        def process_edge(e, query_rect, query_time):
            """
            Process edge to check spatiotemporal intersection and generate edge-level VO data.
            
            Args:
                e (Edge): Edge object to process
                query_rect (list): Query rectangle in format [[min_lng, max_lng], [min_lat, max_lat]]
                query_time (list): Query time range in format [start_ts, end_ts]
            """
            # Case 1: At least one endpoint of the edge is inside the query rectangle
            if (query_rect[0][0] <= e.start.lng <= query_rect[0][1] and query_rect[1][0] <= e.start.lat <=
                query_rect[1][1]) or (
                    query_rect[0][0] <= e.end.lng <= query_rect[0][1] and query_rect[1][0] <= e.end.lat <=
                    query_rect[1][1]):
                # Generate VO hash for edge in query range
                ver = edge_vo_hash(e)
                writer.writerow(["e_vo_start"])
                write_to_csv(ver)

                # Process each trajectory on the edge
                for traj in e.traj_hashList:
                    ver, flag = second_vo_hash_collect_find(traj, query_time[0], query_time[1])
                    # Data to verify: trajectory hash (match) or timestamps (no match)
                    data = [traj.traj_hash] if flag == 1 else [str(traj.start_time), str(traj.end_time)]
                    # Add trajectory to result set if it matches time range (avoid duplicates)
                    if flag == 1:
                        if traj.traj_hash not in traj_set:
                            traj_set.append(traj.traj_hash)
                    write_to_csv(ver, data)
                writer.writerow(["e_vo_end"])

            # Case 2: Edge intersects with the query rectangle (but endpoints are outside)
            elif segment_intersects_rect(((e.start.lng, e.start.lat), (e.end.lng, e.end.lat)),
                                         ((query_rect[0][0], query_rect[1][0]), (query_rect[0][1], query_rect[1][1]))):
                ver = edge_vo_hash(e)
                writer.writerow(["e_vo_start"])
                write_to_csv(ver)

                for traj in e.traj_hashList:
                    ver, flag = second_vo_hash_collect_find(traj, query_time[0], query_time[1])
                    data = [traj.traj_hash] if flag == 1 else [str(traj.start_time), str(traj.end_time)]
                    if flag == 1:
                        if traj.traj_hash not in traj_set:
                            traj_set.append(traj.traj_hash)
                    write_to_csv(ver, data)
                writer.writerow(["e_vo_end"])

            # Case 3: Edge is completely outside the query rectangle
            else:
                # Generate VO hash for edge outside query range
                ver = no_edge_vo_hash(e)
                data = [str(e.start.lng), str(e.start.lat), str(e.end.lng), str(e.end.lat)]
                writer.writerow(["e_vo_start"])
                write_to_csv(ver, data)
                writer.writerow(["e_vo_end"])

        def _query(node, query_rect, query_time):
            """
            Recursive core query function to traverse RP-Tree and collect VO data.
            
            Args:
                node (RPTreeNode): Current RP-Tree node to process
                query_rect (list): Query rectangle in format [[min_lng, max_lng], [min_lat, max_lat]]
                query_time (list): Query time range in format [start_ts, end_ts]
            """
            if node is None:
                return
            # Record traversed node
            rp_path.append(node)
            # Get current node's geographic bounds
            node_rect = [node.border_lng, node.border_lat]

            # Case 1: Node bounds do NOT intersect with the query rectangle
            if not rectangles_intersect(node_rect, query_rect):
                # Collect nodes for VO generation (non-intersecting case)
                for i in rp_path:
                    a.append(i)
                ver = first_vo_hash_collect2(a)
                data = [node.border_lng[0], node.border_lng[1], node.border_lat[0], node.border_lat[1]]
                writer.writerow(["lng_lat_vo_start"])
                write_to_csv(ver, [data])
                writer.writerow(["lng_lat_vo_end"])
                rp_path.pop()
                return

            # Case 2: Node bounds intersect with the query rectangle
            # Collect nodes for VO generation (intersecting case)
            for i in rp_path:
                a.append(i)
            ver = first_vo_hash_collect1(a)
            writer.writerow(["lng_lat_vo_start"])
            write_to_csv(ver)

            # Process leaf node (contains adjacent edges)
            if node.is_leaf():
                if node.adjacent_list:
                    for e in node.adjacent_list:
                        process_edge(e, query_rect, query_time)
                    writer.writerow(["lng_lat_vo_end"])
            
            # Process internal node (contains linking edges)
            if not node.is_leaf():
                if node.linking:
                    for e in node.linking:
                        process_edge(e, query_rect, query_time)
                writer.writerow(["lng_lat_vo_end"])
                # Recursively query child nodes
                _query(node.left, query_rect, query_time)
                _query(node.right, query_rect, query_time)
            
            # Remove current node from path (backtrack)
            rp_path.pop()

        # Start recursive query from root node
        _query(root, query_rect, query_time)

    # Return list of matching trajectory IDs
    return traj_set


def proof_vo(filename):
    """
    Verify the integrity of query results by re-computing hashes from VO data in the CSV file.
    Validates both spatial (RP-Tree node/edge) and temporal (trajectory) hash chains.
    
    Args:
        filename (str): Path to the CSV file containing VO data
    
    Returns:
        bool: True if all hash verifications pass, False otherwise
    """
    # Open VO CSV file for reading
    with open(filename, 'r', encoding='utf-8') as file:
        # Temporary storage for verification flags and hash components
        flag1 = []
        flag2 = []
        list1 = []
        list2 = []
        reader = csv.reader(file)

        for row in reader:
            # Process spatial (longitude/latitude) VO start marker
            if row[0] == "lng_lat_vo_start":
                flag1 = next(reader)

                # Case 1: VO length ≤1 means node intersects the query range
                if len(flag1) <= 1:
                    flag1 = ast.literal_eval(flag1[0])
                    flag1.reverse()
                    list1 = []
                # Case 2: VO length >1 means node does NOT intersect the query range (fill coordinates)
                else:
                    c = flag1[0]
                    b = flag1[1]
                    c = ast.literal_eval(c)
                    c.reverse()
                    a = c[-1]
                    # Read coordinate data to fill placeholders
                    b = ast.literal_eval(b)
                    b = b[0]
                    # Insert coordinate data into placeholders
                    j = 0
                    for i in range(len(a)):
                        if a[i] == " ":
                            a[i] = str(b[j])
                            j = j + 1
                            if j > 3:
                                break
                    # Compute SHA-256 hash for verification
                    all_hash_join = '+'.join(item for item in a)
                    encoded_data = all_hash_join.encode('utf-8')
                    hash_object = hashlib.sha256(encoded_data)
                    hash_hex = hash_object.hexdigest()

                    # Process remaining hash components in the chain
                    t = c.pop()
                    while c:
                        t = c.pop()
                        for i in range(len(t)):
                            if t[i] == " ":
                                t[i] = hash_hex
                                break
                        all_hash_join = '+'.join(item for item in t)
                        encoded_data = all_hash_join.encode('utf-8')
                        hash_object = hashlib.sha256(encoded_data)
                        hash_hex = hash_object.hexdigest()
                    row = next(reader)

                    list1 = []
                    # Verify root hash against expected value
                    if hash_hex != "e3f8271c6a401ab67b4991b3919204ec7a7e76637391be89484ef9d11a7f19fe":
                        return False

            # Process edge VO start marker
            elif row[0] == "e_vo_start":
                flag2 = next(reader)

                # Case 1: VO length ≤1 means edge intersects the query range
                if len(flag2) <= 1:
                    flag2 = ast.literal_eval(flag2[0])
                    list2 = []
                # Case 2: VO length >1 means edge does NOT intersect the query range (fill coordinates)
                else:
                    a = flag2[0]
                    b = flag2[1]
                    a = ast.literal_eval(a)
                    b = ast.literal_eval(b)
                    # Insert edge coordinates into placeholders
                    for i in range(4):
                        a[i] = b[i]
                    # Compute edge hash
                    all_hash_join = '+'.join(item for item in a)
                    encoded_data = all_hash_join.encode('utf-8')
                    hash_object = hashlib.sha256(encoded_data)
                    hash_hex = hash_object.hexdigest()
                    list1.append(hash_hex)
                    row = next(reader)

            # Process edge VO end marker
            elif row[0] == "e_vo_end":
                a = flag2
                # Fill trajectory hash placeholders
                j = 0
                for i in range(len(a)):
                    if a[i] == " ":
                        a[i] = list2[j]
                        j = j + 1
                # Compute edge hash with filled placeholders
                all_hash_join = '+'.join(item for item in a)
                encoded_data = all_hash_join.encode('utf-8')
                hash_object = hashlib.sha256(encoded_data)
                hash_hex = hash_object.hexdigest()
                list1.append(hash_hex)

            # Process spatial VO end marker
            elif row[0] == "lng_lat_vo_end":
                # Fill node hash placeholders
                t = flag1.pop()
                j = 0
                for i in range(len(t)):
                    if t[i] == " ":
                        t[i] = list1[j]
                        j = j + 1

                # Compute node hash with filled placeholders
                all_hash_join = '+'.join(item for item in t)
                encoded_data = all_hash_join.encode('utf-8')
                hash_object = hashlib.sha256(encoded_data)
                hash_hex = hash_object.hexdigest()

                # Process remaining nodes in the hash chain
                while flag1:
                    t = flag1.pop()
                    for i in range(len(t)):
                        if t[i] == " ":
                            t[i] = hash_hex
                            break
                    all_hash_join = '+'.join(item for item in t)
                    encoded_data = all_hash_join.encode('utf-8')
                    hash_object = hashlib.sha256(encoded_data)
                    hash_hex = hash_object.hexdigest()
                list1 = []
                # Verify root hash against expected value
                if hash_hex != "e3f8271c6a401ab67b4991b3919204ec7a7e76637391be89484ef9d11a7f19fe":
                    return False
            
            # Process trajectory verification data
            else:
                a = row[0]
                b = row[1]
                a = ast.literal_eval(a)
                b = ast.literal_eval(b)
                # Fill trajectory placeholders with verification data
                j = 0
                for i in range(len(a)):
                    if a[i] == " ":
                        a[i] = b[j]
                        j = j + 1
                # Compute trajectory hash
                str_all = a[0] + a[1] + a[2]
                encoded_data = str_all.encode('utf-8')
                hash_object = hashlib.sha256(encoded_data)
                hash_hex = hash_object.hexdigest()
                list2.append(hash_hex)

    # All verifications passed
    return True


# ------------------------------ Deprecated Original Query Function ------------------------------
# def range_query(root, query_lng_range, query_lat_range, query_time_start, query_time_end):
#     # Range query
#     rp_path = []
#     a = []
#
#     def _query(node, query_rect, query_time):
#         if node is None:
#             return
#         # Record each traversed node
#         rp_path.append(node)
#         node_rect = [node.border_lng, node.border_lat]
#         # Return verification data for non-intersecting regions
#         if not rectangles_intersect(node_rect, query_rect):
#             for i in rp_path:
#                 a.append(i)
#             ver = first_vo_hash_collect2(a)
#             try:
#                 # Open CSV file in append mode
#                 with open(f'vo.csv', 'a', newline='') as csvfile:
#                     writer = csv.writer(csvfile)
#                     # Write data row
#                     writer.writerow([ver, [node.border_lng[0], node.border_lng[1], node.border_lat[0], node.border_lat[1]]])
#                 print(f"collect2 data successfully written to vo.csv")
#             except Exception as e:
#                 print(f"Error writing to file: {e}")
#             return
#
#         # Write first-level index VO for intersecting regions
#         for i in rp_path:
#             a.append(i)
#         ver = first_vo_hash_collect1(a)
#         try:
#             # Open CSV file in append mode
#             with open(f'vo.csv', 'a', newline='') as csvfile:
#                 writer = csv.writer(csvfile)
#                 # Write data row
#                 writer.writerow([ver])
#             print(f"collect1 data successfully written to vo.csv")
#         except Exception as e:
#             print(f"Error writing to file: {e}")
#
#
#         if node.is_leaf():
#                 # Function to determine edges intersecting the query range
#             for e in node.adjacent_list:
#                 # Case 1: At least one endpoint in the query range
#                 if (query_rect[0][0] <= e.start.lng <= query_rect[0][1] and query_rect[1][0] <= e.start.lat <=
#                     query_rect[1][1]) or (
#                         query_rect[0][0] <= e.end.lng <= query_rect[0][1] and query_rect[1][0] <= e.end.lat <=
#                         query_rect[1][1]):
#
#                 # Edge in query range
#                     ver = edge_vo_hash(e)
#                     try:
#                         # Open CSV file in append mode
#                         with open(f'vo.csv', 'a', newline='') as csvfile:
#                             writer = csv.writer(csvfile)
#                             # Write data row
#                             writer.writerow([ver])
#                         print(f"edge data successfully written to vo.csv")
#                     except Exception as e:
#                         print(f"Error writing to file: {e}")
#
#                 # Query trajectories on this edge
#                     for traj in e.traj_hashList:
#                         ver, flag = second_vo_hash_collect_find(traj, query_time[0], query_time[1])
#                         try:
#                             # Open CSV file in append mode
#                             with open(f'vo.csv', 'a', newline='') as csvfile:
#                                 writer = csv.writer(csvfile)
#                                 if flag == 1:
#                                 # Write data row
#                                     writer.writerow([ver, [traj.traj_hash]])
#                                 else:
#                                     writer.writerow([ver, [str(traj.start_time), str(traj.end_time)]])
#                             print(f"traj data successfully written to vo.csv")
#                         except Exception as e:
#                             print(f"Error writing to file: {e}")
#
#                 # Case 2: Check segment intersection if endpoints are outside the query range
#                 elif segment_intersects_rect(((e.start.lng, e.start.lat), (e.end.lng, e.end.lat)), (
#                 (query_rect[0][0], query_rect[1][0]), (query_rect[0][1], query_rect[1][1]))):
#                     # Edge intersects the query range
#                     ver = edge_vo_hash(e)
#                     try:
#                         # Open CSV file in append mode
#                         with open(f'vo.csv', 'a', newline='') as csvfile:
#                             writer = csv.writer(csvfile)
#                             # Write data row
#                             writer.writerow([ver])
#                         print(f"edge data successfully written to vo.csv")
#                     except Exception as e:
#                         print(f"Error writing to file: {e}")
#
#                     # Query trajectories on this edge
#                     for traj in e.traj_hashList:
#                         ver, flag = second_vo_hash_collect_find(traj, query_time[0], query_time[1], flag)
#                         try:
#                             # Open CSV file in append mode
#                             with open(f'vo.csv', 'a', newline='') as csvfile:
#                                 writer = csv.writer(csvfile)
#                                 if flag == 1:
#                                     # Write data row
#                                     writer.writerow([ver, [traj.traj_hash]])
#                                 else:
#                                     writer.writerow([ver, [str(traj.start_time), str(traj.end_time)]])
#                             print(f"traj data successfully written to vo.csv")
#                         except Exception as e:
#                             print(f"Error writing to file: {e}")
#                 else:
#             # Edge outside query range
#                     ver = no_edge_vo_hash(e)
#                     try:
#                         # Open CSV file in append mode
#                         with open(f'vo.csv', 'a', newline='') as csvfile:
#                             writer = csv.writer(csvfile)
#                             # Write data row
#                             writer.writerow([ver,[str(e.start.lng),str(e.start.lat),str(e.end.lat),str(e.emd.lat)]])
#                         print(f"no_edge data successfully written to vo.csv")
#                     except Exception as e:
#                         print(f"Error writing to file: {e}")
#
#
#         if not node.is_leaf():
#             if node.linking:
#
#                 for e in node.linking:
#                     # Case 1: At least one endpoint in the query range
#                     if (query_rect[0][0] <= e.start.lng <= query_rect[0][1] and query_rect[1][0] <= e.start.lat <=
#                         query_rect[1][1]) or (
#                             query_rect[0][0] <= e.end.lng <= query_rect[0][1] and query_rect[1][0] <= e.end.lat <=
#                             query_rect[1][1]):
#                         # Edge in query range
#                         ver = edge_vo_hash(e)
#                         try:
#                             # Open CSV file in append mode
#                             with open(f'vo.csv', 'a', newline='') as csvfile:
#                                 writer = csv.writer(csvfile)
#                                 # Write data row
#                                 writer.writerow([ver])
#                             print(f"edge data successfully written to vo.csv")
#                         except Exception as e:
#                             print(f"Error writing to file: {e}")
#
#                         # Query trajectories on this edge
#                         for traj in e.traj_hashList:
#                             ver, flag = second_vo_hash_collect_find(traj, query_time[0], query_time[1])
#                             try:
#                                 # Open CSV file in append mode
#                                 with open(f'vo.csv', 'a', newline='') as csvfile:
#                                     writer = csv.writer(csvfile)
#                                     if flag == 1:
#                                         # Write data row
#                                         writer.writerow([ver, [traj.traj_hash]])
#                                     else:
#                                         writer.writerow([ver, [str(traj.start_time), str(traj.end_time)]])
#                                 print(f"traj data successfully written to vo.csv")
#                             except Exception as e:
#                                 print(f"Error writing to file: {e}")
#
#                     # Case 2: Check segment intersection if endpoints are outside the query range
#                     elif segment_intersects_rect(((e.start.lng, e.start.lat), (e.end.lng, e.end.lat)),
#                                                  ((query_rect[0][0], query_rect[1][0]),
#                                                   (query_rect[0][1], query_rect[1][1]))):
#                         ver = edge_vo_hash(e)
#                         try:
#                             # Open CSV file in append mode
#                             with open(f'vo.csv', 'a', newline='') as csvfile:
#                                 writer = csv.writer(csvfile)
#                                 # Write data row
#                                 writer.writerow([ver])
#                             print(f"edge data successfully written to vo.csv")
#                         except Exception as e:
#                             print(f"Error writing to file: {e}")
#
#                         # Query trajectories on this edge
#                         for traj in e.traj_hashList:
#                             ver, flag = second_vo_hash_collect_find(traj, query_time[0], query_time[1], flag)
#                             try:
#                                 # Open CSV file in append mode
#                                 with open(f'vo.csv', 'a', newline='') as csvfile:
#                                     writer = csv.writer(csvfile)
#                                     if flag == 1:
#                                         # Write data row
#                                         writer.writerow([ver, [traj.traj_hash]])
#                                     else:
#                                         writer.writerow([ver, [str(traj.start_time), str(traj.end_time)]])
#                                 print(f"traj data successfully written to vo.csv")
#                             except Exception as e:
#                                 print(f"Error writing to file: {e}")
#                     else:
#                         # Edge outside query range
#                         ver = no_edge_vo_hash(e)
#                         try:
#                             # Open CSV file in append mode
#                             with open(f'vo.csv', 'a', newline='') as csvfile:
#                                 writer = csv.writer(csvfile)
#                                 # Write data row
#                                 writer.writerow(
#                                     [ver, [str(e.start.lng), str(e.start.lat), str(e.end.lat), str(e.emd.lat)]])
#                             print(f"no_edge data successfully written to vo.csv")
#                         except Exception as e:
#                             print(f"Error writing to file: {e}")
#
#             _query(node.left, query_rect, query_time)
#             _query(node.right, query_rect, query_time)
#         rp_path.pop()
#
#     # Query geographic range
#     query_rect = [query_lng_range, query_lat_range]
#     query_time = [query_time_start, query_time_end]
#     _query(root, query_rect, query_time)
