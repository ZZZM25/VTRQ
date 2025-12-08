import ast
import csv
import hashlib
import json

from interval_tree import Interval
from new_interval_tree import IntervalTree


# Check if two longitude/latitude rectangles intersect
def rectangles_intersect(rect1, rect2):
    if rect1:  # Some non-leaf nodes may store no edges
        min_lng1, max_lng1 = rect1[0]
        min_lat1, max_lat1 = rect1[1]
        min_lng2, max_lng2 = rect2[0]
        min_lat2, max_lat2 = rect2[1]
        return not (max_lat1 <= min_lat2 or min_lat1 >= max_lat2 or max_lng1 <= min_lng2 or min_lng1 >= max_lng2)

def traj_timestamp_insert(node, traj, V):
    # First calculate the minimum bounding rectangle (MBR) formed by all points of this trajectory
    traj_point_list = []
    for v in V:
        if v.id in traj[0]:
            traj_point_list.append(v)
    # Calculate the minimum latitude of the area covered by the current node
    min_lat = min(v.lat for v in traj_point_list)
    # Calculate the maximum latitude of the area covered by the current node
    max_lat = max(v.lat for v in traj_point_list)
    # Calculate the minimum longitude of the area covered by the current node
    min_lng = min(v.lng for v in traj_point_list)
    # Calculate the maximum longitude of the area covered by the current node
    max_lng = max(v.lng for v in traj_point_list)
    border_lat = [min_lat, max_lat]
    border_lng = [min_lng, max_lng]
    # Minimum bounding rectangle range of this trajectory
    rect_range = [border_lng, border_lat]

    time_insert_query(node, rect_range, traj)




def time_insert_query(node, rect_range, traj):
    if node is None:
        return
    node_rect = (node.border_lng, node.border_lat)
    # No intersection between the trajectory's minimum range and this region
    if not rectangles_intersect(node_rect, rect_range):
        return
    # For leaf nodes with intersections, check if edges exist
    if node.is_leaf():
        # Edges of the trajectory in the edge list
        for e in node.adjacent_list:
            # Extract an edge from the trajectory
            for e1 in traj[1]:
                e1_start = e1[0]
                e1_end = e1[1]
                # If an edge extracted from the trajectory overlaps with an edge in the list, it proves the trajectory passes through this edge
                if (e.start.id == e1_start and e.end.id == e1_end) or (e.start.id == e1_end and e.end.id == e1_start):
                    traj_str = '->'.join(str(item) for item in traj[0])
                    # Find the overall start and end time of the trajectory
                    traj_start_time = traj[2][0][0][0]
                    traj_end_time = traj[2][-1][-1][0]

                    traj_start_time_str = str(traj_start_time)
                    traj_end_time_str = str(traj_end_time)
                    str_time_traj = traj_str + traj_start_time_str + traj_end_time_str
                    encoded_data = str_time_traj.encode('utf-8')
                    hash_object = hashlib.sha256(encoded_data)
                    hash_hex = hash_object.hexdigest()
                    e.traj_hashList.append(hash_hex)
                    break

    if not node.is_leaf():
        if rectangles_intersect(node.linking_box, rect_range):
        # For branch nodes, consider linking edges
            for e in node.linking:
                flag = 0
                # Extract an edge from the trajectory
                for e1 in traj[1]:
                    e1_start = e1[0]
                    e1_end = e1[1]

                    # If an edge extracted from the trajectory overlaps with an edge in the list, it proves the trajectory passes through this edge
                    if (e.start.id == e1_start and e.end.id == e1_end) or (e.start.id == e1_end and e.end.id == e1_start):
                        # Write logic here to insert this trajectory into the edge's list - first create a trajectory object, insert in start time order
                        traj_str = '->'.join(str(item) for item in traj[0])
                        # Find the overall start and end time of the trajectory
                        traj_start_time = traj[2][0][0][0]
                        traj_end_time = traj[2][-1][-1][0]

                        traj_start_time_str = str(traj_start_time)
                        traj_end_time_str = str(traj_end_time)
                        str_time_traj = traj_str + traj_start_time_str + traj_end_time_str
                        encoded_data = str_time_traj.encode('utf-8')
                        hash_object = hashlib.sha256(encoded_data)
                        hash_hex = hash_object.hexdigest()
                        e.traj_hashList.append(hash_hex)
                        break

# Previously, the trajectory hash only included the path and start/end times
        time_insert_query(node.left, rect_range, traj)
        time_insert_query(node.right, rect_range, traj)




# Insert trajectory information into the interval tree
def insert_traj_into_interval_tree(interval_tree, traj):
    traj_start_time = traj[2][0][0][0]
    traj_end_time = traj[2][-1][-1][0]
    traj_str = '->'.join(str(item) for item in traj[0])
    traj_start_time_str = str(traj_start_time)
    traj_end_time_str = str(traj_end_time)
    str_time_traj = traj_str + traj_start_time_str + traj_end_time_str
    encoded_data = str_time_traj.encode('utf-8')
    hash_object = hashlib.sha256(encoded_data)
    hash_hex = hash_object.hexdigest()
    interval_tree.interval_t_insert(Interval(traj_start_time, traj_end_time), hash_hex)





def insert_time_stamp(node, V):
    for i in range(1, 4):
        traj_name_path =  f"C:\\Users\\maoyusen\\Desktop\\Graph-Diffusion-Planning-main\\chengdu-tra-json\\traj-10-{i}.json"
        # Open JSON file
       # traj_name_path = f"C:\\Users\\maoyusen\\Desktop\\Graph-Diffusion-Planning-main\\loader\\preprocess\\mm\\sets_data\\real2\\trajectories\\traj_mapped_xian_xian10-{i}.json"
        with open(traj_name_path, 'r', encoding='utf-8') as file:
            # Parse JSON data
            nested_list = json.load(file)
        j = 0
        for traj in nested_list:
            j = j + 1
            traj_timestamp_insert(node, traj, V)
            print(f"Completed processing {j}th trajectory of traj-10-{i}")
    # for i in range(1, 31):
    #     traj_name_path = f"C:\\Users\\maoyusen\\Desktop\\Graph-Diffusion-Planning-main\\chengdu-tra-json\\traj-11-{i}.json"
    #     # Open JSON file
    #
    #     with open(traj_name_path, 'r', encoding='utf-8') as file:
    #         # Parse JSON data
    #         nested_list = json.load(file)
    #     j = 0
    #     for traj in nested_list:
    #         j = j + 1
    #         traj_timestamp_insert(node, traj, V)
    #         print(f"Completed processing {j}th trajectory of traj-11-{i}")


def insert_time_stamp2(T):
    j = 0
    for i in range(1, 4):
        #traj_name_path = f"C:\\Users\\maoyusen\\Desktop\\Graph-Diffusion-Planning-main\\loader\\preprocess\\mm\\sets_data\\real2\\trajectories\\traj_mapped_xian_xian10-{i}.json"
        traj_name_path = f"C:\\Users\\maoyusen\\Desktop\\Graph-Diffusion-Planning-main\\chengdu-tra-json\\traj-10-{i}.json"
        # Open JSON file
        with open(traj_name_path, 'r', encoding='utf-8') as file:
            # Parse JSON data
            nested_list = json.load(file)

        for traj in nested_list:
            j = j + 1
            insert_traj_into_interval_tree(T, traj)
            print(f"Completed processing {j}th trajectory of traj-10-{i}")

    # for i in range(1,31):
    #     traj_name_path = f"C:\\Users\\maoyusen\\Desktop\\Graph-Diffusion-Planning-main\\chengdu-tra-json\\traj-11-{i}.json"
    #     # Open JSON file
    #     with open(traj_name_path, 'r', encoding='utf-8') as file:
    #         # Parse JSON data
    #         nested_list = json.load(file)
    #
    #     for traj in nested_list:
    #         j = j + 1
    #         insert_traj_into_interval_tree(T, traj)
    #
    #         print(f"Completed processing {j}th trajectory of traj-11-{i}")
    # print(j)







# update

def insert_time_stamp1(node, V):
    j = 0
    for i in range(16, 23):
        traj_name_path =f"C:\\Users\\maoyusen\\Desktop\\Graph-Diffusion-Planning-main\\loader\\preprocess\\mm\\sets_data\\real2\\trajectories\\traj_mapped_xian_xian10-{i}.json"
        # Open JSON file

        with open(traj_name_path, 'r', encoding='utf-8') as file:
            # Parse JSON data
            nested_list = json.load(file)

        for traj in nested_list:
            if j == 5000:
                return
            j = j + 1
            traj_timestamp_insert(node, traj, V)
            print(f"Completed processing {j}th trajectory of traj-10-{i}")


def insert_time_stamp21(T):
    j = 0
    for i in range(16, 23):
        traj_name_path = f"C:\\Users\\maoyusen\\Desktop\\Graph-Diffusion-Planning-main\\loader\\preprocess\\mm\\sets_data\\real2\\trajectories\\traj_mapped_xian_xian10-{i}.json"
        # Open JSON file
        with open(traj_name_path, 'r', encoding='utf-8') as file:
            # Parse JSON data
            nested_list = json.load(file)

        for traj in nested_list:
            if j == 5000:
                return
            j = j + 1
            insert_traj_into_interval_tree(T, traj)
            print(f"Completed processing {j}th trajectory of traj-10-{i}")

from typing import List, Tuple


# V is a dictionary

#[[323080529, 116.673939, 39.873818, 1202090710], [606787516, 116.669716, 39.87734, 1202091309], [606787637, 116.670906, 39.878194, 1202091909], [2504904016, 116.671331, 39.878438, 1202092509], [606787632, 116.671806, 39.878659, 1202093109], [1205600825, 116.673354, 39.879236, 1202093709], [2504904019, 116.673287, 39.87933, 1202094309]]
def traj_timestamp_insert_beijing(node, traj, V):
    # Extract the list of IDs from the trajectory (trajectory_points stores ID numbers)
    trajectory_points = [point[0] for point in traj[1]]  # Simplify list comprehension

    # Collect node objects corresponding to trajectory IDs from dictionary V (in trajectory ID order)
    traj_point_list = []
    for point_id in trajectory_points:
        # Directly look up from dictionary V by ID, time complexity O(1)
        if point_id in V:
            traj_point_list.append(V[point_id])
        else:
            # Optional: handle cases where the ID is not in V (e.g., print a warning)
            print(f"Warning: ID {point_id} in trajectory not found in V")

    # Fault tolerance: avoid min/max errors caused by empty lists
    if not traj_point_list:
        print("No valid nodes found in the trajectory, skip processing")
        return

    # Calculate boundaries (logic unchanged)
    min_lat = min(v.lat for v in traj_point_list)
    max_lat = max(v.lat for v in traj_point_list)
    min_lng = min(v.lng for v in traj_point_list)
    max_lng = max(v.lng for v in traj_point_list)

    border_lat = [min_lat, max_lat]
    border_lng = [min_lng, max_lng]
    rect_range = [border_lng, border_lat]  # Minimum bounding rectangle range of the trajectory

    time_insert_query_beijing(node, rect_range, traj)


def time_insert_query_beijing(node, rect_range, traj):
    if node is None:
        return
    node_rect = (node.border_lng, node.border_lat)
    # No intersection between the trajectory's minimum range and this region
    if not rectangles_intersect(node_rect, rect_range):
        return
    # For leaf nodes with intersections, check if edges exist
    if node.is_leaf():
        # Edges of the trajectory in the edge list
        for e in node.adjacent_list:
            # Extract an edge from the trajectory
            for i in range(len(traj[1]) - 1):
                e1_start = traj[1][i][0]
                e1_end = traj[1][i + 1][0]
                # If an edge extracted from the trajectory overlaps with an edge in the list, it proves the trajectory passes through this edge
                if (e.start.id == e1_start and e.end.id == e1_end) or (e.start.id == e1_end and e.end.id == e1_start):
                    traj_start_time = traj[1][0][-1]
                    traj_end_time = traj[1][-1][-1]
                    traj_start_time_str = str(traj_start_time)
                    traj_end_time_str = str(traj_end_time)
                    str_time_traj = traj[0] + traj_start_time_str + traj_end_time_str
                    encoded_data = str_time_traj.encode('utf-8')
                    hash_object = hashlib.sha256(encoded_data)
                    hash_hex = hash_object.hexdigest()
                    e.traj_hashList.append(hash_hex)
                    #print("Found", e.start.id, e.end.id)
                    break

    if not node.is_leaf():
        if rectangles_intersect(node.linking_box, rect_range):
        # For branch nodes, consider linking edges
            for e in node.linking:
                for i in range(len(traj[1]) - 1):
                    e1_start = traj[1][i][0]
                    e1_end = traj[1][i + 1][0]
                    # If an edge extracted from the trajectory overlaps with an edge in the list, it proves the trajectory passes through this edge
                    if (e.start.id == e1_start and e.end.id == e1_end) or (e.start.id == e1_end and e.end.id == e1_start):
                        # Write logic here to insert this trajectory into the edge's list - first create a trajectory object, insert in start time order
                        # Find the overall start and end time of the trajectory
                        traj_start_time = traj[1][0][-1]
                        traj_end_time = traj[1][-1][-1]

                        traj_start_time_str = str(traj_start_time)
                        traj_end_time_str = str(traj_end_time)
                        str_time_traj = traj[0] + traj_start_time_str + traj_end_time_str
                        encoded_data = str_time_traj.encode('utf-8')
                        hash_object = hashlib.sha256(encoded_data)
                        hash_hex = hash_object.hexdigest()
                        e.traj_hashList.append(hash_hex)
                        #print("Found",e.start.id,e.end.id)
                        break

        time_insert_query_beijing(node.left, rect_range, traj)
        time_insert_query_beijing(node.right, rect_range, traj)


def insert_beijing_to_vtrq(file_path, node, V):
    with open(file_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        next(reader)  # Skip header row
        i = 0
        for row in reader:
            traj = []  # Initialize traj at the start of each loop to ensure it is empty
            i += 1
            if len(row) < 2:
                print(f"Row {i}: Incomplete data, skip")
                continue

            # Extract traj_id
            traj_id = row[0].strip()

            # Parse points_data
            try:
                points_data = ast.literal_eval(row[1].strip())
                traj.append(traj_id)  # traj[0] = trajectory ID
                traj.append(points_data)  # traj[1] = point data list
            except (SyntaxError, ValueError) as e:
                print(f"Parsing error in row {i} (traj_id: {traj_id}): {e}")
                continue  # traj remains empty at this point, no additional reset needed
            if 14000 < i <= 14500:
            # Process current trajectory
                traj_timestamp_insert_beijing(node, traj, V)
                print(f"Completed processing row {i}")  # Fix print format





# Insert trajectory information into the interval tree
def insert_traj_into_interval_tree_beijing(interval_tree, traj):
    traj_start_time = traj[1][0][-1]
    traj_end_time = traj[1][-1][-1]
    traj_start_time_str = str(traj_start_time)
    traj_end_time_str = str(traj_end_time)
    str_time_traj = traj[0] + traj_start_time_str + traj_end_time_str
    encoded_data = str_time_traj.encode('utf-8')
    hash_object = hashlib.sha256(encoded_data)
    hash_hex = hash_object.hexdigest()
    interval_tree.interval_t_insert(Interval(traj_start_time, traj_end_time), hash_hex)

def insert_beijing_to_interval_beijing(file_path, T):
    with open(file_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        next(reader)  # Skip header row
        i = 0
        for row in reader:
            traj = []  # Initialize traj at the start of each loop to ensure it is empty
            i += 1
            if len(row) < 2:
                print(f"Row {i}: Incomplete data, skip")
                continue

            # Extract traj_id
            traj_id = row[0].strip()

            # Parse points_data
            try:
                points_data = ast.literal_eval(row[1].strip())
                traj.append(traj_id)  # traj[0] = trajectory ID
                traj.append(points_data)  # traj[1] = point data list
            except (SyntaxError, ValueError) as e:
                print(f"Parsing error in row {i} (traj_id: {traj_id}): {e}")
                continue  # traj remains empty at this point, no additional reset needed
            if 14000 < i <= 14500:
                insert_traj_into_interval_tree_beijing(T, traj)
                print(f"Completed processing row {i}")  # Fix print format
