import csv
import hashlib
import json
import time

import gmpy2

from ACC import initialize_accumulator, add_string_element
from filter_traj_id import filter_id, final_filter_id
from range_query import range_query, proof_vo
from traj_timestamp_insert import insert_time_stamp, insert_time_stamp2, insert_time_stamp1, insert_time_stamp21
from new_interval_tree import IntervalTree, Interval, proof_interval
from traj_id_to_raw_data import idtraj

def rptree_to_dict(node):
    if node is None:
        return None
    return {
        "border_lng": node.border_lng,
        "border_lat": node.border_lat,
        "linking_box": node.linking_box,
        "linking": [edge_to_dict(edge) for edge in node.linking] if node.linking else None,
        "left": rptree_to_dict(node.left),
        "right": rptree_to_dict(node.right),
        "adjacent_list": [edge_to_dict(edge) for edge in node.adjacent_list] if node.adjacent_list else None,
        "rp_hash_merge":node.rp_hash_merge
    }


def edge_to_dict(edge):
    return {
        "id": edge.id,
        "start": vertex_to_dict(edge.start),
        "end": vertex_to_dict(edge.end),
        "traj_hashList": edge.traj_hashList,
        "traj_hashList_merge": edge.traj_hashList_merge,
        "edge_merge": edge.edge_merge,
        "weight":edge.weight
    }


def vertex_to_dict(vertex):
    return {
        'id': vertex.id,
        'lat': vertex.lat,
        'lng': vertex.lng
    }

def save_rp_tree_json(root, filename):
    data = rptree_to_dict(root)
    with open(filename, 'w') as f:
        json.dump(data, f)

# Define vertex class to represent vertices in graphs
class Vertex:
    def __init__(self, id, lat, lng):
        # 顶点的唯一标识符
        self.id = id
        self.lat = lat
        self.lng = lng
   

# Define edge class to represent connections between two vertices in a graph
class Edge:
    def __init__(self, id, start, end, weight):
        self.id = id
        self.start = start
        self.end = end
        self.edge_merge = ""
        self.traj_hashList = []
        self.traj_hashList_merge = ""
        self.weight = weight



    def __repr__(self):
        return f"Edge(id={self.id}, start={self.start}, end={self.end}), min_timestamp={self.min_timestamp},max_timestamp={self.max_timestamp}"


# Define RP-Tree node class to build recursive partitioning trees
class RPTreeNode:
    def __init__(self, border_lat, border_lng, linking=None, left=None, right=None, linking_box=None,
                 adjacent_list=None):     
        self.border_lat = border_lat
        self.border_lng = border_lng
        self.linking_box = linking_box if linking_box else []
        self.linking = linking if linking else []
        self.left = left
        self.right = right
        self.adjacent_list = adjacent_list if adjacent_list else []
        self.rp_hash_merge = ""

    def is_leaf(self):
        return self.left is None and self.right is None



def find_best_split(sorted_V, E, d):
    n = len(sorted_V)
    mid = n // 2
    vertex_edges = {v: [] for v in sorted_V}
    for e in E:
        vertex_edges[e.start].append(e)
        vertex_edges[e.end].append(e)


    def calculate_edges_weight(left_vertices, right_vertices):
        left_vertices_set = set(left_vertices)
        right_vertices_set = set(right_vertices)
        left_weight = 0
        right_weight = 0
        for v in left_vertices:
            for e in vertex_edges[v]:
                if e.start in left_vertices_set and e.end in left_vertices_set:
                    left_weight += e.weight
        for v in right_vertices:
            for e in vertex_edges[v]:
                if e.start in right_vertices_set and e.end in right_vertices_set:
                    right_weight += e.weight
        return left_weight, right_weight

    left_vertices_mid = sorted_V[:mid + 1]
    right_vertices_mid = sorted_V[mid + 1:]
    left_edges_weight_mid, right_edges_weight_mid = calculate_edges_weight(left_vertices_mid, right_vertices_mid)
    min_diff = abs(left_edges_weight_mid - right_edges_weight_mid)
    best_p = mid

   
    left_diff = min_diff
    left_p = mid
    for p in range(mid - 1, -1, -1):
   
        left_vertices = sorted_V[:p + 1]
        right_vertices = sorted_V[p + 1:]
        left_edges_weight, right_edges_weight = calculate_edges_weight(left_vertices, right_vertices)   
        diff = abs(left_edges_weight - right_edges_weight)
        if diff < left_diff:
            left_diff = diff
            left_p = p
        else:
            break

    right_diff = min_diff
    right_p = mid
    # 从中间分割点向右遍历
    for p in range(mid + 1, n):
        left_vertices = sorted_V[:p + 1]
        right_vertices = sorted_V[p + 1:]
        left_edges_weight, right_edges_weight = calculate_edges_weight(left_vertices, right_vertices)
        diff = abs(left_edges_weight - right_edges_weight)
        if diff < right_diff:
            right_diff = diff
            right_p = p
        else:
            break
    if left_diff < min_diff:
        min_diff = left_diff
        best_p = left_p
    if right_diff < min_diff:
        min_diff = right_diff
        best_p = right_p

    left_vertices_best = sorted_V[:best_p + 1]
    right_vertices_best = sorted_V[best_p + 1:]
    left_edges_weight_best, right_edges_weight_best = calculate_edges_weight(left_vertices_best, right_vertices_best)
    print(f"At level {d}, best split point p = {best_p}, left edges weight: {left_edges_weight_best}, right edges weight: {right_edges_weight_best}")
    return best_p



# Function to recursively split the graph for building an RP-Tree
def split(V, E, h, uH, d):
    # Calculate the minimum latitude of the area covered by the current node
    min_lat = min(v.lat for v in V)
    # Calculate the maximum latitude of the area covered by the current node
    max_lat = max(v.lat for v in V)
    # Calculate the minimum longitude of the area covered by the current node
    min_lng = min(v.lng for v in V)
    # Calculate the maximum longitude of the area covered by the current node
    max_lng = max(v.lng for v in V)
    # Create a new RP-Tree node
    node = RPTreeNode((min_lat, max_lat), (min_lng, max_lng))

    # Print the boundary information of the current node
    print(f"Level {d}: Node border - Lat: ({min_lat}, {max_lat}), Lng: ({min_lng}, {max_lng})")

    # Termination condition 1: Reach the preset maximum index level threshold
    if d >= uH:
        # Filter out edges that are completely within the current area
        node.adjacent_list = [edge for edge in E if min_lat <= edge.start.lat <= max_lat and
                              min_lng <= edge.start.lng <= max_lng and
                              min_lat <= edge.end.lat <= max_lat and
                              min_lng <= edge.end.lng <= max_lng]
        print(f"Reached max level {uH}, making node at level {d} a leaf node with {len(node.adjacent_list)} edges.")
        node.left = None
        node.right = None
        return node

    # Termination condition 2: The number of edges associated with the current node is less than the preset threshold h
    if len(E) < h:
        # Filter out edges that are completely within the current area
        node.adjacent_list = [edge for edge in E if min_lat <= edge.start.lat <= max_lat and
                              min_lng <= edge.start.lng <= max_lng and
                              min_lat <= edge.end.lat <= max_lat and
                              min_lng <= edge.end.lng <= max_lng]
        print(f"Edge count {len(node.adjacent_list)} less than threshold {h}, making node at level {d} a leaf node.")
        node.left = None
        node.right = None
        return node

    # Select the split direction based on the size of latitude and longitude ranges
    lat_range = max_lat - min_lat
    lng_range = max_lng - min_lng
    split_by_lat = lat_range > lng_range
    # Sort the vertex list according to the split direction
    sorted_V = sorted(V, key=lambda v: v.lat) if split_by_lat else sorted(V, key=lambda v: v.lng)
    print(f"At level {d}, splitting by {'latitude' if split_by_lat else 'longitude'}.")


    # Call the optimized function to find the optimal split point
    best_p = find_best_split(sorted_V, E, d)

    # Split the vertex set based on the optimal split point
    V_left = sorted_V[:best_p + 1]
    V_right = sorted_V[best_p + 1:]

    # Split the edge set based on the split vertex sets
    E_left = [e for e in E if (e.start in V_left and e.end in V_left)]
    E_right = [e for e in E if (e.start in V_right and e.end in V_right)]

    # Recursively build the left subtree
    print(f"Recursively building left subtree at level {d + 1}...")
    node.left = split(V_left, E_left, h, uH, d + 1)
    # Recursively build the right subtree
    print(f"Recursively building right subtree at level {d + 1}...")
    node.right = split(V_right, E_right, h, uH, d + 1)

    # Find edges spanning the left and right subtrees (i.e., linking edges)
    linking_edges = [e for e in E if ((e.start in V_left and e.end in V_right) or (e.start in V_right and e.end in V_left))]
    # Set the list of linking edges to the 'linking' attribute of the current node
    node.linking = linking_edges

    if linking_edges:
        min_link_lat = min(min(e.start.lat, e.end.lat) for e in linking_edges)
        max_link_lat = max(max(e.start.lat, e.end.lat) for e in linking_edges)
        min_link_lng = min(min(e.start.lng, e.end.lng) for e in linking_edges)
        max_link_lng = max(max(e.start.lng, e.end.lng) for e in linking_edges)
        node.linking_box = [[min_link_lng, max_link_lng], [min_link_lat, max_link_lat]]

    return node


def load_graph(node_file, edge_file):

    V = []
  
    E = []
   
    vertex_dict = {}

  
    with open(node_file, 'r') as f:
        for line in f:
          
            parts = line.strip().split()
          
            id = int(parts[0])
          
            lat = float(parts[2])
           
            lng = float(parts[1])
          
            vertex = Vertex(id, lat, lng)
           
            V.append(vertex)
           
            vertex_dict[id] = vertex

  
    with open(edge_file, 'r') as f:
        for line in f:
          
            parts = line.strip().split()
          
            id = int(parts[0])
           
            start_id = int(parts[1])
            
            end_id = int(parts[2])

            weight = float(parts[3])
           
            if start_id not in vertex_dict:
                print(f"错误: 顶点 {start_id} 不存在",id)
            if end_id not in vertex_dict:
                print(f"错误: 顶点 {end_id} 不存在",id)
            start_vertex = vertex_dict[start_id]
           
            end_vertex = vertex_dict[end_id]
         
            edge = Edge(id, start_vertex, end_vertex, weight)
          
            E.append(edge)

    return V, E


# 从字典构建 RPTreeNode
def dict_to_rptree(data):
    if data is None:
        return None
    if data["adjacent_list"] is not None:
        adjacent_list = [dict_to_edge(edge_data) for edge_data in data["adjacent_list"]]
    else:
        adjacent_list = []
    if data["linking"] is not None:
        linking = [dict_to_edge(edge_data) for edge_data in data["linking"]]
    else:
        linking = []
    node = RPTreeNode(data["border_lat"], data["border_lng"], linking_box=data["linking_box"])
    node.adjacent_list = adjacent_list
    node.linking = linking
    node.rp_hash_merge = data["rp_hash_merge"]
    node.left = dict_to_rptree(data["left"])
    node.right = dict_to_rptree(data["right"])
    return node

def dict_to_vertex(vertex_data):
    return Vertex(vertex_data['id'], vertex_data['lat'], vertex_data['lng'])

def dict_to_edge(edge_data):
    start = dict_to_vertex(edge_data['start'])
    end = dict_to_vertex(edge_data['end'])
    edge = Edge(edge_data["id"], start, end, edge_data["weight"])
    edge.edge_merge = edge_data["edge_merge"]
    edge.traj_hashList = edge_data["traj_hashList"]
    edge.traj_hashList_merge = edge_data["traj_hashList_merge"]
    return edge


def load_rp_tree_json(filename):
        with open(filename, 'r') as f:
            data = json.load(f)
        return dict_to_rptree(data)



def second_hash_merge(edge):
    hash_list = []
    str1 = str(edge.start.lng)
    str2 = str(edge.start.lat)
    str3 = str(edge.end.lng)
    str4 = str(edge.end.lat)
    hash_list.append(str1)
    hash_list.append(str2)
    hash_list.append(str3)
    hash_list.append(str4)


    if edge.traj_hashList:
        traj_list = []
        for traj in edge.traj_hashList:
            traj_list.append(traj)
        all_traj_str = '+'.join(item for item in traj_list)
        encoded_data = all_traj_str.encode('utf-8')
        hash_object = hashlib.sha256(encoded_data)
        hash_hex = hash_object.hexdigest()
        edge.traj_hashList_merge = hash_hex
        hash_list.append(edge.traj_hashList_merge)


    all_hash_join = '+'.join(item for item in hash_list)
    encoded_data = all_hash_join.encode('utf-8')
    hash_object = hashlib.sha256(encoded_data)
    hash_hex = hash_object.hexdigest()
    edge.edge_merge = hash_hex

def first_hash_merge(node):
        if node is None:
            return
 
        first_hash_merge(node.left)
 
        first_hash_merge(node.right)

        # print(f"Node with lat bounds: {node.border_lat}, lng bounds: {node.border_lng}")

        if node.is_leaf():
            if node.adjacent_list:
               
                for edge in node.adjacent_list:
                    second_hash_merge(edge)

                str_list = []
                str_border_lng0 = str(node.border_lng[0])
                str_list.append(str_border_lng0)
                str_border_lng1 = str(node.border_lng[1])
                str_list.append(str_border_lng1)
                str_border_lat0 = str(node.border_lat[0])
                str_list.append(str_border_lat0)
                str_border_lat1 = str(node.border_lat[1])
                str_list.append(str_border_lat1)


                for edge in node.adjacent_list:
                    str_list.append(edge.edge_merge)

                all_hash_join = '+'.join(item for item in str_list)
                encoded_data = all_hash_join.encode('utf-8')
                hash_object = hashlib.sha256(encoded_data)
                hash_hex = hash_object.hexdigest()
                node.rp_hash_merge = hash_hex
   
        else:
            str_list = []
            str_border_lng0 = str(node.border_lng[0])
            str_list.append(str_border_lng0)
            str_border_lng1 = str(node.border_lng[1])
            str_list.append(str_border_lng1)
            str_border_lat0 = str(node.border_lat[0])
            str_list.append(str_border_lat0)
            str_border_lat1 = str(node.border_lat[1])
            str_list.append(str_border_lat1)

            if node.linking_box:
                str_list.append(str(node.linking_box[0][0]))
                str_list.append(str(node.linking_box[0][1]))
                str_list.append(str(node.linking_box[1][0]))
                str_list.append(str(node.linking_box[1][1]))
     

            if node.linking:
                for edge in node.linking:
                    second_hash_merge(edge)
                for edge in node.linking:
                    str_list.append(edge.edge_merge)
            if node.left:
                str_list.append(node.left.rp_hash_merge)
            if node.right:
                str_list.append(node.right.rp_hash_merge)
            all_hash_join = '+'.join(item for item in str_list)
            encoded_data = all_hash_join.encode('utf-8')
            hash_object = hashlib.sha256(encoded_data)
            hash_hex = hash_object.hexdigest()
            node.rp_hash_merge = hash_hex






if __name__ == "__main__":
# Build index
    # start_time = time.time()
    # # File name of the vertex data file
    # node_file = "chengdu_nodes.txt"
    # # File name of the edge data file
    # edge_file = "chengdu_edges_lianggeyue.txt"
    # # Call the load_graph function to load vertex and edge data
    # V, E = load_graph(node_file, edge_file)
    # # # idtraj(V)
    # # # Call the split function to build the RP-Tree, set the minimum edge count threshold to 10, max level to 10, current level to 0
    # root = split(V, E, h=8, uH=9, d=0)
    #
    # insert_time_stamp(root, V)
    # first_hash_merge(root)
    # save_rp_tree_json(root, "chengdu_rp.json")
    # T = IntervalTree()
    # insert_time_stamp2(T)
    # T.trag_hash_merge(T.root)
    # end_time = time.time()
    # print(end_time-start_time)
    # T.save_to_json("xian_interval_3_days.json")
    load_rp_tree = load_rp_tree_json("chengdu_rp.json")
    # Query the spatial tree
    rp_set = range_query(load_rp_tree, (115.4257844, 116.5802607), (40.4473009, 42.147308))
    load_interval_tree = IntervalTree.load_from_json("beijing_interval.json")
    # Query the temporal tree
    interval_set = load_interval_tree.search_intersecting_intervals(Interval(1202360244, 1202363559))
    result = interval_set & rp_set
    #print(interval_set)
    # print(rp_set)
    # print(result)
    # Aggregation
    filter_id(result, "beijing_trajectories_really.csv")
    time_all = final_filter_id("filter_beijing.csv", 115.4257844, 116.5802607, 40.4473009, 42.147308, 1202360244, 1202363559)

    #traverse_all_edges(root)
    #for i in network_id:
    #    print(i.min_timestamp, i.max_timestamp)
    # end_time = time.time()
    # time_insert_rp = end_time - start_time
    #
    # start_time = time.time()
    # first_hash_merge(root)
    # end_time = time.time()
    # time_rp_merge = end_time - start_time
    #
    # T = IntervalTree()
    # start_time = time.time()
    # insert_time_stamp2(T)
    # end_time = time.time()
    # time_interval_insert = end_time - start_time
    # #
    # start_time = time.time()
    # T.trag_hash_merge(T.root)
    # end_time = time.time()
    # time_interval_merge = end_time - start_time
    #
    #
    #
    # print(time_rp_construct)
    # print(time_insert_rp)
    # print(time_rp_merge)
    # print(time_interval_insert)
    # print(time_interval_merge)

    # save_rp_tree_json(root, "chengdu_rpYIZHOU.json")
    # T.save_to_json("chengdu_intervalYIZHOU.json")

    #
    # load_rp_tree = load_rp_tree_json("chengdu_rp.json")
    # load_interval_tree = IntervalTree.load_from_json("chengdu_interval.json")
    # interval_set = load_interval_tree.search_intersecting_intervals(Interval(1539689000, 1539696200))
    # print(len(interval_set))
    # rp_set = range_query(load_rp_tree, (104.01, 104.07), (30.67, 30.69))
    # # [104.0, 104.14958], "border_lat": [30.64, 30.73]
    # print(len(rp_set))
    # result = interval_set & rp_set
    # print(len(result))
    # filter_id(result, "trajectories_chengdu.csv")
    # final_filter_id("traj_filtered_chengdu.csv",104.01, 104.07,  30.67, 30.69,1539689000, 1539696200)

    # start_time = time.time()
    # insert_time_stamp1(load_rp_tree,V)
    # end_time = time.time()
    # time_rp_insert = end_time - start_time
    #
    # start_time = time.time()
    # first_hash_merge(load_rp_tree)
    # end_time = time.time()
    # time_rp_merge = end_time - start_time
    #
    # start_time = time.time()
    # insert_time_stamp21(load_interval_tree)
    # end_time = time.time()
    # time_interval_insert = end_time - start_time
    #
    # start_time = time.time()
    # load_interval_tree.trag_hash_merge(load_interval_tree.root)
    # end_time = time.time()
    # time_interval_merge = end_time - start_time
    # print(time_rp_insert)
    # print(time_rp_merge)
    # print(time_interval_insert)
    # print(time_interval_merge)





    # insert_time_stamp(load_rp_tree, V)
    # insert_time_stamp2(load_interval_tree)
    # first_hash_merge(load_rp_tree)
    # load_interval_tree.trag_hash_merge(load_interval_tree.root)

    #1540131000, 1540174600
    #1539691743, 1539692474
    #1539691743 ,1539693300
    #1538691743 ,1538693300
    #1539666283, 1539666930
    # start_time = time.time()
    # interval_set = load_interval_tree.search_intersecting_intervals(Interval(1539689000, 1539696200))
    # print(len(interval_set))
    # # CSV file path
    # csv_file_path = 'trajectories_chengdu.csv_process.csv'  # Replace with your CSV file path
    # # Collect all points
    # result_points = collect_points(csv_file_path, interval_set)
    #
    # # Output results
    # print(f"Total number of unique points passed by all trajectories: {len(result_points)}")
    # print(result_points)
    # end_time = time.time()
    # print("Interval Tree query time:", end_time - start_time)
    #
    # start_time = time.time()
    # rp_set = range_query(load_rp_tree, (108.92, 108.97),  (34.23, 34.28))
    # end_time = time.time()
    # print("Spatial Tree query time:", end_time - start_time)
    #
    # start_time = time.time()
    # result = interval_set & rp_set
    # end_time = time.time()
    # print("Intersection time:", end_time - start_time)
    #
    # filter_id(result, "trajectories_xian.csv")
    #
    #
    # time_all = final_filter_id("traj_filtered_xian.csv",108.92, 108.97,  34.23 , 34.28,1539301779,1539388179)
    #
    # print("Filtering time:", time_all)
    #
    # # Interval Tree root hash
    # # root_hash = load_interval_tree.root_hash()
    # # print(root_hash)
    # #
    # import csv
    # csv.field_size_limit(10 * 1024 * 1024)  # 10MB
    # start_time = time.time()  # Record start time
    # if proof_interval("vo_time.csv"):
    #       print("Verification succeeded")
    # end_time = time.time()
    # print("Interval Tree verification time:", end_time - start_time)
    #
    # start_time = time.time()  # Record start time
    # if proof_vo('vo.csv'):
    #     print("Verification succeeded")
    # else:
    #     print("Verification failed")
    # end_time = time.time()
    # print("Spatial Tree verification time:", end_time - start_time)




