import csv
import hashlib
import json
import time

from filter_traj_id import filter_id, final_filter_id
from traj_time_insert import traj_to_dict, dict_to_traj, insert_time_stamp, update, insert_beijing_to_vtrq
from range_query import range_query, proof_vo


# Define function to convert RPTreeNode to dictionary (for JSON serialization)
def rptree_to_dict(node):
    if node is None:
        return None
    return {
        "border_lat": node.border_lat,
        "border_lng": node.border_lng,
        "linking": [edge_to_dict(edge) for edge in node.linking] if node.linking else None,
        "left": rptree_to_dict(node.left),
        "right": rptree_to_dict(node.right),
        "adjacent_list": [edge_to_dict(edge) for edge in node.adjacent_list] if node.adjacent_list else None,
        "rp_hash_merge": node.rp_hash_merge
    }


def edge_to_dict(edge):
    """Convert Edge object to dictionary for JSON serialization"""
    return {
        "id": edge.id,
        "start": vertex_to_dict(edge.start),
        "end": vertex_to_dict(edge.end),
        "traj_hashList": [traj_to_dict(traj) for traj in edge.traj_hashList],
        "edge_merge": edge.edge_merge,
        "weight": edge.weight
    }


# Convert Vertex class to dictionary (for JSON serialization)
def vertex_to_dict(vertex):
    """Convert Vertex object to dictionary for JSON serialization"""
    return {
        'id': vertex.id,
        'lat': vertex.lat,
        'lng': vertex.lng
    }


# Save RP-Tree to JSON file
def save_rp_tree_json(root, filename):
    """
    Save Recursive Partitioning Tree (RP-Tree) to JSON file
    
    :param root: Root node of RP-Tree
    :param filename: Path to output JSON file
    """
    data = rptree_to_dict(root)
    try:
        with open(filename, 'w') as f:
            json.dump(data, f)
    except Exception as e:
        print(f"Error saving file: {e}")


# Define Vertex class to represent vertices in the graph
class Vertex:
    """Class representing a vertex (node) in the spatial graph"""
    def __init__(self, id, lat, lng):
        # Unique identifier of the vertex
        self.id = id
        # Latitude coordinate of the vertex
        self.lat = lat
        # Longitude coordinate of the vertex
        self.lng = lng


# Define Edge class to represent connections between two vertices in the graph
class Edge:
    """Class representing an edge (connection) between two vertices in the spatial graph"""
    def __init__(self, id, start, end, weight):
        # Unique identifier of the edge
        self.id = id
        # Start vertex of the edge
        self.start = start
        # End vertex of the edge
        self.end = end
        # New attribute to store trajectory hash list (interval tree root hash)
        self.traj_hashList = []
        # Merged hash value for the edge
        self.edge_merge = ""
        # Weight/cost of the edge
        self.weight = weight

    def __repr__(self):
        return f"Edge(id={self.id}, start={self.start}, end={self.end})"


# Define RP-Tree Node class for building Recursive Partitioning Tree
class RPTreeNode:
    """Class representing a node in the Recursive Partitioning Tree (RP-Tree)"""
    def __init__(self, border_lat, border_lng, linking=None, left=None, right=None,
                 adjacent_list=None):
        # Latitude bounds of the region covered by this node (min_lat, max_lat)
        self.border_lat = border_lat
        # Longitude bounds of the region covered by this node (min_lng, max_lng)
        self.border_lng = border_lng
        # Linking edges (cross-subtree edges)
        self.linking = linking if linking else []
        # Left child node
        self.left = left
        # Right child node
        self.right = right
        # Adjacent edges (edges within the region)
        self.adjacent_list = adjacent_list if adjacent_list else []
        # Merged hash value for the RP-Tree node
        self.rp_hash_merge = ""

    def is_leaf(self):
        """Check if the node is a leaf node (no children)"""
        return self.left is None and self.right is None


def find_best_split(sorted_V, E, d):
    """
    Find the optimal split point to minimize weight difference between left/right subtrees
    
    :param sorted_V: Sorted list of vertices (by lat/lng)
    :param E: List of edges in current region
    :param d: Current recursion depth level
    :return: Index of optimal split point
    """
    # Get length of sorted vertex list
    n = len(sorted_V)
    # Calculate middle split point index
    mid = n // 2

    # Precompute adjacency list for each vertex (optimization)
    vertex_edges = {v: [] for v in sorted_V}
    for e in E:
        vertex_edges[e.start].append(e)
        vertex_edges[e.end].append(e)

    # Inner function to calculate total edge weight for left/right vertex sets
    def calculate_edges_weight(left_vertices, right_vertices):
        """Calculate total edge weight for left and right vertex subsets"""
        left_vertices_set = set(left_vertices)
        right_vertices_set = set(right_vertices)
        left_weight = 0
        right_weight = 0
        
        # Calculate weight for left subset edges
        for v in left_vertices:
            for e in vertex_edges[v]:
                if e.start in left_vertices_set and e.end in left_vertices_set:
                    left_weight += e.weight
        
        # Calculate weight for right subset edges
        for v in right_vertices:
            for e in vertex_edges[v]:
                if e.start in right_vertices_set and e.end in right_vertices_set:
                    right_weight += e.weight
        
        return left_weight, right_weight

    # Calculate left/right vertex sets for middle split point
    left_vertices_mid = sorted_V[:mid + 1]
    right_vertices_mid = sorted_V[mid + 1:]
    # Calculate edge weights for middle split
    left_edges_weight_mid, right_edges_weight_mid = calculate_edges_weight(left_vertices_mid, right_vertices_mid)
    # Calculate weight difference for middle split
    min_diff = abs(left_edges_weight_mid - right_edges_weight_mid)
    # Initialize best split point as middle point
    best_p = mid

    # Search left for better split point
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
            # Stop when difference starts increasing
            break

    # Search right for better split point
    right_diff = min_diff
    right_p = mid
    for p in range(mid + 1, n):
        left_vertices = sorted_V[:p + 1]
        right_vertices = sorted_V[p + 1:]
        left_edges_weight, right_edges_weight = calculate_edges_weight(left_vertices, right_vertices)
        diff = abs(left_edges_weight - right_edges_weight)
        
        if diff < right_diff:
            right_diff = diff
            right_p = p
        else:
            # Stop when difference starts increasing
            break

    # Update best split point based on minimal difference
    if left_diff < min_diff:
        min_diff = left_diff
        best_p = left_p
    if right_diff < min_diff:
        min_diff = right_diff
        best_p = right_p

    # Recalculate weights for best split point
    left_vertices_best = sorted_V[:best_p + 1]
    right_vertices_best = sorted_V[best_p + 1:]
    left_edges_weight_best, right_edges_weight_best = calculate_edges_weight(left_vertices_best, right_vertices_best)

    # Print split point information
    print(f"At level {d}, best split point p = {best_p}, left edges weight: {left_edges_weight_best}, right edges weight: {right_edges_weight_best}")
    return best_p


# Optimized split function (using pre-sorted vertex lists)
def split(V, E, h, uH, d, global_sorted_by_lat, global_sorted_by_lng):
    """
    Recursively split the spatial region to build RP-Tree
    
    :param V: List of vertices in current region
    :param E: List of edges in current region
    :param h: Minimum edge count for leaf node
    :param uH: Maximum recursion depth
    :param d: Current recursion depth
    :param global_sorted_by_lat: Globally pre-sorted vertices (by latitude)
    :param global_sorted_by_lng: Globally pre-sorted vertices (by longitude)
    :return: RPTreeNode for current region
    """
    # Calculate current node bounds
    min_lat = min(v.lat for v in V)
    max_lat = max(v.lat for v in V)
    min_lng = min(v.lng for v in V)
    max_lng = max(v.lng for v in V)
    
    node = RPTreeNode((min_lat, max_lat), (min_lng, max_lng))
    print(f"Level {d}: Lat ({min_lat}, {max_lat}), Lng ({min_lng}, {max_lng})")

    # Termination condition: leaf node (max depth or insufficient edges)
    if d >= uH or len(E) < h:
        # Filter edges within current node bounds
        node.adjacent_list = [e for e in E if
                              min_lat <= e.start.lat <= max_lat and
                              min_lng <= e.start.lng <= max_lng and
                              min_lat <= e.end.lat <= max_lat and
                              min_lng <= e.end.lng <= max_lng]
        print(f"Level {d} (leaf): {len(node.adjacent_list)} edges")
        return node

    # Choose split direction (latitude/longitude) based on range
    lat_range = max_lat - min_lat
    lng_range = max_lng - min_lng
    split_by_lat = lat_range > lng_range
    current_v_ids = {v.id for v in V}  # Fast filtering using ID set

    # Filter pre-sorted vertex list to current subset (maintain order)
    if split_by_lat:
        sorted_V = [v for v in global_sorted_by_lat if v.id in current_v_ids]
    else:
        sorted_V = [v for v in global_sorted_by_lng if v.id in current_v_ids]
    print(f"Level {d} split by {'latitude' if split_by_lat else 'longitude'}")

    # Find optimal split point
    best_p = find_best_split(sorted_V, E, d)
    V_left = sorted_V[:best_p + 1]
    V_right = sorted_V[best_p + 1:]

    # Split edge set into left/right subsets
    V_left_set = set(V_left)
    V_right_set = set(V_right)
    E_left = [e for e in E if e.start in V_left_set and e.end in V_left_set]
    E_right = [e for e in E if e.start in V_right_set and e.end in V_right_set]

    # Recursively build left subtree
    print(f"Recursively building left subtree at level {d + 1}...")
    node.left = split(V_left, E_left, h, uH, d + 1, global_sorted_by_lat, global_sorted_by_lng)
    
    # Recursively build right subtree
    print(f"Recursively building right subtree at level {d + 1}...")
    node.right = split(V_right, E_right, h, uH, d + 1, global_sorted_by_lat, global_sorted_by_lng)

    # Process cross-subtree edges (linking edges)
    node.linking = [e for e in E if
                    (e.start in V_left_set and e.end in V_right_set) or
                    (e.start in V_right_set and e.end in V_left_set)]

    return node


# Load vertex and edge data from files
def load_graph(node_file, edge_file):
    """
    Load graph data (vertices and edges) from text files
    
    :param node_file: Path to vertex data file
    :param edge_file: Path to edge data file
    :return: Tuple of (vertex_list, edge_list)
    """
    # List to store vertices
    V = []
    # List to store edges
    E = []
    # Dictionary for fast vertex lookup by ID
    vertex_dict = {}

    # Load vertex data
    with open(node_file, 'r') as f:
        for line in f:
            # Split line by whitespace
            parts = line.strip().split()
            # Get vertex ID
            id = int(parts[0])
            # Get vertex latitude
            lat = float(parts[2])
            # Get vertex longitude
            lng = float(parts[1])
            # Create Vertex object
            vertex = Vertex(id, lat, lng)
            # Add to vertex list
            V.append(vertex)
            # Add to vertex dictionary
            vertex_dict[id] = vertex

    # Load edge data
    with open(edge_file, 'r') as f:
        for line in f:
            # Split line by whitespace
            parts = line.strip().split()
            # Get edge ID
            id = int(parts[0])
            # Get start vertex ID
            start_id = int(parts[1])
            # Get end vertex ID
            end_id = int(parts[2])
            # Get edge weight
            weight = float(parts[3])
            
            # Get start/end vertices from dictionary
            start_vertex = vertex_dict[start_id]
            end_vertex = vertex_dict[end_id]
            
            # Create Edge object
            edge = Edge(id, start_vertex, end_vertex, weight)
            # Add to edge list
            E.append(edge)

    return V, E


# Build RPTreeNode from dictionary (JSON deserialization)
def dict_to_rptree(data):
    """
    Reconstruct RP-Tree from dictionary (JSON deserialization)
    
    :param data: Dictionary from JSON file
    :return: Root RPTreeNode
    """
    if data is None:
        return None
    
    # Reconstruct adjacent edges
    if data["adjacent_list"] is not None:
        adjacent_list = [dict_to_edge(edge_data) for edge_data in data["adjacent_list"]]
    else:
        adjacent_list = []
    
    # Reconstruct linking edges
    if data["linking"] is not None:
        linking = [dict_to_edge(edge_data) for edge_data in data["linking"]]
    else:
        linking = []
    
    # Create RPTreeNode
    node = RPTreeNode(data["border_lat"], data["border_lng"])
    node.adjacent_list = adjacent_list
    node.linking = linking
    node.rp_hash_merge = data["rp_hash_merge"]
    node.left = dict_to_rptree(data["left"])
    node.right = dict_to_rptree(data["right"])
    
    return node


def dict_to_vertex(vertex_data):
    """Reconstruct Vertex object from dictionary"""
    return Vertex(vertex_data['id'], vertex_data['lat'], vertex_data['lng'])


def dict_to_edge(edge_data):
    """Reconstruct Edge object from dictionary"""
    # Reconstruct trajectory hash list
    if edge_data["traj_hashList"]:
        traj_hashList = [dict_to_traj(traj_dict) for traj_dict in edge_data["traj_hashList"]]
    else:
        traj_hashList = []
    
    # Reconstruct start/end vertices
    start = dict_to_vertex(edge_data['start'])
    end = dict_to_vertex(edge_data['end'])
    
    # Create Edge object
    edge = Edge(edge_data["id"], start, end, edge_data["weight"])
    edge.traj_hashList = traj_hashList
    edge.edge_merge = edge_data["edge_merge"]
    
    return edge


# Load RP-Tree from JSON file
def load_rp_tree_json(filename):
    """
    Load RP-Tree from JSON file
    
    :param filename: Path to JSON file
    :return: Root RPTreeNode
    """
    with open(filename, 'r') as f:
        data = json.load(f)
    return dict_to_rptree(data)


# Traverse all nodes and count total edges
def count_edges(root):
    """
    Count total number of edges in RP-Tree (adjacent + linking edges)
    
    :param root: Root RPTreeNode
    :return: Total edge count
    """
    if root is None:
        return 0
    
    # Count linking edges
    a = len(root.linking) if root.linking else 0
    # Count adjacent edges
    b = len(root.adjacent_list) if root.adjacent_list else 0
    current_edges = a + b
    
    # Recursively count left/right subtree edges
    left_edges = count_edges(root.left)
    right_edges = count_edges(root.right)
    
    # Return total edge count
    return current_edges + left_edges + right_edges


# Second-level hash merging (for edges within RP-Tree node)
def second_hash_merge(edge):
    """
    Compute merged hash for a single edge (second-level hashing)
    Includes edge coordinates and trajectory hashes
    
    :param edge: Edge object to compute hash for
    """
    hash_list = []
    # Add edge coordinates to hash list
    hash_list.append(str(edge.start.lng))
    hash_list.append(str(edge.start.lat))
    hash_list.append(str(edge.end.lng))
    hash_list.append(str(edge.end.lat))
    
    # Add trajectory hashes to hash list
    for traj in edge.traj_hashList:
        hash_list.append(traj.merge_hash)
    
    # Compute SHA256 hash
    all_hash_join = '+'.join(item for item in hash_list)
    encoded_data = all_hash_join.encode('utf-8')
    hash_object = hashlib.sha256(encoded_data)
    hash_hex = hash_object.hexdigest()
    
    # Store merged hash in edge
    edge.edge_merge = hash_hex


# First-level hash merging (for RP-Tree nodes)
def first_hash_merge(node):
    """
    Compute merged hash for RP-Tree node (first-level hashing)
    Post-order traversal: process children first, then current node
    
    :param node: RPTreeNode to compute hash for
    """
    if node is None:
        return
    
    # Recursively process left subtree
    first_hash_merge(node.left)
    # Recursively process right subtree
    first_hash_merge(node.right)

    # Process current node (post-order)
    if node.is_leaf():
        if node.adjacent_list:
            # Compute merged hash for each edge
            for edge in node.adjacent_list:
                second_hash_merge(edge)

            # Build hash list for leaf node
            str_list = []
            # Add node bounds to hash list
            str_list.append(str(node.border_lng[0]))
            str_list.append(str(node.border_lng[1]))
            str_list.append(str(node.border_lat[0]))
            str_list.append(str(node.border_lat[1]))

            # Add edge merged hashes to hash list
            for edge in node.adjacent_list:
                str_list.append(edge.edge_merge)

            # Compute node merged hash
            all_hash_join = '+'.join(item for item in str_list)
            encoded_data = all_hash_join.encode('utf-8')
            hash_object = hashlib.sha256(encoded_data)
            hash_hex = hash_object.hexdigest()
            node.rp_hash_merge = hash_hex
    # Internal node (non-leaf)
    else:
        # Build hash list for internal node
        str_list = []
        # Add node bounds to hash list
        str_list.append(str(node.border_lng[0]))
        str_list.append(str(node.border_lng[1]))
        str_list.append(str(node.border_lat[0]))
        str_list.append(str(node.border_lat[1]))
        
        # Process linking edges
        if node.linking:
            for edge in node.linking:
                second_hash_merge(edge)
            for edge in node.linking:
                str_list.append(edge.edge_merge)
        
        # Add child node hashes
        if node.left:
            str_list.append(node.left.rp_hash_merge)
        if node.right:
            str_list.append(node.right.rp_hash_merge)
        
        # Compute node merged hash
        all_hash_join = '+'.join(item for item in str_list)
        encoded_data = all_hash_join.encode('utf-8')
        hash_object = hashlib.sha256(encoded_data)
        hash_hex = hash_object.hexdigest()
        node.rp_hash_merge = hash_hex


if __name__ == "__main__":
    # Vertex data file path
    node_file = "filtered_node.txt"
    # Edge data file path
    edge_file = "filtered_edges_indirected_weight.txt"

    # Load graph data
    print("Starting data loading...")
    V, E = load_graph(node_file, edge_file)
    
    if not V or not E:
        print("Data loading failed, exiting program")
    else:
        # Pre-sort vertices (once globally)
        print("Pre-sorting vertices...")
        global_sorted_by_lat = sorted(V, key=lambda v: v.lat)
        global_sorted_by_lng = sorted(V, key=lambda v: v.lng)

    # Build vertex dictionary for fast lookup
    vertex_dict = {v.id: v for v in V}

    # Load pre-built RP-Tree from JSON
    load_rp_tree = load_rp_tree_json("chengdu_rp_3days.json")

    # Perform spatiotemporal range query
    start_time = time.time()
    result = range_query(load_rp_tree, (104.08, 104.10), (30.66, 30.68), 1538413860, 1538417460)
    end_time = time.time()
    query_time = end_time - start_time

    # Set CSV field size limit (10MB)
    csv.field_size_limit(10 * 1024 * 1024)
    
    # Verify VO (Verification Object)
    start_time = time.time()
    if proof_vo('vo.csv'):
        print("Verification successful")
    else:
        print("Verification failed")
    end_time = time.time()
    verification_time = end_time - start_time

    # Print query results
    print(len(result), "spatiotemporal query results")

    # Filter trajectory IDs
    filter_id(result, "trajectories_chengdu.csv")
    
    # Final filter by spatiotemporal range
    aggregation_time = final_filter_id("traj_filtered_chengdu.csv", 104.08, 104.10, 30.66, 30.68, 1538413860, 1538417460)

    # Print performance metrics
    print(query_time, "query time (seconds)")
    print(verification_time, "verification time (seconds)")
    print(aggregation_time, "aggregation time (seconds)")
