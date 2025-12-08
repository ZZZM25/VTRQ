import hashlib
import csv
import json
import time
from collections import deque  # Optimize queue operations to improve large-scale data processing efficiency
from typing import List, Optional, Union, Tuple, Dict, Any, Iterable

from filter_traj_id import filter_id, final_filter_id
#from process_beijing_trace_point_files import process_all_files, process_all_files2


def hash_value(data: Union[str, bytes, float]) -> str:
    """Calculate SHA-256 hash value, uniformly process longitude with 6-decimal string (resolve floating-point precision issues)"""
    if isinstance(data, float):
        # Force convert longitude to 6-decimal string to completely avoid floating-point storage errors
        data = f"{round(data, 6):.6f}"
    if isinstance(data, str):
        data = data.encode('utf-8')
    return hashlib.sha256(data).hexdigest()


class MerkleBTreeNode:
    """Merkle B-Tree Node: Store longitude as string to fix verification failures caused by floating-point precision"""

    def __init__(self, is_leaf: bool = False):
        self.keys: List[str] = []  # Store 6-decimal strings (e.g., "116.405290")
        self.values: List[List[str]] = []  # Trace hash list
        self.children: List[MerkleBTreeNode] = []  # Child node list
        self.is_leaf: bool = is_leaf  # Whether it's a leaf node
        self.hash: str = ""  # Node hash
        self.parent: Optional[MerkleBTreeNode] = None  # Parent node reference

    def update_hash(self) -> None:
        """Update node hash, ensure child node hashes are calculated first during recursive update"""
        if self.is_leaf:
            entries = [f"{lon_str}:{','.join(hashes)}" for lon_str, hashes in zip(self.keys, self.values)]
            self.hash = hash_value("|".join(entries))
        else:
            child_hashes = "|".join([child.hash for child in self.children])
            lon_str = "|".join(self.keys)  # Direct string concatenation
            self.hash = hash_value(f"{child_hashes}|{lon_str}")

        # Trigger update if parent node exists (ensure complete hash chain)
        if self.parent:
            self.parent.update_hash()

    def find_lon_index(self, longitude_str: str) -> int:
        """Precisely find longitude index through string matching (completely resolve floating-point errors)"""
        for i, lon_str in enumerate(self.keys):
            if lon_str == longitude_str:
                return i
        return -1

    def to_dict(self) -> Dict[str, Any]:
        """Serialize node to dictionary (maintain string longitude)"""
        node_dict = {
            "is_leaf": self.is_leaf,
            "keys": self.keys,  # Store strings directly without conversion
            "values": self.values,
            "hash": self.hash
        }
        if not self.is_leaf:
            node_dict["children"] = [child.to_dict() for child in self.children]
        return node_dict

    @classmethod
    def from_dict(cls, node_dict: Dict[str, Any], parent: Optional["MerkleBTreeNode"] = None) -> "MerkleBTreeNode":
        """Deserialize node from dictionary (restore string longitude)"""
        node = cls(is_leaf=node_dict["is_leaf"])
        node.keys = node_dict["keys"]  # Directly read string longitude
        node.values = node_dict["values"]
        node.hash = node_dict["hash"]
        node.parent = parent

        if not node.is_leaf and "children" in node_dict:
            node.children = [cls.from_dict(child_dict, parent=node) for child_dict in node_dict["children"]]
        return node

    def __str__(self) -> str:
        return f"Node(lons={self.keys[:3]}..., hash={self.hash[:8]}..., is_leaf={self.is_leaf})"


class LocationMerkleBTree:
    """Merkle B-Tree with fixed verification failure issues (supports large-scale data processing)"""

    def __init__(self, order: int = 4):
        self.root: MerkleBTreeNode = MerkleBTreeNode(is_leaf=True)
        self.order: int = order  # Order:建议 set to 30~50 for 500,000 nodes

    def insert_trace(self, longitude: float, trace_hash: str) -> None:
        """Insert trace hash (use string longitude to avoid duplicates)"""
        # Convert to 6-decimal string as unique identifier (core fix point)
        target_lon_str = f"{round(longitude, 6):.6f}"
        root = self.root

        if len(root.keys) == self.order - 1:
            new_root = MerkleBTreeNode(is_leaf=False)
            self.root = new_root
            new_root.children.append(root)
            root.parent = new_root
            self._split_child(new_root, 0)
            self._insert_non_full(new_root, target_lon_str, trace_hash)
        else:
            self._insert_non_full(root, target_lon_str, trace_hash)

    def _insert_non_full(self, node: MerkleBTreeNode, longitude_str: str, trace_hash: str) -> None:
        i = len(node.keys) - 1
        lon_index = node.find_lon_index(longitude_str)

        if node.is_leaf:
            if lon_index != -1:
                # Longitude exists, add trace hash after deduplication
                if trace_hash not in node.values[lon_index]:
                    node.values[lon_index].append(trace_hash)
                    node.update_hash()  # Update hash immediately
            else:
                # Insert new longitude (sorted by string)
                while i >= 0 and longitude_str < node.keys[i]:
                    i -= 1
                node.keys.insert(i + 1, longitude_str)
                node.values.insert(i + 1, [trace_hash])
                node.update_hash()  # Update hash immediately
        else:
            # Find child node in internal node
            while i >= 0 and longitude_str < node.keys[i]:
                i -= 1
            i += 1

            # Split child node if full
            if len(node.children[i].keys) == self.order - 1:
                self._split_child(node, i)
                if longitude_str > node.keys[i]:
                    i += 1

            # Recursively insert into child node
            self._insert_non_full(node.children[i], longitude_str, trace_hash)

    def _split_child(self, parent: MerkleBTreeNode, child_idx: int) -> None:
        """Split child node (fix hash update order to ensure child nodes are updated first)"""
        order = self.order
        child = parent.children[child_idx]
        new_node = MerkleBTreeNode(is_leaf=child.is_leaf)
        new_node.parent = parent

        mid_idx = (order - 1) // 2
        mid_key = child.keys[mid_idx]

        # Split data
        new_node.keys = child.keys[mid_idx + 1:]
        new_node.values = child.values[mid_idx + 1:]
        child.keys = child.keys[:mid_idx]
        child.values = child.values[:mid_idx]

        # Split child nodes (non-leaf nodes)
        if not child.is_leaf:
            new_node.children = child.children[mid_idx + 1:]
            for c in new_node.children:
                c.parent = new_node
                c.update_hash()  # First update hashes of new node's children
            child.children = child.children[:mid_idx + 1]
            for c in child.children:
                c.update_hash()  # Then update hashes of original node's children

        # Critical fix: Update hashes of child and new nodes first (ensure completion)
        child.update_hash()
        new_node.update_hash()

        # Insert into parent node and update parent hash
        parent.children.insert(child_idx + 1, new_node)
        parent.keys.insert(child_idx, mid_key)
        parent.update_hash()

    def query_and_extract(self, longitude: float) -> Tuple[bool, Optional[List[str]], Optional[Dict]]:
        """Query trace and validation path for single longitude (use string longitude matching)"""
        target_lon_str = f"{round(longitude, 6):.6f}"
        current = self.root

        while True:
            lon_index = current.find_lon_index(target_lon_str)
            if lon_index != -1 and current.is_leaf:
                # Extract trace hash list
                trace_hashes = current.values[lon_index].copy()
                # Build validation path
                validation_path = {
                    "longitude": float(target_lon_str),  # Store as float for display
                    "longitude_str": target_lon_str,  # Store string for precise verification
                    "leaf_entries": [f"{lon_str}:{','.join(hashes)}" for lon_str, hashes in
                                     zip(current.keys, current.values)],
                    "leaf_hash": current.hash,
                    "lon_index": lon_index,
                    "path": [],
                    "root_hash": self.root.hash
                }

                # Collect path from leaf to root
                leaf_node = current
                while leaf_node.parent is not None:
                    parent = leaf_node.parent
                    child_idx = parent.children.index(leaf_node)
                    sibling_hashes = [child.hash for i, child in enumerate(parent.children) if i != child_idx]
                    validation_path["path"].append({
                        "child_idx": child_idx,
                        "sibling_hashes": sibling_hashes.copy(),
                        "parent_keys": parent.keys.copy()  # Store string longitude
                    })
                    leaf_node = parent

                return (True, trace_hashes, validation_path)
            elif current.is_leaf:
                return (False, None, None)
            else:
                # Navigate internal nodes
                i = 0
                while i < len(current.keys) and target_lon_str > current.keys[i]:
                    i += 1
                current = current.children[i]

    def query_by_range(self, min_lon: float, max_lon: float) -> List[Tuple[float, List[str], Dict]]:
        """Range query (use string comparison to ensure precision)"""
        min_lon_str = f"{round(min_lon, 6):.6f}"
        max_lon_str = f"{round(max_lon, 6):.6f}"
        result = []

        # Optimize queue operations with deque (critical for large-scale data processing)
        queue = deque([self.root])
        while queue:
            node = queue.popleft()
            if node.is_leaf:
                for lon_str in node.keys:
                    # Ensure accurate range judgment with string comparison
                    if min_lon_str <= lon_str <= max_lon_str:
                        exists, traces, path = self.query_and_extract(float(lon_str))
                        if exists and traces and path:
                            result.append((float(lon_str), traces, path))
            else:
                queue.extend(node.children)

        return sorted(result, key=lambda x: x[0])

    def save_range_query_results_to_csv(self, range_results: List[Tuple[float, List[str], Dict]],
                                        csv_path: str) -> None:
        """Save range query results to CSV"""
        if not range_results:
            print("⚠️  Range query results are empty, not saving CSV")
            return

        with open(csv_path, 'w', encoding='utf-8', newline='') as f:
            fieldnames = [
                "longitude", "trace_hash", "leaf_hash", "root_hash",
                "leaf_entries", "validation_path", "verify_result"
            ]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for lon, traces, path in range_results:
                is_valid = self.independent_verify(traces, path)
                for trace in traces:
                    row = {
                        "longitude": f"{lon:.6f}",
                        "trace_hash": trace,
                        "leaf_hash": path["leaf_hash"],
                        "root_hash": path["root_hash"],
                        "leaf_entries": json.dumps(path["leaf_entries"], ensure_ascii=False),
                        "validation_path": json.dumps(path["path"], ensure_ascii=False),
                        "verify_result": is_valid
                    }
                    writer.writerow(row)

        print(f"✅ Range query results saved to CSV: {csv_path}")

    @staticmethod
    def independent_verify(trace_hashes: List[str], validation_path: Dict) -> bool:
        """Independently verify trace authenticity (use string longitude to ensure matching)"""
        if not validation_path:
            return False

        # Core fix: Use string longitude for precise matching
        lon_str = validation_path["longitude_str"]
        leaf_entries = validation_path["leaf_entries"]
        lon_index = validation_path["lon_index"]
        leaf_hash = validation_path["leaf_hash"]

        # Verify index validity
        if lon_index < 0 or lon_index >= len(leaf_entries):
            return False
        # Verify consistency between trace list and leaf entries
        expected_trace_str = ",".join(trace_hashes)
        expected_entry = f"{lon_str}:{expected_trace_str}"
        if leaf_entries[lon_index] != expected_entry:
            return False
        # Verify leaf node hash
        calculated_leaf_hash = hash_value("|".join(leaf_entries))
        if calculated_leaf_hash != leaf_hash:
            return False

        # Verify path to root node
        current_hash = leaf_hash
        for step in validation_path["path"]:
            child_idx = step["child_idx"]
            sibling_hashes = step["sibling_hashes"].copy()
            parent_keys = step["parent_keys"]  # String longitude list

            sibling_hashes.insert(child_idx, current_hash)
            child_hashes_str = "|".join(sibling_hashes)
            parent_keys_str = "|".join(parent_keys)  # Direct string concatenation
            current_hash = hash_value(f"{child_hashes_str}|{parent_keys_str}")

        # Final root hash verification
        return current_hash == validation_path["root_hash"]

    @staticmethod
    def batch_verify(verify_items: Iterable[Tuple[List[str], Dict]]) -> List[Tuple[float, bool]]:
        """Batch verify trace list"""
        results = []
        for traces, path in verify_items:
            if not path:
                continue
            longitude = path.get("longitude", 0.0)
            is_valid = LocationMerkleBTree.independent_verify(traces, path)
            results.append((longitude, is_valid))
        return results

    def save_all_validation_paths_to_csv(self, csv_path: str) -> None:
        """Save all trace validation paths to CSV (deduplicate using string longitude)"""
        all_traces = self._collect_all_traces()
        if not all_traces:
            print("⚠️  No trace data to save to CSV")
            return

        with open(csv_path, 'w', encoding='utf-8', newline='') as f:
            fieldnames = [
                "longitude", "trace_hash", "leaf_hash", "root_hash",
                "leaf_entries", "validation_path", "verify_result"
            ]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            # Deduplicate using string longitude (core fix point)
            processed_lons = set()
            for lon_str, trace_hash in all_traces:
                if lon_str not in processed_lons:
                    exists, _, validation_path = self.query_and_extract(float(lon_str))
                    if not exists:
                        continue
                    processed_lons.add(lon_str)

                row = {
                    "longitude": lon_str,
                    "trace_hash": trace_hash,
                    "leaf_hash": validation_path["leaf_hash"],
                    "root_hash": validation_path["root_hash"],
                    "leaf_entries": json.dumps(validation_path["leaf_entries"], ensure_ascii=False),
                    "validation_path": json.dumps(validation_path["path"], ensure_ascii=False),
                    "verify_result": self.independent_verify(
                        trace_hashes=[t for l, t in all_traces if l == lon_str],
                        validation_path=validation_path
                    )
                }
                writer.writerow(row)

        print(f"✅ All trace validation paths saved to: {csv_path}")

    def save_to_json(self, json_path: str) -> None:
        """Save tree to JSON"""
        tree_data = {
            "order": self.order,
            "root": self.root.to_dict(),
            "root_hash": self.root.hash
        }

        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(tree_data, f, ensure_ascii=False, indent=2)
        print(f"✅ Tree saved to JSON file: {json_path}")

    @classmethod
    def load_from_json(cls, json_path: str) -> "LocationMerkleBTree":
        """Load tree from JSON and verify integrity"""
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                tree_data = json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"❌ JSON file not found: {json_path}")
        except json.JSONDecodeError:
            raise ValueError(f"❌ Invalid JSON format: {json_path}")

        required_fields = ["order", "root", "root_hash"]
        if not all(field in tree_data for field in required_fields):
            raise ValueError(f"❌ Missing required fields in JSON: {required_fields}")

        tree = cls(order=tree_data["order"])
        tree.root = MerkleBTreeNode.from_dict(tree_data["root"])

        # Verify root hash
        if tree.root.hash != tree_data["root_hash"]:
            raise ValueError(f"❌ Tree hash verification failed, data may be corrupted")
        # Verify all node hashes
        if not tree._verify_all_node_hashes(tree.root):
            raise ValueError(f"❌ Inconsistent node hashes, data may be tampered with")

        print(f"✅ Tree loaded from JSON file: {json_path}")
        return tree

    def _verify_all_node_hashes(self, node: MerkleBTreeNode) -> bool:
        """Recursively verify all node hashes"""
        original_hash = node.hash
        node.update_hash()
        if node.hash != original_hash:
            return False

        if not node.is_leaf:
            for child in node.children:
                if not self._verify_all_node_hashes(child):
                    return False
        return True

    def _collect_all_traces(self) -> List[Tuple[str, str]]:
        """Collect all traces (return string longitude + hash)"""
        all_traces = []
        queue = deque([self.root])  # Optimize large-scale traversal with deque
        while queue:
            node = queue.popleft()
            if node.is_leaf:
                for lon_str, trace_hashes in zip(node.keys, node.values):
                    for trace in trace_hashes:
                        all_traces.append((lon_str, trace))  # Store string longitude
            else:
                queue.extend(node.children)
        return all_traces

    def get_root_hash(self) -> str:
        """Get root node hash"""
        return self.root.hash

    def print_tree(self, node: Optional[MerkleBTreeNode] = None, level: int = 0) -> None:
        """Print tree structure (for debugging)"""
        if node is None:
            node = self.root
        print("  " * level + f"Level {level}: {node}")
        if not node.is_leaf:
            for child in node.children:
                self.print_tree(child, level + 1)


def traj_timestamp_insert(traj, tree):
    traj_str = '->'.join(str(item) for item in traj[0])
    # Find the overall start and end time of the trajectory
    traj_start_time = traj[2][0][0][0]
    traj_end_time = traj[2][-1][-1][0]
    traj_start_time_str = str(traj_start_time)
    traj_end_time_str = str(traj_end_time)
    str_time_traj = traj_str + traj_start_time_str + traj_end_time_str
    encoded_data = str_time_traj.encode('utf-8')
    hash_object = hashlib.sha256(encoded_data)
    traj_hash = hash_object.hexdigest()
    for i in traj[2]:
        for j in i:
            rounded_num = round(j[1], 7)  # Longitude
            tree.insert_trace(rounded_num, traj_hash)


def insert_traj(tree):
    for i in range(1, 4):
        traj_name_path = f"C:\\Users\\maoyusen\\Desktop\\Graph-Diffusion-Planning-main\\chengdu-tra-json\\traj-10-{i}.json"
        # Open JSON file
        with open(traj_name_path, 'r', encoding='utf-8') as file:
            # Parse JSON data
            nested_list = json.load(file)
        j = 0
        for traj in nested_list:
            j = j + 1
            #if j < 1500:
            traj_timestamp_insert(traj, tree)
            print(f"Completed {j}th trajectory of 10-{i}")


def get_trajectory_intersection(jingdu_results, weidu_results, shijian_results):
    """
    Calculate the intersection of trajectories in three tree query results (based on trace hash)
    :param jingdu_results: Longitude tree query results
    :param weidu_results: Latitude tree query results
    :param shijian_results: Time tree query results
    :return: Set of intersecting trace hashes
    """
    # Extract all trace hashes from each tree query result (deduplicated)
    jingdu_hashes = set()
    for _, traces, _ in jingdu_results:
        jingdu_hashes.update(traces)

    weidu_hashes = set()
    for _, traces, _ in weidu_results:
        weidu_hashes.update(traces)

    shijian_hashes = set()
    for _, traces, _ in shijian_results:
        shijian_hashes.update(traces)

    # Calculate the intersection of the three sets
    intersection_hashes = jingdu_hashes & weidu_hashes & shijian_hashes
    return intersection_hashes


def get_trajectory_intersection(jingdu_results, weidu_results, shijian_results):
    """
    Calculate the intersection of trajectories in three tree query results (based on trace hash)
    :param jingdu_results: Longitude tree query results
    :param weidu_results: Latitude tree query results
    :param shijian_results: Time tree query results
    :return: Set of intersecting trace hashes
    """
    # Extract all trace hashes from each tree query result (deduplicated)
    jingdu_hashes = set()
    for _, traces, _ in jingdu_results:
        jingdu_hashes.update(traces)

    weidu_hashes = set()
    for _, traces, _ in weidu_results:
        weidu_hashes.update(traces)

    shijian_hashes = set()
    for _, traces, _ in shijian_results:
        shijian_hashes.update(traces)

    # Calculate the intersection of the three sets
    intersection_hashes = jingdu_hashes & weidu_hashes & shijian_hashes
    return intersection_hashes


if __name__ == "__main__":
    time1 = time.time()
    tree = LocationMerkleBTree(order=30)
    process_all_files2(tree)  # Time, Longitude, Latitude

    print("Insertion completed, original tree structure:")
    original_root_hash = tree.get_root_hash()
    time2 = time.time()
    print(time2 - time1)
    # 2. Save tree to JSON
    json_path = "mbt_time_beijing_3days.json"
    tree.save_to_json(json_path)
    # Load three trees (assumed to be loaded completely)
    loaded_tree_jingdu = LocationMerkleBTree.load_from_json("mbt_longitude_beijing_3days.json")
    loaded_tree_weidu = LocationMerkleBTree.load_from_json("mbt_latitude_beijing_3days.json")
    loaded_tree_shijian = LocationMerkleBTree.load_from_json("mbt_time_beijing_3days.json")

    # Execute respective range queries
    # Longitude range query
    min_jing, max_jing = 116.30, 116.33
    jingdu_results = loaded_tree_jingdu.query_by_range(min_jing, max_jing)

    # Latitude range query
    min_wei, max_wei = 39.97, 40
    weidu_results = loaded_tree_weidu.query_by_range(min_wei, max_wei)

    # Time range query (Note: Time tree stores timestamps, example range used here)
    min_time, max_time = 1233955024, 1233958624
    shijian_results = loaded_tree_shijian.query_by_range(min_time, max_time)

    # Calculate trajectory intersection
    intersection = get_trajectory_intersection(jingdu_results, weidu_results, shijian_results)

    # Output results
    print(f"Number of trajectories found in longitude tree: {len(set(t for _, traces, _ in jingdu_results for t in traces))}")
    print(f"Number of trajectories found in latitude tree: {len(set(t for _, traces, _ in weidu_results for t in traces))}")
    print(f"Number of trajectories found in time tree: {len(set(t for _, traces, _ in shijian_results for t in traces))}")
    print(f"Number of intersecting trajectories in three tree query results: {len(intersection)}")
    filter_id(intersection, "beijing_trace_summary_results.csv")
    time_all = final_filter_id("filter_beijing.csv", 116.30, 116.33, 39.97, 40, 1233955024, 1233958624)
    if intersection:
        print("Example of intersecting trace hashes:", list(intersection)[:5])  # Print first 5 examples
#if __name__ == "__main__":
    # time1 = time.time()
    # tree = LocationMerkleBTree(order=30)
    # #process_all_files2(tree)#Time, Longitude, Latitude
    # insert_traj(tree)
    # print("Insertion completed, original tree structure:")
    # original_root_hash = tree.get_root_hash()
    # time2 = time.time()
    # print(time2-time1)
    # # 2. Save tree to JSON
    # json_path = "mbt_latitude_chengdu_1days.json"
    # tree.save_to_json(json_path)

    #
    # loaded_tree_weidu = LocationMerkleBTree.load_from_json("mbt_latitude_beijing_3days.json")
    # loaded_tree_jingdu = LocationMerkleBTree.load_from_json("mbt_longitude_beijing_3days.json")
    # loaded_tree_shijian = LocationMerkleBTree.load_from_json("mbt_time_beijing_3days.json")
    # #
    # #
    #
    #
    # # # 4. Loaded tree executes range query and saves to CSV
    # #
    # min_lon, max_lon =  116.31, 116.34
    # time1 = time.time()
    # range_results = loaded_tree_jingdu.query_by_range(min_lon, max_lon)
    # time2 = time.time()
    # query_time_jingdu = time2 - time1
    # csv_path = "vo_longitude.csv"
    # loaded_tree_jingdu.save_range_query_results_to_csv(range_results, csv_path)
    # print(f"\nQuery results contain {len(range_results)} longitude points:")
    # for lon, traces, _ in range_results:
    #     print(f"Longitude {lon}: {len(traces)} trajectories")
    #
    # time1 = time.time()
    # # 5. Verify batch verification function of loaded tree
    # print("\n=== 5. Batch Verify Query Results ===")
    # verify_items = [(traces, path) for _, traces, path in range_results]
    # verify_results = loaded_tree_jingdu.batch_verify(verify_items)
    # for lon, is_valid in verify_results:
    #     print(f"Longitude {lon} verification result: {'Passed' if is_valid else 'Failed'}")
    # time2 = time.time()
    # vo_time_jingdu = time2 - time1
    #
    # min_lon, max_lon =  40.0, 40.03
    # time1 = time.time()
    # range_results = loaded_tree_weidu.query_by_range(min_lon, max_lon)
    # time2 = time.time()
    # query_time_weidu = time2 - time1
    # csv_path = "vo_latitude.csv"
    # loaded_tree_weidu.save_range_query_results_to_csv(range_results, csv_path)
    # print(f"\nQuery results contain {len(range_results)} longitude points:")
    # for lon, traces, _ in range_results:
    #     print(f"Longitude {lon}: {len(traces)} trajectories")
    #
    # time1 = time.time()
    # # 5. Verify batch verification function of loaded tree
    # print("\n=== 5. Batch Verify Query Results ===")
    # verify_items = [(traces, path) for _, traces, path in range_results]
    # verify_results = loaded_tree_weidu.batch_verify(verify_items)
    # for lon, is_valid in verify_results:
    #     print(f"Longitude {lon} verification result: {'Passed' if is_valid else 'Failed'}")
    # time2 = time.time()
    # vo_time_weidu = time2 - time1
    #
    # min_lon, max_lon = 1212121233 , 1212121293
    # time1 = time.time()
    # range_results = loaded_tree_shijian.query_by_range(min_lon, max_lon)
    # time2 = time.time()
    # query_time_shijian = time2 - time1
    # csv_path = "vo_time.csv"
    # loaded_tree_shijian.save_range_query_results_to_csv(range_results, csv_path)
    # print(f"\nQuery results contain {len(range_results)} longitude points:")
    # for lon, traces, _ in range_results:
    #     print(f"Longitude {lon}: {len(traces)} trajectories")
    #
    # time1 = time.time()

    # 5. Verify batch verification function of loaded tree
    # print("\n=== 5. Batch Verify Query Results ===")
    # verify_items = [(traces, path) for _, traces, path in range_results]
    # verify_results = loaded_tree_shijian.batch_verify(verify_items)
    # for lon, is_valid in verify_results:
    #     print(f"Longitude {lon} verification result: {'Passed' if is_valid else 'Failed'}")
    # time2 = time.time()
    # vo_time_shijian = time2 - time1
    #
    # print(query_time_weidu+query_time_shijian+query_time_shijian)
    #

    # print(vo_time_jingdu+vo_time_shijian+vo_time_weidu)
