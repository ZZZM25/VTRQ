import ast
import csv
import hashlib
import json
import random
import string
import time

from typing import List, Dict, Tuple, Optional, Union, Any

import pandas as pd
from tqdm import tqdm

from process_beijing_trajectory_points import process_all_files, Trajectory  # Translated module name


def verify_point_exists_exact(rtree, target_lng, target_lat):
    """Point verification function without error tolerance (strict equality)"""

    def traverse(node):
        if node.is_leaf:
            for rn in node.children:
                if rn.longitude == target_lng and rn.latitude == target_lat:
                    print(f"Found point: {rn.longitude}, {rn.latitude} (node children count: {len(node.children)})", node.mbr)
                    for i in node.children:
                        print(i.node_id, i.longitude, i.latitude)
                    while node.parent:
                        node = node.parent
                        print(node.mbr)
                    return True
            return False
        for child in node.children:
            if traverse(child):
                return True
        return False

    return traverse(rtree.root)


class RoadNode:
    """Road network node (adapted to 7-decimal longitude/latitude precision)"""

    def __init__(self, node_id: int, longitude: float, latitude: float, trajectories=None):
        self.node_id = node_id
        self.longitude = round(longitude, 7)  # Force to retain 7 decimal places
        self.latitude = round(latitude, 7)  # Force to retain 7 decimal places
        self.trajectories = trajectories if trajectories else []
        self.RoadNode_hash = ""

    def add_trajectory(self, trajectory: Trajectory) -> None:
        """Add trajectory, ensure no duplicates (judged by trajectory_hash)"""
        if not isinstance(trajectory, Trajectory):
            raise ValueError("Must add objects of Trajectory type")

        # Check if trajectory with the same hash already exists
        for existing_traj in self.trajectories:
            if existing_traj.trajectory_hash == trajectory.trajectory_hash:
                return

        self.trajectories.append(trajectory)

    def compute_combined_hash(self) -> str:
        """Compute combined hash containing all trajectories (maintain trajectory addition order)"""
        hashes = [str(self.node_id), str(self.longitude), str(self.latitude)]
        for traj in self.trajectories:
            hashes.append(traj.trajectory_hash)
        combined = '+'.join(hashes).encode('utf-8')
        self.RoadNode_hash = hashlib.sha256(combined).hexdigest()
        return self.RoadNode_hash

    @property
    def coords(self) -> Tuple[float, float]:
        return self.longitude, self.latitude

    def equals(self, other: "RoadNode") -> bool:
        """Exact matching for 7-decimal precision"""
        return (self.longitude == other.longitude) and (self.latitude == other.latitude)

    def __repr__(self) -> str:
        return f"RoadNode(id={self.node_id}, lng={self.longitude:.7f}, lat={self.latitude:.7f})"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to serializable dictionary"""
        return {
            "node_id": self.node_id,
            "longitude": self.longitude,
            "latitude": self.latitude,
            "trajectories": [t.to_dict() for t in self.trajectories],
            "RoadNode_hash": self.RoadNode_hash
        }

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "RoadNode":
        """Reconstruct RoadNode object from dictionary"""
        node = RoadNode(
            node_id=data["node_id"],
            longitude=data["longitude"],
            latitude=data["latitude"],
            trajectories=[Trajectory.from_dict(t) for t in data["trajectories"]]
        )
        node.RoadNode_hash = data["RoadNode_hash"]
        return node


class MBR:
    """Minimum Bounding Rectangle"""

    def __init__(self, min_lng: float, min_lat: float, max_lng: float, max_lat: float):
        self.min_lng = min_lng
        self.min_lat = min_lat
        self.max_lng = max_lng
        self.max_lat = max_lat

    def expand(self, target: Union["MBR", RoadNode, "RTreeNode", Tuple[float, float]]) -> "MBR":
        if isinstance(target, MBR):
            min_lng = min(self.min_lng, target.min_lng)
            min_lat = min(self.min_lat, target.min_lat)
            max_lng = max(self.max_lng, target.max_lng)
            max_lat = max(self.max_lat, target.max_lat)
        elif isinstance(target, RoadNode):
            min_lng = min(self.min_lng, target.longitude)
            min_lat = min(self.min_lat, target.latitude)
            max_lng = max(self.max_lng, target.longitude)
            max_lat = max(self.max_lat, target.latitude)
        elif isinstance(target, RTreeNode) and target.mbr:
            min_lng = min(self.min_lng, target.mbr.min_lng)
            min_lat = min(self.min_lat, target.mbr.min_lat)
            max_lng = max(self.max_lng, target.mbr.max_lng)
            max_lat = max(self.max_lat, target.mbr.max_lat)
        else:
            lng, lat = target
            min_lng = min(self.min_lng, lng)
            min_lat = min(self.min_lat, lat)
            max_lng = max(self.max_lng, lng)
            max_lat = max(self.max_lat, lat)
        return MBR(min_lng, min_lat, max_lng, max_lat)

    def contains(self, target: Union["MBR", RoadNode, Tuple[float, float]]) -> bool:
        if isinstance(target, MBR):
            return (self.min_lng <= target.min_lng and self.min_lat <= target.min_lat and
                    self.max_lng >= target.max_lng and self.max_lat >= target.max_lat)
        elif isinstance(target, RoadNode):
            return (self.min_lng <= target.longitude <= self.max_lng and
                    self.min_lat <= target.latitude <= self.max_lat)
        else:
            lng, lat = target
            return (self.min_lng <= lng <= self.max_lng and
                    self.min_lat <= lat <= self.max_lat)

    def intersects(self, other: "MBR") -> bool:
        return (self.min_lng <= other.max_lng and self.max_lng >= other.min_lng and
                self.min_lat <= other.max_lat and self.max_lat >= other.min_lat)

    def area(self) -> float:
        return (self.max_lng - self.min_lng) * (self.max_lat - self.min_lat)

    def __repr__(self) -> str:
        return f"MBR(lng: [{self.min_lng:.7f}, {self.max_lng:.7f}], " \
               f"lat: [{self.min_lat:.7f}, {self.max_lat:.7f}])"

    def to_dict(self) -> Dict[str, float]:
        return {
            "min_lng": self.min_lng,
            "min_lat": self.min_lat,
            "max_lng": self.max_lng,
            "max_lat": self.max_lat
        }

    @staticmethod
    def from_dict(data: Dict[str, float]) -> "MBR":
        return MBR(
            min_lng=data["min_lng"],
            min_lat=data["min_lat"],
            max_lng=data["max_lng"],
            max_lat=data["max_lat"]
        )


class RTreeNode:
    """R-Tree Node"""

    def __init__(self, is_leaf: bool = True):
        self.is_leaf = is_leaf
        self.children: List[Union["RTreeNode", RoadNode]] = []
        self.mbr: Optional[MBR] = None
        self.parent: Optional["RTreeNode"] = None
        self.RTreeNode_hash = ""

    def update_mbr(self, recursive: bool = False) -> None:
        """Update MBR to contain all child elements, support recursive update of child nodes"""
        if not self.children:
            self.mbr = None
            return

        # Recursively update child nodes (only needed for internal nodes)
        if not self.is_leaf and recursive:
            for child in self.children:
                if isinstance(child, RTreeNode):
                    child.update_mbr(recursive=True)

        # Calculate current node's MBR
        if self.is_leaf:
            road_nodes = [c for c in self.children if isinstance(c, RoadNode)]
            if not road_nodes:
                self.mbr = None
                return
            first = road_nodes[0]
            current_mbr = MBR(
                first.longitude, first.latitude,
                first.longitude, first.latitude
            )
            for node in road_nodes[1:]:
                current_mbr = current_mbr.expand(node)
        else:
            child_nodes = [c for c in self.children if isinstance(c, RTreeNode) and c.mbr]
            if not child_nodes:
                self.mbr = None
                return
            first = child_nodes[0]
            current_mbr = MBR(
                first.mbr.min_lng, first.mbr.min_lat,
                first.mbr.max_lng, first.mbr.max_lat
            )
            for child in child_nodes[1:]:
                current_mbr = current_mbr.expand(child)

        self.mbr = current_mbr

    def __repr__(self) -> str:
        return f"RTreeNode(is_leaf={self.is_leaf}, mbr={self.mbr}, children_count={len(self.children)})"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_leaf": self.is_leaf,
            "children": [child.to_dict() for child in self.children],
            "mbr": self.mbr.to_dict() if self.mbr else None,
            "RTreeNode_hash": self.RTreeNode_hash
        }

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "RTreeNode":
        node = RTreeNode(is_leaf=data["is_leaf"])
        node.mbr = MBR.from_dict(data["mbr"]) if data["mbr"] else None
        node.RTreeNode_hash = data["RTreeNode_hash"]
        node.children = []
        for child_data in data["children"]:
            if "node_id" in child_data:
                child = RoadNode.from_dict(child_data)
            else:
                child = RTreeNode.from_dict(child_data)
            node.children.append(child)
        return node

    def _rebuild_parent_references(self, parent: "RTreeNode" = None) -> None:
        self.parent = parent
        for child in self.children:
            if isinstance(child, RTreeNode):
                child._rebuild_parent_references(self)

    def post_order_hash(self) -> str:
        if self.is_leaf:
            hashes = [str(self.mbr.min_lng), str(self.mbr.min_lat), str(self.mbr.max_lng), str(self.mbr.max_lat)]
            for child in self.children:
                if isinstance(child, RoadNode):
                    hashes.append(child.compute_combined_hash())
            combined = '+'.join(hashes).encode('utf-8')
            self.RTreeNode_hash = hashlib.sha256(combined).hexdigest()
            return self.RTreeNode_hash
        else:
            hashes = [str(self.mbr.min_lng), str(self.mbr.min_lat), str(self.mbr.max_lng), str(self.mbr.max_lat)]
            for child in self.children:
                if isinstance(child, RTreeNode):
                    child_hash = child.post_order_hash()
                    hashes.append(child_hash)
            combined = '+'.join(hashes).encode('utf-8')
            self.RTreeNode_hash = hashlib.sha256(combined).hexdigest()
            return self.RTreeNode_hash

    def compute_root_hash(self) -> str:
        self.RTreeNode_hash = self.post_order_hash()
        return self.RTreeNode_hash


class RoadNetworkRTree:
    """R-Tree Core Class (integrates all acceleration optimizations)"""

    def __init__(self, max_children: int = 512, min_children: int = 256):
        # Increase node capacity to reduce split times (default values increased from 10/5 to 512/256)
        self.max_children = max_children
        self.min_children = min_children
        self.root: RTreeNode = RTreeNode(is_leaf=True)
        self.total_nodes = 0
        self._delay_mbr_update = False  # Delayed MBR update switch
        self._seed_sample_size = 20  # Seed sampling size for splitting

    def insert(self, road_node: RoadNode) -> None:
        """Single node insertion (supports delayed MBR update)"""
        leaf = self._choose_leaf(self.root, road_node)
        leaf.children.append(road_node)
        self.total_nodes += 1

        # Do not update MBR in real-time in delay mode, only update during splitting
        if not self._delay_mbr_update:
            leaf.update_mbr()

        # Split node when overflow (MBR must be updated during splitting)
        if len(leaf.children) > self.max_children:
            new_node = self._split_node(leaf)
            self._adjust_tree(leaf, new_node)

    def bulk_insert(self, road_nodes: List[RoadNode], batch_size: int = 10000, sort: bool = True) -> None:
        """Bulk insertion (integrates all optimizations)"""
        if not road_nodes:
            return

        # 1. Spatial sorting optimization (Z-order Curve)
        if sort and len(road_nodes) > 100:
            road_nodes = self._sort_by_z_order(road_nodes)

        # 2. Enable delayed update mode
        self._delay_mbr_update = True
        total_batches = (len(road_nodes) + batch_size - 1) // batch_size

        try:
            # 3. Insert in batches
            for batch_idx in tqdm(range(total_batches), desc="Bulk Insert Batches"):
                start = batch_idx * batch_size
                end = min((batch_idx + 1) * batch_size, len(road_nodes))
                batch = road_nodes[start:end]

                for node in batch:
                    self.insert(node)  # Reuse single insertion logic, MBR update delayed at this time

                # Manually update leaf node MBRs after each batch insertion (avoid split logic errors)
                self._update_leaf_mbrs(self.root)

        finally:
            # 4. Disable delay mode and perform full recursive update (ensure final MBR correctness)
            self._delay_mbr_update = False
            self.root.update_mbr(recursive=True)
            # Handle possible top-level node overflow
            self._check_and_split_root()

    def _sort_by_z_order(self, road_nodes: List[RoadNode]) -> List[RoadNode]:
        """Spatially sort road network nodes using Z-order Curve"""

        def z_order_key(node: RoadNode) -> int:
            # Map longitude/latitude to integer range (adapt to 7-decimal precision)
            lng_int = int((node.longitude + 180) * 1e7)  # Range: 0 ~ 3.6e9
            lat_int = int((node.latitude + 90) * 1e7)  # Range: 0 ~ 1.8e9
            return self._interleave_bits(lng_int, lat_int)

        # Sort by Z-order value
        return sorted(road_nodes, key=z_order_key)

    @staticmethod
    def _interleave_bits(x: int, y: int) -> int:
        """Interleave binary bits of x and y to generate Z-order value"""
        result = 0
        # Only process first 32 bits (sufficient to cover range after longitude/latitude mapping)
        for i in range(32):
            result |= ((x >> i) & 1) << (2 * i)
            result |= ((y >> i) & 1) << (2 * i + 1)
        return result

    def _update_leaf_mbrs(self, node: RTreeNode) -> None:
        """Only update MBRs of all leaf nodes (called during intermediate stages of bulk insertion)"""
        if node.is_leaf:
            node.update_mbr()
        else:
            for child in node.children:
                if isinstance(child, RTreeNode):
                    self._update_leaf_mbrs(child)

    def _check_and_split_root(self) -> None:
        """Check if root node needs splitting after full update"""
        if len(self.root.children) > self.max_children:
            new_node = self._split_node(self.root)
            self._adjust_tree(self.root, new_node)

    def _choose_leaf(self, node: RTreeNode, road_node: RoadNode) -> RTreeNode:
        if node.is_leaf:
            return node

        min_cost = float("inf")
        best_child = None
        for child in node.children:
            if not isinstance(child, RTreeNode) or not child.mbr:
                continue
            original_area = child.mbr.area()
            new_area = child.mbr.expand(road_node).area()
            cost = new_area - original_area
            if cost < min_cost:
                min_cost = cost
                best_child = child

        return self._choose_leaf(best_child, road_node) if best_child else node

    def _split_node(self, node: RTreeNode) -> RTreeNode:
        """Split node (optimization: seed selection using random sampling)"""
        seed1, seed2 = self._pick_seeds(node)
        group1, group2 = [seed1], [seed2]
        remaining = [c for c in node.children if c not in (seed1, seed2)]

        while remaining:
            if len(group1) + len(remaining) == self.min_children:
                group1.extend(remaining)
                break
            if len(group2) + len(remaining) == self.min_children:
                group2.extend(remaining)
                break

            # Calculate cost difference (optimization: precompute MBR to reduce repeated calculations)
            mbr1 = self._compute_mbr(group1)
            mbr2 = self._compute_mbr(group2)
            cost_diffs = []

            for child in remaining:
                cost1 = mbr1.expand(child).area() - mbr1.area() if mbr1 else 0
                cost2 = mbr2.expand(child).area() - mbr2.area() if mbr2 else 0
                cost_diffs.append((child, abs(cost1 - cost2)))

            # Allocate element with maximum cost difference
            cost_diffs.sort(key=lambda x: x[1], reverse=True)
            best_child, _ = cost_diffs[0]
            remaining.remove(best_child)

            # Add to group with lower cost
            if mbr1 and mbr2:
                cost1 = mbr1.expand(best_child).area() - mbr1.area()
                cost2 = mbr2.expand(best_child).area() - mbr2.area()
                if cost1 < cost2:
                    group1.append(best_child)
                    mbr1 = mbr1.expand(best_child)  # Reuse MBR to reduce calculations
                else:
                    group2.append(best_child)
                    mbr2 = mbr2.expand(best_child)  # Reuse MBR to reduce calculations
            elif mbr1:
                group1.append(best_child)
                mbr1 = mbr1.expand(best_child)
            else:
                group2.append(best_child)
                mbr2 = mbr2.expand(best_child)

        # Force update MBR after splitting (ensure split node's MBR is correct)
        new_node = RTreeNode(is_leaf=node.is_leaf)
        node.children = group1
        new_node.children = group2
        node.update_mbr()
        new_node.update_mbr()

        return new_node

    def _pick_seeds(self, node: RTreeNode) -> Tuple[Union[RoadNode, RTreeNode], Union[RoadNode, RTreeNode]]:
        """Optimization: seed selection using random sampling instead of full calculation"""
        max_waste = -1
        seeds = (node.children[0], node.children[1])  # Default seeds
        children = node.children
        n = len(children)

        # Randomly sample fixed number of candidate pairs (avoid O(n²) calculation)
        if n > self._seed_sample_size * 2:
            pairs = []
            # Ensure sampling randomness
            sampled_indices = random.sample(range(n), self._seed_sample_size * 2)
            for i in range(self._seed_sample_size):
                a_idx = sampled_indices[2 * i]
                b_idx = sampled_indices[2 * i + 1]
                if a_idx < b_idx:
                    pairs.append((a_idx, b_idx))
                else:
                    pairs.append((b_idx, a_idx))
        else:
            # Still use full calculation when quantity is small
            pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]

        # Calculate separation degree of sampled pairs
        for i, j in pairs:
            a, b = children[i], children[j]
            if node.is_leaf:
                if not (isinstance(a, RoadNode) and isinstance(b, RoadNode)):
                    continue
                a_mbr = MBR(a.longitude, a.latitude, a.longitude, a.latitude)
                b_mbr = MBR(b.longitude, b.latitude, b.longitude, b.latitude)
            else:
                if not (isinstance(a, RTreeNode) and isinstance(b, RTreeNode) and a.mbr and b.mbr):
                    continue
                a_mbr, b_mbr = a.mbr, b.mbr

            union_mbr = a_mbr.expand(b_mbr)
            waste = union_mbr.area() - (a_mbr.area() + b_mbr.area())
            if waste > max_waste:
                max_waste = waste
                seeds = (a, b)
        return seeds

    def _compute_mbr(self, elements: List[Union[RoadNode, RTreeNode]]) -> Optional[MBR]:
        if not elements:
            return None
        if isinstance(elements[0], RoadNode):
            valid = [e for e in elements if isinstance(e, RoadNode)]
            if not valid:
                return None
            first = valid[0]
            mbr = MBR(first.longitude, first.latitude, first.longitude, first.latitude)
            for e in valid[1:]:
                mbr = mbr.expand(e)
            return mbr
        else:
            valid = [e for e in elements if isinstance(e, RTreeNode) and e.mbr]
            if not valid:
                return None
            first = valid[0]
            mbr = MBR(first.mbr.min_lng, first.mbr.min_lat, first.mbr.max_lng, first.mbr.max_lat)
            for e in valid[1:]:
                mbr = mbr.expand(e)
            return mbr

    def _adjust_tree(self, node: RTreeNode, new_node: RTreeNode) -> None:
        """Optimization: prioritize splitting to reduce repeated MBR calculations"""
        current = node.parent

        if current is None:
            # Root node split: create new root
            new_root = RTreeNode(is_leaf=False)
            new_root.children.extend([node, new_node])
            node.parent = new_root
            new_node.parent = new_root
            new_root.update_mbr()  # New root MBR based on two child nodes
            self.root = new_root
            current = new_root  # New root has no parent, no need to propagate further
        else:
            # Non-root node: first add new node to parent node
            current.children.append(new_node)
            new_node.parent = current
            current.update_mbr()  # First update current parent node's MBR

            # Core optimization: check if parent node overflows first, handle split immediately
            if len(current.children) > self.max_children:
                # Split parent node to get new node
                grandchild = self._split_node(current)
                # Recursively adjust upper tree structure (handles splitting and MBR update of higher layers at this time)
                self._adjust_tree(current, grandchild)
            else:
                # Parent node not overflowing, only propagate update to ancestor MBRs
                ancestor = current.parent
                while ancestor is not None and not self._delay_mbr_update:
                    ancestor.update_mbr()
                    ancestor = ancestor.parent

    # Query methods (keep original logic completely)
    def point_query(self, longitude: float, latitude: float) -> Optional[RoadNode]:
        target = RoadNode(-1, longitude, latitude)
        return self._point_query_recursive(self.root, target)

    def _point_query_recursive(self, node: RTreeNode, target: RoadNode) -> Optional[RoadNode]:
        if not node.mbr or not node.mbr.contains(target):
            return None

        if node.is_leaf:
            for child in node.children:
                if isinstance(child, RoadNode) and child.equals(target):
                    return child

        for child in node.children:
            if isinstance(child, RTreeNode):
                result = self._point_query_recursive(child, target)
                if result:
                    return result

        return None

    def range_query(self, query_mbr: MBR, start_time: int, end_time: int, trajs: set[str]) -> List[RoadNode]:
        result = []
        csv_vo_data = []
        r_path = []
        self._range_query_recursive(self.root, query_mbr, start_time, end_time, result, trajs, csv_vo_data, r_path)
        with open('r_vo.csv', 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerows(csv_vo_data)
        return result

    def _range_query_recursive(self, node: RTreeNode, query_mbr: MBR, start_time: int, end_time: int,
                               result: List[RoadNode], trajs: set[str], csv_vo_data, r_path) -> None:
        r_path.append(node)
        if not node.mbr or not node.mbr.intersects(query_mbr):
            a = list(r_path)
            ver = vo_hash_collect_rnode_mbr_not(a)
            data = [str(node.mbr.min_lng), str(node.mbr.min_lat), str(node.mbr.max_lng), str(node.mbr.max_lat)]
            csv_vo_data.append(["lng_lat_vo_start"])
            csv_vo_data.append([ver, data])
            csv_vo_data.append(["lng_lat_vo_end"])
            r_path.pop()
            return

        if node.is_leaf:
            a = list(r_path)
            ver = vo_hash_collect_rnode_mbr_yes(a)
            csv_vo_data.append(["lng_lat_vo_start"])
            csv_vo_data.append([ver])
            for child in node.children:
                csv_vo_data.append(["road_node_start"])
                if isinstance(child, RoadNode) and query_mbr.contains(child):
                    node_collect_list = [str(child.node_id), str(child.longitude), str(child.latitude)]
                    for traj in child.trajectories:
                        node_collect_list.append(" ")
                    csv_vo_data.append([node_collect_list])
                    for traj in child.trajectories:
                        trajlist = []
                        if traj.start_ts <= end_time and traj.end_ts >= start_time:
                            trajs.add(traj.trajectory_str)
                            trajlist.append(traj.trajectory_str)
                            traj_vo = [" ", str(traj.start_ts), str(traj.end_ts)]
                            csv_vo_data.append([[traj_vo, trajlist]])
                        else:
                            timelist = [str(traj.start_ts), str(traj.end_ts)]
                            traj_vo = [traj.trajectory_str, " ", " "]
                            csv_vo_data.append([[traj_vo, timelist]])

                elif isinstance(child, RoadNode) and not query_mbr.contains(child):
                    node_collect_list = [str(child.node_id), " ", " "]
                    data = [str(child.longitude), str(child.latitude)]
                    for traj in child.trajectories:
                        node_collect_list.append(traj.trajectory_hash)
                    csv_vo_data.append([node_collect_list, data])
                csv_vo_data.append(["road_node_end"])
            csv_vo_data.append(["lng_lat_vo_end"])
            r_path.pop()
        else:
            for child in node.children:
                if isinstance(child, RTreeNode):
                    self._range_query_recursive(child, query_mbr, start_time, end_time, result, trajs, csv_vo_data, r_path)
            r_path.pop()

    def __len__(self) -> int:
        return self.total_nodes

    def save(self, file_path: str) -> None:
        data = {
            "max_children": self.max_children,
            "min_children": self.min_children,
            "total_nodes": self.total_nodes,
            "root": self.root.to_dict()
        }
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    @staticmethod
    def load(file_path: str) -> "RoadNetworkRTree":
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        rtree = RoadNetworkRTree(
            max_children=data["max_children"],
            min_children=data["min_children"]
        )
        rtree.total_nodes = data["total_nodes"]
        rtree.root = RTreeNode.from_dict(data["root"])
        rtree.root._rebuild_parent_references()
        return rtree


def traj_timestamp_insert(traj, rtree):
    traj_str = '->'.join(str(item) for item in traj[0])
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
            lat = round(j[1], 7)
            lng = round(j[2], 7)
            time_point = j[0]
            target_node = rtree.point_query(lng, lat)
            trajectory = Trajectory(traj_hash, time_point, time_point)
            trajectory.trajectory_hash = hashlib.sha256(
                f"{trajectory.trajectory_str}{trajectory.start_ts}{trajectory.end_ts}".encode('utf-8')
            ).hexdigest()
            if target_node:
                target_node.add_trajectory(trajectory)


def insert_time_stamp(rtree):
    for i in range(1, 2):
        traj_name_path =  f"C:\\Users\\maoyusen\\Desktop\\Graph-Diffusion-Planning-main\\loader\\preprocess\\mm\\sets_data\\real2\\trajectories\\traj_mapped_xian_xian10-{i}.json"
        with open(traj_name_path, 'r', encoding='utf-8') as file:
            nested_list = json.load(file)
        j = 0
        for traj in nested_list:
            j = j + 1

            traj_timestamp_insert(traj, rtree)
            print(f"Trajectory {j} of 10-{i} completed")


def read_road_nodes_from_csv(file_path: str) -> List[RoadNode]:
    df = pd.read_csv(file_path)
    required_columns = ['id', 'longitude', 'latitude']
    if not set(required_columns).issubset(df.columns):
        raise ValueError(f"CSV file must contain columns: {required_columns}")
    nodes = []
    total_rows = len(df)
    for _, row in tqdm(df.iterrows(), total=total_rows, desc="Reading and converting node data"):
        node = RoadNode(
            node_id=int(row['id']),
            longitude=float(row['longitude']),
            latitude=float(row['latitude'])
        )
        nodes.append(node)
    return nodes


def vo_hash_collect_rnode_mbr_not(r_path):
    stack = []
    a = []
    flag = r_path.pop()
    a.append(" ")
    a.append(" ")
    a.append(" ")
    a.append(" ")

    if flag.is_leaf:
        for road_node in flag.children:
            a.append(road_node.RoadNode_hash)
    else:
        for rtree_node in flag.children:
            a.append(rtree_node.RTreeNode_hash)
    stack.append(a)

    while r_path:
        b = []
        flag1 = r_path.pop()
        b.append(str(flag1.mbr.min_lng))
        b.append(str(flag1.mbr.min_lat))
        b.append(str(flag1.mbr.max_lng))
        b.append(str(flag1.mbr.max_lat))

        for child in flag1.children:
            if child == flag:
                b.append(" ")
            else:
                b.append(child.RTreeNode_hash)

        stack.append(b)
        flag = flag1
    return stack


def vo_hash_collect_rnode_mbr_yes(r_path):
    stack = []
    a = []
    flag = r_path.pop()

    if flag.is_leaf:
        a.append(str(flag.mbr.min_lng))
        a.append(str(flag.mbr.min_lat))
        a.append(str(flag.mbr.max_lng))
        a.append(str(flag.mbr.max_lat))
        for road_node in flag.children:
            a.append(" ")
    stack.append(a)

    while r_path:
        b = []
        flag1 = r_path.pop()
        b.append(str(flag1.mbr.min_lng))
        b.append(str(flag1.mbr.min_lat))
        b.append(str(flag1.mbr.max_lng))
        b.append(str(flag1.mbr.max_lat))

        for child in flag1.children:
            if child == flag:
                b.append(" ")
            else:
                b.append(child.RTreeNode_hash)

        stack.append(b)
        flag = flag1
    return stack


def verify(filename):
    file_path = filename
    with open(file_path, 'r', encoding='utf-8') as file:
        flag1 = []
        flag2 = []
        list1 = []
        list2 = []
        reader = csv.reader(file)
        for row in reader:
            if row[0] == "lng_lat_vo_start":
                flag1 = next(reader)
                if len(flag1) <= 1:
                    flag1 = ast.literal_eval(flag1[0])
                    flag1.reverse()
                    list1 = []
                else:
                    c = flag1[0]
                    b = flag1[1]
                    c = ast.literal_eval(c)
                    c.reverse()
                    a = c[-1]
                    b = ast.literal_eval(b)
                    if isinstance(b[0], list):
                        b = b[0]
                    j = 0
                    for i in range(len(a)):
                        if a[i] == " ":
                            a[i] = str(b[j])
                            j = j + 1
                            if j > 3:
                                break
                    all_hash_join = '+'.join(a).encode('utf-8')
                    hash_hex = hashlib.sha256(all_hash_join).hexdigest()

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
                  #  print(hash_hex,111111111111111)
                    if hash_hex != loaded_rtree.root.RTreeNode_hash:
                        return False

            elif row[0] == "road_node_start":
                flag2 = next(reader)
                if len(flag2) <= 1:
                    flag2 = ast.literal_eval(flag2[0])
                    list2 = []
                else:
                    a = flag2[0]
                    b = flag2[1]
                    a = ast.literal_eval(a)
                    b = ast.literal_eval(b)
                    j = 1
                    for i in range(2):
                        a[j] = b[i]
                        j = j + 1
                    all_hash_join = '+'.join(item for item in a if item)
                    encoded_data = all_hash_join.encode('utf-8')
                    hash_object = hashlib.sha256(encoded_data)
                    hash_hex = hash_object.hexdigest()
                    list1.append(hash_hex)
                    row = next(reader)

            elif row[0] == "road_node_end":
                a = flag2
                j = 0
                if list2:
                    for i in range(len(a)):
                        if a[i] == " ":
                            a[i] = list2[j]
                            j = j + 1
                else:
                    a[-1] = ""
                all_hash_join = '+'.join(item for item in a if item)
                encoded_data = all_hash_join.encode('utf-8')
                hash_object = hashlib.sha256(encoded_data)
                hash_hex = hash_object.hexdigest()
                list1.append(hash_hex)

            elif row[0] == "lng_lat_vo_end":

                t = flag1.pop()
                j = 0
                for i in range(len(t)):
                    if t[i] == " ":
                        t[i] = list1[j]
                        j = j + 1
                all_hash_join = '+'.join(item for item in t)
                encoded_data = all_hash_join.encode('utf-8')
                hash_object = hashlib.sha256(encoded_data)
                hash_hex = hash_object.hexdigest()

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
             #   print(hash_hex,222222222222222222)
                if hash_hex != loaded_rtree.root.RTreeNode_hash:
                    return False

            else:
                a = row[0]
                a = ast.literal_eval(a)
                tianchong = a[0]
                daan = a[1]

                j = 0
                for i in range(len(tianchong)):
                    if tianchong[i] == " ":
                        tianchong[i] = daan[j]
                        j = j + 1

                all_traj_str = ''.join(item for item in tianchong)
                encoded_data = all_traj_str.encode('utf-8')
                hash_object = hashlib.sha256(encoded_data)
                hash_hex = hash_object.hexdigest()
                list2.append(hash_hex)
    return True


def insert_traj_to_r(traj,rtree):
    traj_start_time = traj[1][0][-1]
    traj_end_time = traj[1][-1][-1]
    traj_start_time_str = str(traj_start_time)
    traj_end_time_str = str(traj_end_time)
    str_time_traj = traj[0] + traj_start_time_str + traj_end_time_str
    encoded_data = str_time_traj.encode('utf-8')
    hash_object = hashlib.sha256(encoded_data)
    traj_hash = hash_object.hexdigest()

    for point in traj[1]:
       lng = round(point[1],7)
       lat = round(point[2],7)
       time_point = point[0]
       target_node = rtree.point_query(lng, lat)
       trajectory = Trajectory(traj_hash, time_point, time_point)
       trajectory.trajectory_hash = hashlib.sha256(
           f"{trajectory.trajectory_str}{trajectory.start_ts}{trajectory.end_ts}".encode('utf-8')
       ).hexdigest()
       if target_node:
           target_node.add_trajectory(trajectory)


def insert_beijing_to_r_beijing(file_path,rtree):
    with open(file_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        next(reader)  # Skip header
        i = 0
        for row in reader:
            traj = []  # Initialize traj as empty at the start of each loop
            i += 1
            if len(row) < 2:
                print(f"Row {i}: Incomplete data, skipped")
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
                continue  # traj remains empty list at this time, no additional reset needed

            # Process current trajectory
            insert_traj_to_r(traj,rtree)
            print(f"Row {i} processed successfully")  # Fix print format




if __name__ == "__main__":
    # csv.field_size_limit(10 * 1024 * 1024)  # 10MB
   #  # #Initialize R-Tree (use optimized node capacity)
   #  time1 = time.time()
   #  rtree = RoadNetworkRTree(max_children=128, min_children=64)
   #  csv_file_path = "Xi'an_one_day_coordinates.csv"  # Replace with actual CSV path
   #
   #  # Read road network nodes
   #  road_nodes = read_road_nodes_from_csv(csv_file_path)
   #
   #  # Bulk insert (enable all optimizations)
   #  rtree.bulk_insert(road_nodes, batch_size=20000)  # Larger batch size reduces loop overhead
   #  #
   #  # # Insert trajectories (function unchanged)
   #  insert_time_stamp(rtree)
   # # process_all_files(rtree)
   #  rtree.root.compute_root_hash()
   #  time2 = time.time()
   #  print(time2 - time1)
    # rtree.save("rtree_beijing_3_days.json")
    loaded_rtree = RoadNetworkRTree.load("rtree_beijing_3_days.json")
    print(f"Loaded R-Tree contains {len(loaded_rtree)} nodes")
    #
    query_mbr1 = MBR(
        min_lng= 116.3,   # Central longitude 104.1, ±0.02
        min_lat=39.96,  # Central latitude 30.7, ±0.02
        max_lng= 116.32,
        max_lat=40.98
    )

    start_time1 = 1225779455
    end_time1 =  1225779995

    trajs = set()
    time1 = time.time()
    range_result = loaded_rtree.range_query(query_mbr1, start_time1, end_time1, trajs)
    time2 = time.time()
    query_time = time2 - time1

    time1 = time.time()
    verify("r_vo.csv")
    time2 = time.time()
    vo_time = time2 - time1


    print(query_time)
    print(vo_time)
    print(len(trajs))
