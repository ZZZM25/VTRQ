import hashlib
import csv
import json
import time
from collections import deque  # 优化队列操作，提升大规模数据处理效率
from typing import List, Optional, Union, Tuple, Dict, Any, Iterable

from filter_traj_id import filter_id, final_filter_id
#from 处理北京轨迹点文件 import process_all_files, process_all_files2


def hash_value(data: Union[str, bytes, float]) -> str:
    """计算SHA-256哈希值，统一使用6位小数字符串处理经度（解决浮点数精度问题）"""
    if isinstance(data, float):
        # 经度强制转为6位小数字符串，彻底避免浮点数存储误差
        data = f"{round(data, 6):.6f}"
    if isinstance(data, str):
        data = data.encode('utf-8')
    return hashlib.sha256(data).hexdigest()


class MerkleBTreeNode:
    """默克尔B树节点：使用字符串存储经度，修复浮点数精度导致的验证失败"""

    def __init__(self, is_leaf: bool = False):
        self.keys: List[str] = []  # 存储6位小数字符串（如"116.405290"）
        self.values: List[List[str]] = []  # 轨迹哈希列表
        self.children: List[MerkleBTreeNode] = []  # 子节点列表
        self.is_leaf: bool = is_leaf  # 是否为叶子节点
        self.hash: str = ""  # 节点哈希
        self.parent: Optional[MerkleBTreeNode] = None  # 父节点引用

    def update_hash(self) -> None:
        """更新节点哈希，确保递归更新时子节点哈希已优先计算"""
        if self.is_leaf:
            entries = [f"{lon_str}:{','.join(hashes)}" for lon_str, hashes in zip(self.keys, self.values)]
            self.hash = hash_value("|".join(entries))
        else:
            child_hashes = "|".join([child.hash for child in self.children])
            lon_str = "|".join(self.keys)  # 直接使用字符串拼接
            self.hash = hash_value(f"{child_hashes}|{lon_str}")

        # 父节点存在则触发更新（确保哈希链完整）
        if self.parent:
            self.parent.update_hash()

    def find_lon_index(self, longitude_str: str) -> int:
        """通过字符串精确匹配查找经度索引（彻底解决浮点数误差）"""
        for i, lon_str in enumerate(self.keys):
            if lon_str == longitude_str:
                return i
        return -1

    def to_dict(self) -> Dict[str, Any]:
        """序列化节点为字典（保持字符串经度）"""
        node_dict = {
            "is_leaf": self.is_leaf,
            "keys": self.keys,  # 直接存储字符串，无需转换
            "values": self.values,
            "hash": self.hash
        }
        if not self.is_leaf:
            node_dict["children"] = [child.to_dict() for child in self.children]
        return node_dict

    @classmethod
    def from_dict(cls, node_dict: Dict[str, Any], parent: Optional["MerkleBTreeNode"] = None) -> "MerkleBTreeNode":
        """从字典反序列化节点（恢复字符串经度）"""
        node = cls(is_leaf=node_dict["is_leaf"])
        node.keys = node_dict["keys"]  # 直接读取字符串经度
        node.values = node_dict["values"]
        node.hash = node_dict["hash"]
        node.parent = parent

        if not node.is_leaf and "children" in node_dict:
            node.children = [cls.from_dict(child_dict, parent=node) for child_dict in node_dict["children"]]
        return node

    def __str__(self) -> str:
        return f"Node(lons={self.keys[:3]}..., hash={self.hash[:8]}..., is_leaf={self.is_leaf})"


class LocationMerkleBTree:
    """修复验证失败问题的默克尔B树（支持大规模数据处理）"""

    def __init__(self, order: int = 4):
        self.root: MerkleBTreeNode = MerkleBTreeNode(is_leaf=True)
        self.order: int = order  # 阶数：建议50万节点时设为30~50

    def insert_trace(self, longitude: float, trace_hash: str) -> None:
        """插入轨迹哈希（使用字符串经度避免重复）"""
        # 转为6位小数字符串作为唯一标识（核心修复点）
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
                # 经度已存在，轨迹哈希去重后添加
                if trace_hash not in node.values[lon_index]:
                    node.values[lon_index].append(trace_hash)
                    node.update_hash()  # 立即更新哈希
            else:
                # 插入新经度（按字符串排序）
                while i >= 0 and longitude_str < node.keys[i]:
                    i -= 1
                node.keys.insert(i + 1, longitude_str)
                node.values.insert(i + 1, [trace_hash])
                node.update_hash()  # 立即更新哈希
        else:
            # 内部节点查找子节点
            while i >= 0 and longitude_str < node.keys[i]:
                i -= 1
            i += 1

            # 子节点满则分裂
            if len(node.children[i].keys) == self.order - 1:
                self._split_child(node, i)
                if longitude_str > node.keys[i]:
                    i += 1

            # 递归插入子节点
            self._insert_non_full(node.children[i], longitude_str, trace_hash)

    def _split_child(self, parent: MerkleBTreeNode, child_idx: int) -> None:
        """分裂子节点（修复哈希更新顺序，确保子节点先更新）"""
        order = self.order
        child = parent.children[child_idx]
        new_node = MerkleBTreeNode(is_leaf=child.is_leaf)
        new_node.parent = parent

        mid_idx = (order - 1) // 2
        mid_key = child.keys[mid_idx]

        # 分割数据
        new_node.keys = child.keys[mid_idx + 1:]
        new_node.values = child.values[mid_idx + 1:]
        child.keys = child.keys[:mid_idx]
        child.values = child.values[:mid_idx]

        # 分割子节点（非叶子节点）
        if not child.is_leaf:
            new_node.children = child.children[mid_idx + 1:]
            for c in new_node.children:
                c.parent = new_node
                c.update_hash()  # 先更新新节点的子节点哈希
            child.children = child.children[:mid_idx + 1]
            for c in child.children:
                c.update_hash()  # 再更新原节点的子节点哈希

        # 关键修复：先更新子节点和新节点的哈希（确保完成）
        child.update_hash()
        new_node.update_hash()

        # 再插入父节点并更新父节点哈希
        parent.children.insert(child_idx + 1, new_node)
        parent.keys.insert(child_idx, mid_key)
        parent.update_hash()

    def query_and_extract(self, longitude: float) -> Tuple[bool, Optional[List[str]], Optional[Dict]]:
        """查询单个经度的轨迹及验证路径（使用字符串经度匹配）"""
        target_lon_str = f"{round(longitude, 6):.6f}"
        current = self.root

        while True:
            lon_index = current.find_lon_index(target_lon_str)
            if lon_index != -1 and current.is_leaf:
                # 提取轨迹哈希列表
                trace_hashes = current.values[lon_index].copy()
                # 构建验证路径
                validation_path = {
                    "longitude": float(target_lon_str),  # 存储为浮点数便于显示
                    "longitude_str": target_lon_str,  # 存储字符串用于精确验证
                    "leaf_entries": [f"{lon_str}:{','.join(hashes)}" for lon_str, hashes in
                                     zip(current.keys, current.values)],
                    "leaf_hash": current.hash,
                    "lon_index": lon_index,
                    "path": [],
                    "root_hash": self.root.hash
                }

                # 收集从叶子到根的路径
                leaf_node = current
                while leaf_node.parent is not None:
                    parent = leaf_node.parent
                    child_idx = parent.children.index(leaf_node)
                    sibling_hashes = [child.hash for i, child in enumerate(parent.children) if i != child_idx]
                    validation_path["path"].append({
                        "child_idx": child_idx,
                        "sibling_hashes": sibling_hashes.copy(),
                        "parent_keys": parent.keys.copy()  # 存储字符串经度
                    })
                    leaf_node = parent

                return (True, trace_hashes, validation_path)
            elif current.is_leaf:
                return (False, None, None)
            else:
                # 内部节点导航
                i = 0
                while i < len(current.keys) and target_lon_str > current.keys[i]:
                    i += 1
                current = current.children[i]

    def query_by_range(self, min_lon: float, max_lon: float) -> List[Tuple[float, List[str], Dict]]:
        """范围查询（使用字符串比较确保精度）"""
        min_lon_str = f"{round(min_lon, 6):.6f}"
        max_lon_str = f"{round(max_lon, 6):.6f}"
        result = []

        # 使用deque优化队列操作（大规模数据处理关键）
        queue = deque([self.root])
        while queue:
            node = queue.popleft()
            if node.is_leaf:
                for lon_str in node.keys:
                    # 字符串比较确保范围判断准确
                    if min_lon_str <= lon_str <= max_lon_str:
                        exists, traces, path = self.query_and_extract(float(lon_str))
                        if exists and traces and path:
                            result.append((float(lon_str), traces, path))
            else:
                queue.extend(node.children)

        return sorted(result, key=lambda x: x[0])

    def save_range_query_results_to_csv(self, range_results: List[Tuple[float, List[str], Dict]],
                                        csv_path: str) -> None:
        """保存范围查询结果到CSV"""
        if not range_results:
            print("⚠️  范围查询结果为空，不保存CSV")
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

        print(f"✅ 范围查询结果已保存到CSV：{csv_path}")

    @staticmethod
    def independent_verify(trace_hashes: List[str], validation_path: Dict) -> bool:
        """独立验证轨迹真实性（使用字符串经度确保匹配）"""
        if not validation_path:
            return False

        # 核心修复：使用字符串经度进行精确匹配
        lon_str = validation_path["longitude_str"]
        leaf_entries = validation_path["leaf_entries"]
        lon_index = validation_path["lon_index"]
        leaf_hash = validation_path["leaf_hash"]

        # 验证索引合法性
        if lon_index < 0 or lon_index >= len(leaf_entries):
            return False
        # 验证轨迹列表与叶子条目一致性
        expected_trace_str = ",".join(trace_hashes)
        expected_entry = f"{lon_str}:{expected_trace_str}"
        if leaf_entries[lon_index] != expected_entry:
            return False
        # 验证叶子节点哈希
        calculated_leaf_hash = hash_value("|".join(leaf_entries))
        if calculated_leaf_hash != leaf_hash:
            return False

        # 验证路径到根节点
        current_hash = leaf_hash
        for step in validation_path["path"]:
            child_idx = step["child_idx"]
            sibling_hashes = step["sibling_hashes"].copy()
            parent_keys = step["parent_keys"]  # 字符串经度列表

            sibling_hashes.insert(child_idx, current_hash)
            child_hashes_str = "|".join(sibling_hashes)
            parent_keys_str = "|".join(parent_keys)  # 直接拼接字符串
            current_hash = hash_value(f"{child_hashes_str}|{parent_keys_str}")

        # 最终验证根哈希
        return current_hash == validation_path["root_hash"]

    @staticmethod
    def batch_verify(verify_items: Iterable[Tuple[List[str], Dict]]) -> List[Tuple[float, bool]]:
        """批量验证轨迹列表"""
        results = []
        for traces, path in verify_items:
            if not path:
                continue
            longitude = path.get("longitude", 0.0)
            is_valid = LocationMerkleBTree.independent_verify(traces, path)
            results.append((longitude, is_valid))
        return results

    def save_all_validation_paths_to_csv(self, csv_path: str) -> None:
        """保存所有轨迹的验证路径到CSV（使用字符串去重）"""
        all_traces = self._collect_all_traces()
        if not all_traces:
            print("⚠️  无轨迹数据可保存到CSV")
            return

        with open(csv_path, 'w', encoding='utf-8', newline='') as f:
            fieldnames = [
                "longitude", "trace_hash", "leaf_hash", "root_hash",
                "leaf_entries", "validation_path", "verify_result"
            ]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            # 用字符串经度去重（核心修复点）
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

        print(f"✅ 所有轨迹验证路径已保存到：{csv_path}")

    def save_to_json(self, json_path: str) -> None:
        """保存树到JSON"""
        tree_data = {
            "order": self.order,
            "root": self.root.to_dict(),
            "root_hash": self.root.hash
        }

        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(tree_data, f, ensure_ascii=False, indent=2)
        print(f"✅ 树已保存到JSON文件：{json_path}")

    @classmethod
    def load_from_json(cls, json_path: str) -> "LocationMerkleBTree":
        """从JSON加载树并验证完整性"""
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                tree_data = json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"❌ 未找到JSON文件：{json_path}")
        except json.JSONDecodeError:
            raise ValueError(f"❌ JSON文件格式错误：{json_path}")

        required_fields = ["order", "root", "root_hash"]
        if not all(field in tree_data for field in required_fields):
            raise ValueError(f"❌ JSON文件缺少必要字段：{required_fields}")

        tree = cls(order=tree_data["order"])
        tree.root = MerkleBTreeNode.from_dict(tree_data["root"])

        # 验证根哈希
        if tree.root.hash != tree_data["root_hash"]:
            raise ValueError(f"❌ 树哈希校验失败，数据可能已损坏")
        # 验证所有节点哈希
        if not tree._verify_all_node_hashes(tree.root):
            raise ValueError(f"❌ 节点哈希不一致，数据可能已篡改")

        print(f"✅ 树已从JSON文件加载：{json_path}")
        return tree

    def _verify_all_node_hashes(self, node: MerkleBTreeNode) -> bool:
        """递归验证所有节点哈希"""
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
        """收集所有轨迹（返回字符串经度+哈希）"""
        all_traces = []
        queue = deque([self.root])  # 用deque优化大规模遍历
        while queue:
            node = queue.popleft()
            if node.is_leaf:
                for lon_str, trace_hashes in zip(node.keys, node.values):
                    for trace in trace_hashes:
                        all_traces.append((lon_str, trace))  # 存储字符串经度
            else:
                queue.extend(node.children)
        return all_traces

    def get_root_hash(self) -> str:
        """获取根节点哈希"""
        return self.root.hash

    def print_tree(self, node: Optional[MerkleBTreeNode] = None, level: int = 0) -> None:
        """打印树结构（调试用）"""
        if node is None:
            node = self.root
        print("  " * level + f"Level {level}: {node}")
        if not node.is_leaf:
            for child in node.children:
                self.print_tree(child, level + 1)


def traj_timestamp_insert(traj,tree):
    traj_str = '->'.join(str(item) for item in traj[0])
    # 找到轨迹总开始和结束的时间
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
            rounded_num = round(j[1], 7) #经度
            tree.insert_trace(rounded_num,traj_hash)



def insert_traj(tree):
    for i in range(1,4):
        traj_name_path = f"C:\\Users\\maoyusen\\Desktop\\Graph-Diffusion-Planning-main\\chengdu-tra-json\\traj-10-{i}.json"
        # 打开 JSON 文件
        with open(traj_name_path, 'r', encoding='utf-8') as file:
            # 解析 JSON 数据
            nested_list = json.load(file)
        j = 0
        for traj in nested_list:
            j = j + 1
            #if j < 1500:
            traj_timestamp_insert(traj,tree)
            print(f"10-{i}的第{j}条轨迹完成")


def get_trajectory_intersection(jingdu_results, weidu_results, shijian_results):
    """
    计算三个树查询结果中轨迹的交集（基于轨迹哈希）
    :param jingdu_results: 经度树查询结果
    :param weidu_results: 纬度树查询结果
    :param shijian_results: 时间树查询结果
    :return: 交集轨迹哈希集合
    """
    # 提取每个树查询结果中的所有轨迹哈希（去重）
    jingdu_hashes = set()
    for _, traces, _ in jingdu_results:
        jingdu_hashes.update(traces)

    weidu_hashes = set()
    for _, traces, _ in weidu_results:
        weidu_hashes.update(traces)

    shijian_hashes = set()
    for _, traces, _ in shijian_results:
        shijian_hashes.update(traces)

    # 计算三者的交集
    intersection_hashes = jingdu_hashes & weidu_hashes & shijian_hashes
    return intersection_hashes


def get_trajectory_intersection(jingdu_results, weidu_results, shijian_results):
    """
    计算三个树查询结果中轨迹的交集（基于轨迹哈希）
    :param jingdu_results: 经度树查询结果
    :param weidu_results: 纬度树查询结果
    :param shijian_results: 时间树查询结果
    :return: 交集轨迹哈希集合
    """
    # 提取每个树查询结果中的所有轨迹哈希（去重）
    jingdu_hashes = set()
    for _, traces, _ in jingdu_results:
        jingdu_hashes.update(traces)

    weidu_hashes = set()
    for _, traces, _ in weidu_results:
        weidu_hashes.update(traces)

    shijian_hashes = set()
    for _, traces, _ in shijian_results:
        shijian_hashes.update(traces)

    # 计算三者的交集
    intersection_hashes = jingdu_hashes & weidu_hashes & shijian_hashes
    return intersection_hashes


if __name__ == "__main__":
    time1 = time.time()
    tree = LocationMerkleBTree(order=30)
    process_all_files2(tree)#时间   经度   纬度

    print("插入完成，原始树结构：")
    original_root_hash = tree.get_root_hash()
    time2 = time.time()
    print(time2-time1)
    # 2. 保存树到JSON
    json_path = "mbt_时间_beijing_3days.json"
    tree.save_to_json(json_path)
    # 加载三个树（假设已加载完成）
    loaded_tree_jingdu = LocationMerkleBTree.load_from_json("mbt_经度_beijing_3days.json")
    loaded_tree_weidu = LocationMerkleBTree.load_from_json("mbt_纬度_beijing_3days.json")
    loaded_tree_shijian = LocationMerkleBTree.load_from_json("mbt_时间_beijing_3days.json")

    # 执行各自的范围查询
    # 经度范围查询
    min_jing, max_jing = 116.30, 116.33
    jingdu_results = loaded_tree_jingdu.query_by_range(min_jing, max_jing)

    # 纬度范围查询
    min_wei, max_wei = 39.97, 40
    weidu_results = loaded_tree_weidu.query_by_range(min_wei, max_wei)

    # 时间范围查询（注意：时间树存储的是时间戳，这里用示例范围）
    min_time, max_time =  1233955024, 1233958624
    shijian_results = loaded_tree_shijian.query_by_range(min_time, max_time)

    # 计算轨迹交集
    intersection = get_trajectory_intersection(jingdu_results, weidu_results, shijian_results)

    # 输出结果
    print(f"经度树查询到的轨迹数量：{len(set(t for _, traces, _ in jingdu_results for t in traces))}")
    print(f"纬度树查询到的轨迹数量：{len(set(t for _, traces, _ in weidu_results for t in traces))}")
    print(f"时间树查询到的轨迹数量：{len(set(t for _, traces, _ in shijian_results for t in traces))}")
    print(f"三个树查询结果的轨迹交集数量：{len(intersection)}")
    filter_id(intersection, "北京轨迹汇总结果.csv")
    time_all = final_filter_id("filter_beijing.csv", 116.30, 116.33, 39.97, 40,1233955024, 1233958624	 )
    if intersection:
        print("交集轨迹哈希示例：", list(intersection)[:5])  # 打印前5个示例
#if __name__ == "__main__":
    # time1 = time.time()
    # tree = LocationMerkleBTree(order=30)
    # #process_all_files2(tree)#时间   经度   纬度
    # insert_traj(tree)
    # print("插入完成，原始树结构：")
    # original_root_hash = tree.get_root_hash()
    # time2 = time.time()
    # print(time2-time1)
    # # 2. 保存树到JSON
    # json_path = "mbt_纬度_成都_1days.json"
    # tree.save_to_json(json_path)

    #
    # loaded_tree_weidu = LocationMerkleBTree.load_from_json("mbt_纬度_beijing_3days.json")
    # loaded_tree_jingdu = LocationMerkleBTree.load_from_json("mbt_经度_beijing_3days.json")
    # loaded_tree_shijian = LocationMerkleBTree.load_from_json("mbt_时间_beijing_3days.json")
    # #
    # #
    #
    #
    # # # 4. 加载的树执行范围查询并保存CSV
    # #
    # min_lon, max_lon =  116.31, 116.34
    # time1 = time.time()
    # range_results = loaded_tree_jingdu.query_by_range(min_lon, max_lon)
    # time2 = time.time()
    # query_time_jingdu = time2 - time1
    # csv_path = "vo_jingdu.csv"
    # loaded_tree_jingdu.save_range_query_results_to_csv(range_results, csv_path)
    # print(f"\n查询结果包含 {len(range_results)} 个经度点：")
    # for lon, traces, _ in range_results:
    #     print(f"经度 {lon}：{len(traces)} 条轨迹")
    #
    # time1 = time.time()
    # # 5. 验证加载树的批量验证功能
    # print("\n=== 5. 批量验证查询结果 ===")
    # verify_items = [(traces, path) for _, traces, path in range_results]
    # verify_results = loaded_tree_jingdu.batch_verify(verify_items)
    # for lon, is_valid in verify_results:
    #     print(f"经度 {lon} 验证结果：{'通过' if is_valid else '失败'}")
    # time2 = time.time()
    # vo_time_jingdu = time2 - time1
    #
    # min_lon, max_lon =  40.0, 40.03
    # time1 = time.time()
    # range_results = loaded_tree_weidu.query_by_range(min_lon, max_lon)
    # time2 = time.time()
    # query_time_weidu = time2 - time1
    # csv_path = "vo_weidu.csv"
    # loaded_tree_weidu.save_range_query_results_to_csv(range_results, csv_path)
    # print(f"\n查询结果包含 {len(range_results)} 个经度点：")
    # for lon, traces, _ in range_results:
    #     print(f"经度 {lon}：{len(traces)} 条轨迹")
    #
    # time1 = time.time()
    # # 5. 验证加载树的批量验证功能
    # print("\n=== 5. 批量验证查询结果 ===")
    # verify_items = [(traces, path) for _, traces, path in range_results]
    # verify_results = loaded_tree_weidu.batch_verify(verify_items)
    # for lon, is_valid in verify_results:
    #     print(f"经度 {lon} 验证结果：{'通过' if is_valid else '失败'}")
    # time2 = time.time()
    # vo_time_weidu = time2 - time1
    #
    # min_lon, max_lon = 1212121233 , 1212121293
    # time1 = time.time()
    # range_results = loaded_tree_shijian.query_by_range(min_lon, max_lon)
    # time2 = time.time()
    # query_time_shijian = time2 - time1
    # csv_path = "vo_shijian.csv"
    # loaded_tree_shijian.save_range_query_results_to_csv(range_results, csv_path)
    # print(f"\n查询结果包含 {len(range_results)} 个经度点：")
    # for lon, traces, _ in range_results:
    #     print(f"经度 {lon}：{len(traces)} 条轨迹")
    #
    # time1 = time.time()

    # 5. 验证加载树的批量验证功能
    # print("\n=== 5. 批量验证查询结果 ===")
    # verify_items = [(traces, path) for _, traces, path in range_results]
    # verify_results = loaded_tree_shijian.batch_verify(verify_items)
    # for lon, is_valid in verify_results:
    #     print(f"经度 {lon} 验证结果：{'通过' if is_valid else '失败'}")
    # time2 = time.time()
    # vo_time_shijian = time2 - time1
    #
    # print(query_time_weidu+query_time_shijian+query_time_shijian)
    #
    # print(vo_time_jingdu+vo_time_shijian+vo_time_weidu)