import csv
import hashlib
import json
import time

from filter_traj_id import filter_id, final_filter_id
from traj_time_insert import traj_to_dict, dict_to_traj, insert_time_stamp, update, insert_beijing_to_vtrq
from range_query import range_query, proof_vo


# 定义将 RPTreeNode 转换为字典的函数
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
        "rp_hash_merge":node.rp_hash_merge
    }


def edge_to_dict(edge):
    return {
        "id": edge.id,
        "start": vertex_to_dict(edge.start),
        "end": vertex_to_dict(edge.end),
        "traj_hashList": [traj_to_dict(traj) for traj in edge.traj_hashList],
        "edge_merge": edge.edge_merge,
        "weight":edge.weight
    }


# 将 Vertex 类转换为字典的函数
def vertex_to_dict(vertex):
    return {
        'id': vertex.id,
        'lat': vertex.lat,
        'lng': vertex.lng
    }


# 保存 RP 树到 JSON 文件
def save_rp_tree_json(root, filename):
    data = rptree_to_dict(root)
    try:
        with open(filename, 'w') as f:
            json.dump(data, f)
    except Exception as e:
        print(f"保存文件时出错: {e}")



# 定义顶点类，用于表示图中的顶点
class Vertex:
    def __init__(self, id, lat, lng):
        # 顶点的唯一标识符
        self.id = id
        self.lat = lat
        self.lng = lng



# 定义边类，用于表示图中两个顶点之间的连接
class Edge:
    def __init__(self, id, start, end, weight):
        # 边的唯一标识符
        self.id = id
        self.start = start
        self.end = end
        # 新增属性，用于存储区间树的根哈希
        self.traj_hashList = []
        self.edge_merge = ""
        self.weight = weight

    def __repr__(self):
        return f"Edge(id={self.id}, start={self.start}, end={self.end})"


# 定义 RP - Tree 节点类，用于构建递归划分树
class RPTreeNode:
    def __init__(self, border_lat, border_lng, linking=None, left=None, right=None,
                 adjacent_list=None):
        # 节点所覆盖区域的纬度边界，为一个元组 (最小纬度, 最大纬度)
        self.border_lat = border_lat
        self.border_lng = border_lng
        self.linking = linking if linking else []
        self.left = left
        self.right = right
        self.adjacent_list = adjacent_list if adjacent_list else []
        self.rp_hash_merge = ""

    def is_leaf(self):
        return self.left is None and self.right is None



def find_best_split(sorted_V, E, d):
    # 获取排序后顶点列表的长度
    n = len(sorted_V)
    # 计算中间分割点的索引
    mid = n // 2

    # 预先计算每个顶点的邻接边列表
    vertex_edges = {v: [] for v in sorted_V}
    for e in E:
        vertex_edges[e.start].append(e)
        vertex_edges[e.end].append(e)

    # 内部函数，用于计算给定左右顶点集合对应的左右边的总权重
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

    # 计算中间分割点对应的左右顶点集合
    left_vertices_mid = sorted_V[:mid + 1]
    right_vertices_mid = sorted_V[mid + 1:]
    # 计算中间分割点对应的左右边的总权重
    left_edges_weight_mid, right_edges_weight_mid = calculate_edges_weight(left_vertices_mid, right_vertices_mid)
    # 计算中间分割点左右边总权重的差值
    min_diff = abs(left_edges_weight_mid - right_edges_weight_mid)
    # 初始化最优分割点为中间分割点
    best_p = mid

    # 向左寻找更优的分割点
    left_diff = min_diff
    left_p = mid
    # 从中间分割点向左遍历
    for p in range(mid - 1, -1, -1):
        # 计算当前分割点对应的左右顶点集合
        left_vertices = sorted_V[:p + 1]
        right_vertices = sorted_V[p + 1:]
        # 计算当前分割点对应的左右边的总权重
        left_edges_weight, right_edges_weight = calculate_edges_weight(left_vertices, right_vertices)
        # 计算当前分割点左右边总权重的差值
        diff = abs(left_edges_weight - right_edges_weight)
        # 如果当前差值小于之前记录的最小差值
        if diff < left_diff:
            # 更新最小差值
            left_diff = diff
            # 更新向左的最优分割点
            left_p = p
        else:
            # 如果差值开始增大，停止向左寻找
            break

    # 向右寻找更优的分割点
    right_diff = min_diff
    right_p = mid
    # 从中间分割点向右遍历
    for p in range(mid + 1, n):
        # 计算当前分割点对应的左右顶点集合
        left_vertices = sorted_V[:p + 1]
        right_vertices = sorted_V[p + 1:]
        # 计算当前分割点对应的左右边的总权重
        left_edges_weight, right_edges_weight = calculate_edges_weight(left_vertices, right_vertices)
        # 计算当前分割点左右边总权重的差值
        diff = abs(left_edges_weight - right_edges_weight)
        # 如果当前差值小于之前记录的最小差值
        if diff < right_diff:
            # 更新最小差值
            right_diff = diff
            # 更新向右的最优分割点
            right_p = p
        else:
            # 如果差值开始增大，停止向右寻找
            break

    # 比较左右两边找到的最小差值，更新最优分割点
    if left_diff < min_diff:
        min_diff = left_diff
        best_p = left_p
    if right_diff < min_diff:
        min_diff = right_diff
        best_p = right_p

    # 重新计算最优分割点对应的左右顶点集合
    left_vertices_best = sorted_V[:best_p + 1]
    right_vertices_best = sorted_V[best_p + 1:]
    # 重新计算最优分割点对应的左右边的总权重
    left_edges_weight_best, right_edges_weight_best = calculate_edges_weight(left_vertices_best, right_vertices_best)

    # 打印找到的最优分割点以及对应左右子树的边的总权重情况
    print(f"At level {d}, best split point p = {best_p}, left edges weight: {left_edges_weight_best}, right edges weight: {right_edges_weight_best}")
    return best_p



# 优化后的split函数（使用预排序顶点列表）
def split(V, E, h, uH, d, global_sorted_by_lat, global_sorted_by_lng):
    # 计算当前节点边界
    min_lat = min(v.lat for v in V)
    max_lat = max(v.lat for v in V)
    min_lng = min(v.lng for v in V)
    max_lng = max(v.lng for v in V)
    node = RPTreeNode((min_lat, max_lat), (min_lng, max_lng))
    print(f"Level {d}: Lat ({min_lat}, {max_lat}), Lng ({min_lng}, {max_lng})")

    # 终止条件
    if d >= uH or len(E) < h:
        node.adjacent_list = [e for e in E if
                              min_lat <= e.start.lat <= max_lat and
                              min_lng <= e.start.lng <= max_lng and
                              min_lat <= e.end.lat <= max_lat and
                              min_lng <= e.end.lng <= max_lng]
        print(f"Level {d} (leaf): {len(node.adjacent_list)} edges")
        return node

    # 选择分割方向并获取预排序子集
    lat_range = max_lat - min_lat
    lng_range = max_lng - min_lng
    split_by_lat = lat_range > lng_range
    current_v_ids = {v.id for v in V}  # 用ID快速筛选

    # 从预排序列表中筛选当前顶点子集（保持有序）
    if split_by_lat:
        sorted_V = [v for v in global_sorted_by_lat if v.id in current_v_ids]
    else:
        sorted_V = [v for v in global_sorted_by_lng if v.id in current_v_ids]
    print(f"Level {d} split by {'latitude' if split_by_lat else 'longitude'}")

    # 寻找最优分割点
    best_p = find_best_split(sorted_V, E, d)
    V_left = sorted_V[:best_p + 1]
    V_right = sorted_V[best_p + 1:]

    # 分割边集
    V_left_set = set(V_left)
    V_right_set = set(V_right)
    E_left = [e for e in E if e.start in V_left_set and e.end in V_left_set]
    E_right = [e for e in E if e.start in V_right_set and e.end in V_right_set]



    # 递归构建左子树
    print(f"Recursively building left subtree at level {d + 1}...")
    node.left = split(V_left, E_left, h, uH, d + 1, global_sorted_by_lat, global_sorted_by_lng)
    # 递归构建右子树
    print(f"Recursively building right subtree at level {d + 1}...")
    node.right = split(V_right, E_right, h, uH, d + 1, global_sorted_by_lat, global_sorted_by_lng)

    # 处理跨子树边
    node.linking = [e for e in E if
                    (e.start in V_left_set and e.end in V_right_set) or
                    (e.start in V_right_set and e.end in V_left_set)]

    # 将连接边列表设置到当前节点的 linking 属性中

    return node




# 从文件中加载顶点和边的数据
def load_graph(node_file, edge_file):
    # 存储顶点的列表
    V = []
    # 存储边的列表
    E = []
    # 顶点字典，用于根据顶点 id 快速查找顶点
    vertex_dict = {}

    # 加载顶点数据
    with open(node_file, 'r') as f:
        for line in f:
            # 去除每行首尾的空白字符并按空格分割
            parts = line.strip().split()
            # 获取顶点的 id
            id = int(parts[0])
            # 获取顶点的纬度
            lat = float(parts[2])
            # 获取顶点的经度
            lng = float(parts[1])
            # 创建一个新的顶点对象
            vertex = Vertex(id, lat, lng)
            # 将顶点添加到顶点列表中
            V.append(vertex)
            # 将顶点存入顶点字典
            vertex_dict[id] = vertex

    # 加载边数据
    with open(edge_file, 'r') as f:
        for line in f:
            # 去除每行首尾的空白字符并按空格分割
            parts = line.strip().split()
            # 获取边的 id
            id = int(parts[0])
            # 获取边的起始顶点 id
            start_id = int(parts[1])
            # 获取边的结束顶点 id
            end_id = int(parts[2])

            weight = float(parts[3])
            # 从顶点字典中获取起始顶点
            start_vertex = vertex_dict[start_id]
            # 从顶点字典中获取结束顶点
            end_vertex = vertex_dict[end_id]
            # 创建一个新的边对象
            edge = Edge(id, start_vertex, end_vertex, weight)
            # 将边添加到边列表中
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
    node = RPTreeNode(data["border_lat"], data["border_lng"])
    node.adjacent_list = adjacent_list
    node.linking = linking
    node.rp_hash_merge = data["rp_hash_merge"]
    node.left = dict_to_rptree(data["left"])
    node.right = dict_to_rptree(data["right"])
    return node

def dict_to_vertex(vertex_data):
    return Vertex(vertex_data['id'], vertex_data['lat'], vertex_data['lng'])

def dict_to_edge(edge_data):
    if edge_data["traj_hashList"]:
        traj_hashList = [dict_to_traj(traj_dict) for traj_dict in edge_data["traj_hashList"]]
    else:
        traj_hashList = []
    start = dict_to_vertex(edge_data['start'])
    end = dict_to_vertex(edge_data['end'])
    edge = Edge(edge_data["id"], start, end, edge_data["weight"])
    edge.traj_hashList = traj_hashList
    edge.edge_merge = edge_data["edge_merge"]
    return edge

# 从 JSON 文件加载 RP 树
def load_rp_tree_json(filename):
        with open(filename, 'r') as f:
            data = json.load(f)
        return dict_to_rptree(data)

# 遍历所有节点并统计边的数量
def count_edges(root):
    if root is None:
        return 0
    # 统计当前节点的边数
    if root.linking:
        a = len(root.linking)
    else:
        a = 0
    if root.adjacent_list:
        b = len(root.adjacent_list)
    else:
        b = 0

    current_edges =  a + b
    # 递归统计左子树的边数
    left_edges = count_edges(root.left)
    # 递归统计右子树的边数
    right_edges = count_edges(root.right)
    # 返回总边数
    return current_edges + left_edges + right_edges

# 第二层索引hash合并(对于每一个rp节点里面的边)  对每一个边都要用一次
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
    for traj in edge.traj_hashList:
        hash_list.append(traj.merge_hash)
    all_hash_join = '+'.join(item for item in hash_list)
    encoded_data = all_hash_join.encode('utf-8')
    hash_object = hashlib.sha256(encoded_data)
    hash_hex = hash_object.hexdigest()
    edge.edge_merge = hash_hex

def first_hash_merge(node):
        if node is None:
            return
        # 递归遍历左子树
        first_hash_merge(node.left)
        # 递归遍历右子树
        first_hash_merge(node.right)


        # 前序遍历，先处理当前节点
        # print(f"Node with lat bounds: {node.border_lat}, lng bounds: {node.border_lng}")
        # 如果是叶子节点
        if node.is_leaf():
            if node.adjacent_list:
                # 对每一条边先计算合并hash
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
        # 如果不是叶子节点是中间节点
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

    # # 顶点数据文件的文件名
    node_file = "filtered_node.txt"
    # 边数据文件的文件名
    edge_file = "filtered_edges_indirected_weight.txt"
    # # 调用 load_graph 函数加载顶点和边的数据


    # 加载数据
    print("开始加载数据...")
    V, E = load_graph(node_file, edge_file)
    if not V or not E:
        print("数据加载失败，退出程序")
    else:
        # 预排序顶点（全局仅一次）
        print("预排序顶点...")
        global_sorted_by_lat = sorted(V, key=lambda v: v.lat)
        global_sorted_by_lng = sorted(V, key=lambda v: v.lng)
    #
    #     # 构建RP-Tree
    #     print("开始构建RP-Tree...")
    #     start_time = time.time()
    #     root = split(
    #         V=V,
    #         E=E,
    #         h=11,
    #         uH=15,
    #         d=0,
    #         global_sorted_by_lat=global_sorted_by_lat,
    #         global_sorted_by_lng=global_sorted_by_lng
    #     )
    #
    #
        vertex_dict = {v.id: v for v in V}
    #     insert_beijing_to_vtrq("北京轨迹汇总结果.csv", root, vertex_dict)
    #     #insert_time_stamp(root,V)
    #     first_hash_merge(root)
    #     build_time = time.time() - start_time
    #     print(build_time)
        #save_rp_tree_json(root, "beijing_rp_0.2.json")

        load_rp_tree = load_rp_tree_json("chengdu_rp_3days.json")

        start_time = time.time()
        result = range_query(load_rp_tree, (104.08, 104.10), (30.66, 30.68), 1538413860, 1538417460)
        end_time = time.time()
        find = end_time - start_time
        import csv
        csv.field_size_limit(10 * 1024 * 1024)  # 10MB
        start_time = time.time()  # 记录开始时间
        if proof_vo('vo.csv'):
              print("验证成功")
        else:
              print("验证失败")
        end_time = time.time()
        vo = end_time - start_time
        print(len(result),"时空查询数量")
        filter_id(result, "trajectories_chengdu.csv")
        time_all = final_filter_id("traj_filtered_chengdu.csv", 104.08, 104.10, 30.66, 30.68,1538413860, 1538417460)
        print(find,"查询时间")
        print(vo,"验证时间")
        print(time_all,"聚合")

