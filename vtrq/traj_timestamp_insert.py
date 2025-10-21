import ast
import csv
import hashlib
import json

from interval_tree import Interval
from new_interval_tree import IntervalTree


#判断两个经纬度方框是否有交集
def rectangles_intersect(rect1, rect2):
    if rect1:# 因为有的节点不是叶子节点，但他什么边也不存
        min_lng1, max_lng1 = rect1[0]
        min_lat1, max_lat1 = rect1[1]
        min_lng2, max_lng2 = rect2[0]
        min_lat2, max_lat2 = rect2[1]
        return not (max_lat1 <= min_lat2 or min_lat1 >= max_lat2 or max_lng1 <= min_lng2 or min_lng1 >= max_lng2)

def traj_timestamp_insert(node,traj,V):
    # 先计算这条轨迹所有点构成的最小矩形
    traj_point_list = []
    for v in V:
        if v.id in traj[0]:
            traj_point_list.append(v)
    # 计算当前节点所覆盖区域的最小纬度
    min_lat = min(v.lat for v in traj_point_list)
    # 计算当前节点所覆盖区域的最大纬度
    max_lat = max(v.lat for v in traj_point_list)
    # 计算当前节点所覆盖区域的最小经度
    min_lng = min(v.lng for v in traj_point_list)
    # 计算当前节点所覆盖区域的最大经度
    max_lng = max(v.lng for v in traj_point_list)
    border_lat = [min_lat,max_lat]
    border_lng = [min_lng,max_lng]
    # 这条轨迹的最小矩形边界范围
    rect_range = [border_lng, border_lat]

    time_insert_query(node,rect_range,traj)




def time_insert_query(node,rect_range,traj):

    if node is None:
        return
    node_rect = (node.border_lng, node.border_lat)
    # 轨迹最小范围和该区域没有交集
    if not rectangles_intersect(node_rect, rect_range):
        return
    # 对于有交集的叶子节点要考虑边是否存在
    if node.is_leaf():
        # 轨迹在边表中的边
        for e in node.adjacent_list:

            # 从轨迹中拿出一条边
            for e1 in traj[1]:

                e1_start = e1[0]
                e1_end = e1[1]
                # 如果从轨迹中拿出来的一条边和列表中的这条边有重合证明轨迹经过这条边

                if (e.start.id == e1_start and e.end.id == e1_end) or (e.start.id == e1_end and e.end.id == e1_start):
                    traj_str = '->'.join(str(item) for item in traj[0])
                    # 找到轨迹总开始和结束的时间

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
        # 如果是分支节点考虑linking
            for e in node.linking:
                flag = 0
                # 从轨迹中拿出一条边
                for e1 in traj[1]:
                    e1_start = e1[0]
                    e1_end = e1[1]

                    # 如果从轨迹中拿出来的一条边和列表中的这条边有重合证明轨迹经过这条边
                    if (e.start.id == e1_start and e.end.id == e1_end) or (e.start.id == e1_end and e.end.id == e1_start):
                        # 这里写一个逻辑把这个轨迹插入这个边的列表下面  先创建一个轨迹对象  插入是按照开始时间顺序
                        traj_str = '->'.join(str(item) for item in traj[0])
                        # 找到轨迹总开始和结束的时间
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

#之前轨迹hash只包含路径和开始结束时间
        time_insert_query(node.left, rect_range,traj)
        time_insert_query(node.right, rect_range,traj)




# 往区间树中插入轨迹信息
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





def insert_time_stamp(node,V):
    for i in range(1,4):
        traj_name_path =  f"C:\\Users\\maoyusen\\Desktop\\Graph-Diffusion-Planning-main\\chengdu-tra-json\\traj-10-{i}.json"
        # 打开 JSON 文件
       # traj_name_path = f"C:\\Users\\maoyusen\\Desktop\\Graph-Diffusion-Planning-main\\loader\\preprocess\\mm\\sets_data\\real2\\trajectories\\traj_mapped_xian_xian10-{i}.json"
        with open(traj_name_path, 'r', encoding='utf-8') as file:
            # 解析 JSON 数据
            nested_list = json.load(file)
        j = 0
        for traj in nested_list:
            j = j + 1
            traj_timestamp_insert(node,traj,V)
            print(f"traj-10-{i}的第{j}条轨迹完成")
    # for i in range(1, 31):
    #     traj_name_path = f"C:\\Users\\maoyusen\\Desktop\\Graph-Diffusion-Planning-main\\chengdu-tra-json\\traj-11-{i}.json"
    #     # 打开 JSON 文件
    #
    #     with open(traj_name_path, 'r', encoding='utf-8') as file:
    #         # 解析 JSON 数据
    #         nested_list = json.load(file)
    #     j = 0
    #     for traj in nested_list:
    #         j = j + 1
    #         traj_timestamp_insert(node, traj, V)
    #         print(f"traj-11-{i}的第{j}条轨迹完成")


def insert_time_stamp2(T):
    j = 0
    for i in range(1,4):
        #traj_name_path = f"C:\\Users\\maoyusen\\Desktop\\Graph-Diffusion-Planning-main\\loader\\preprocess\\mm\\sets_data\\real2\\trajectories\\traj_mapped_xian_xian10-{i}.json"
        traj_name_path = f"C:\\Users\\maoyusen\\Desktop\\Graph-Diffusion-Planning-main\\chengdu-tra-json\\traj-10-{i}.json"
        # 打开 JSON 文件
        with open(traj_name_path, 'r', encoding='utf-8') as file:
            # 解析 JSON 数据
            nested_list = json.load(file)

        for traj in nested_list:
            j = j + 1
            insert_traj_into_interval_tree(T, traj)
            print(f"traj-10-{i}的第{j}条轨迹完成")

    # for i in range(1,31):
    #     traj_name_path = f"C:\\Users\\maoyusen\\Desktop\\Graph-Diffusion-Planning-main\\chengdu-tra-json\\traj-11-{i}.json"
    #     # 打开 JSON 文件
    #     with open(traj_name_path, 'r', encoding='utf-8') as file:
    #         # 解析 JSON 数据
    #         nested_list = json.load(file)
    #
    #     for traj in nested_list:
    #         j = j + 1
    #         insert_traj_into_interval_tree(T, traj)
    #
    #         print(f"traj-11-{i}的第{j}条轨迹完成")
    # print(j)







# update

def insert_time_stamp1(node,V):
    j = 0
    for i in range(16, 23):

        traj_name_path =f"C:\\Users\\maoyusen\\Desktop\\Graph-Diffusion-Planning-main\\loader\\preprocess\\mm\\sets_data\\real2\\trajectories\\traj_mapped_xian_xian10-{i}.json"
        # 打开 JSON 文件

        with open(traj_name_path, 'r', encoding='utf-8') as file:
            # 解析 JSON 数据
            nested_list = json.load(file)

        for traj in nested_list:
            if j == 5000:
                return
            j = j + 1
            traj_timestamp_insert(node, traj, V)
            print(f"traj-10-{i}的第{j}条轨迹完成")


def insert_time_stamp21(T):
    j = 0
    for i in range(16,23):

        traj_name_path = f"C:\\Users\\maoyusen\\Desktop\\Graph-Diffusion-Planning-main\\loader\\preprocess\\mm\\sets_data\\real2\\trajectories\\traj_mapped_xian_xian10-{i}.json"
        # 打开 JSON 文件
        with open(traj_name_path, 'r', encoding='utf-8') as file:
            # 解析 JSON 数据
            nested_list = json.load(file)

        for traj in nested_list:
            if j == 5000:
                return
            j = j + 1
            insert_traj_into_interval_tree(T, traj)

            print(f"traj-10-{i}的第{j}条轨迹完成")

from typing import List, Tuple


# V是字典

#[[323080529, 116.673939, 39.873818, 1202090710], [606787516, 116.669716, 39.87734, 1202091309], [606787637, 116.670906, 39.878194, 1202091909], [2504904016, 116.671331, 39.878438, 1202092509], [606787632, 116.671806, 39.878659, 1202093109], [1205600825, 116.673354, 39.879236, 1202093709], [2504904019, 116.673287, 39.87933, 1202094309]]
def traj_timestamp_insert_beijing(node, traj, V):
    # 提取轨迹中的id列表（trajectory_points存储的是id号）
    trajectory_points = [point[0] for point in traj[1]]  # 简化列表推导

    # 从字典V中收集轨迹id对应的节点对象（按轨迹id顺序）
    traj_point_list = []
    for point_id in trajectory_points:
        # 直接通过id从字典V中查找，时间复杂度O(1)
        if point_id in V:
            traj_point_list.append(V[point_id])
        else:
            # 可选：处理id不在V中的情况（如打印警告）
            print(f"警告：轨迹中的id {point_id} 在V中未找到对应节点")

    # 容错处理：避免空列表导致min/max报错
    if not traj_point_list:
        print("轨迹中没有找到有效节点，跳过处理")
        return

    # 计算边界（逻辑不变）
    min_lat = min(v.lat for v in traj_point_list)
    max_lat = max(v.lat for v in traj_point_list)
    min_lng = min(v.lng for v in traj_point_list)
    max_lng = max(v.lng for v in traj_point_list)

    border_lat = [min_lat, max_lat]
    border_lng = [min_lng, max_lng]
    rect_range = [border_lng, border_lat]  # 轨迹的最小矩形边界范围

    time_insert_query_beijing(node, rect_range, traj)


def time_insert_query_beijing(node,rect_range,traj):

    if node is None:
        return
    node_rect = (node.border_lng, node.border_lat)
    # 轨迹最小范围和该区域没有交集
    if not rectangles_intersect(node_rect, rect_range):
        return
    # 对于有交集的叶子节点要考虑边是否存在
    if node.is_leaf():
        # 轨迹在边表中的边
        for e in node.adjacent_list:
            # 从轨迹中拿出一条边
            for i in range(len(traj[1]) - 1):
                e1_start = traj[1][i][0]
                e1_end = traj[1][i + 1][0]
                # 如果从轨迹中拿出来的一条边和列表中的这条边有重合证明轨迹经过这条边

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
                    #print("找到", e.start.id, e.end.id)
                    break


    if not node.is_leaf():
        if rectangles_intersect(node.linking_box, rect_range):
        # 如果是分支节点考虑linking
            for e in node.linking:

                for i in range(len(traj[1]) - 1):
                    e1_start = traj[1][i][0]
                    e1_end = traj[1][i + 1][0]
                    # 如果从轨迹中拿出来的一条边和列表中的这条边有重合证明轨迹经过这条边

                    # 如果从轨迹中拿出来的一条边和列表中的这条边有重合证明轨迹经过这条边
                    if (e.start.id == e1_start and e.end.id == e1_end) or (e.start.id == e1_end and e.end.id == e1_start):
                        # 这里写一个逻辑把这个轨迹插入这个边的列表下面  先创建一个轨迹对象  插入是按照开始时间顺序

                        # 找到轨迹总开始和结束的时间
                        traj_start_time = traj[1][0][-1]
                        traj_end_time = traj[1][-1][-1]


                        traj_start_time_str = str(traj_start_time)
                        traj_end_time_str = str(traj_end_time)
                        str_time_traj = traj[0] + traj_start_time_str + traj_end_time_str
                        encoded_data = str_time_traj.encode('utf-8')
                        hash_object = hashlib.sha256(encoded_data)
                        hash_hex = hash_object.hexdigest()
                        e.traj_hashList.append(hash_hex)
                        #print("找到",e.start.id,e.end.id)
                        break


        time_insert_query_beijing(node.left, rect_range,traj)
        time_insert_query_beijing(node.right, rect_range,traj)


def insert_beijing_to_vtrq(file_path, node, V):
    with open(file_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        next(reader)  # 跳过表头
        i = 0
        for row in reader:
            traj = []  # 每次循环开始时初始化traj，确保为空
            i += 1
            if len(row) < 2:
                print(f"第{i}行：数据不完整，跳过")
                continue

            # 提取traj_id
            traj_id = row[0].strip()

            # 解析points_data
            try:
                points_data = ast.literal_eval(row[1].strip())
                traj.append(traj_id)  # traj[0] = 轨迹ID
                traj.append(points_data)  # traj[1] = 点数据列表
            except (SyntaxError, ValueError) as e:
                print(f"第{i}行解析错误（traj_id: {traj_id}）: {e}")
                continue  # 此时traj仍是空列表，无需额外重置
            if 14000< i <= 14500:
            # 处理当前轨迹
                traj_timestamp_insert_beijing(node, traj, V)
                print(f"第{i}行处理完成")  # 修正打印格式





# 往区间树中插入轨迹信息
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
        next(reader)  # 跳过表头
        i = 0
        for row in reader:
            traj = []  # 每次循环开始时初始化traj，确保为空
            i += 1
            if len(row) < 2:
                print(f"第{i}行：数据不完整，跳过")
                continue

            # 提取traj_id
            traj_id = row[0].strip()

            # 解析points_data
            try:
                points_data = ast.literal_eval(row[1].strip())
                traj.append(traj_id)  # traj[0] = 轨迹ID
                traj.append(points_data)  # traj[1] = 点数据列表
            except (SyntaxError, ValueError) as e:
                print(f"第{i}行解析错误（traj_id: {traj_id}）: {e}")
                continue  # 此时traj仍是空列表，无需额外重置
            if 14000< i <= 14500:
                insert_traj_into_interval_tree_beijing(T, traj)
                print(f"第{i}行处理完成")  # 修正打印格式
