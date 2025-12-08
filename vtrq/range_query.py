import ast
import csv
import hashlib

from ACC import initialize_accumulator, add_string_element
from hash_collect import edge_vo_hash, no_edge_vo_hash, first_vo_hash_collect2, \
    first_vo_hash_collect1
from vector_cross_product import segment_intersects_rect
from new_interval_tree import Interval

# node_rect = [node.border_lng, node.border_lat]
# Check if two longitude/latitude rectangles intersect
def rectangles_intersect(rect1, rect2):
    if rect1 != []:  # Because some non-leaf nodes may store no edges
        min_lat1, max_lat1 = rect1[1]
        min_lng1, max_lng1 = rect1[0]
        min_lat2, max_lat2 = rect2[1]
        min_lng2, max_lng2 = rect2[0]
        return not (max_lat1 <= min_lat2 or min_lat1 >= max_lat2 or max_lng1 <= min_lng2 or min_lng1 >= max_lng2)
    else:
        return False


def range_query(root, query_lng_range, query_lat_range):
    # Range query
    traj_set = set()
    rp_path = []
    query_rect = [query_lng_range, query_lat_range]

    # Used to collect all data to be written to CSV
    csv_data = []

    def write_to_csv_buffer(ver, data=None):
        if data is not None:
            csv_data.append([ver, data])
        else:
            csv_data.append([ver])

    def process_edge(e, query_rect):
        # First case: Check if at least one of the two endpoints of this edge falls within the query range
        if (query_rect[0][0] <= e.start.lng <= query_rect[0][1] and query_rect[1][0] <= e.start.lat <= query_rect[1][
            1]) or (
                query_rect[0][0] <= e.end.lng <= query_rect[0][1] and query_rect[1][0] <= e.end.lat <= query_rect[1][
            1]):

            # This edge is within the query range
            ver = edge_vo_hash(e)
            csv_data.append(["e_vo_start"])
            write_to_csv_buffer(ver)
            write_to_csv_buffer(e.traj_hashList)
            traj_set.update(e.traj_hashList)
            csv_data.append(["e_vo_end"])

        # If both endpoints are outside the query range, check if the line segment intersects with the query range
        elif segment_intersects_rect(((e.start.lng, e.start.lat), (e.end.lng, e.end.lat)),
                                     ((query_rect[0][0], query_rect[1][0]), (query_rect[0][1], query_rect[1][1]))):

            ver = edge_vo_hash(e)
            csv_data.append(["e_vo_start"])
            write_to_csv_buffer(ver)
            write_to_csv_buffer(e.traj_hashList)
            traj_set.update(e.traj_hashList)
            csv_data.append(["e_vo_end"])
        else:
            # Edge is outside the query range
            ver = no_edge_vo_hash(e)
            data = [str(e.start.lng), str(e.start.lat), str(e.end.lng), str(e.end.lat)]
            csv_data.append(["e_vo_start"])
            write_to_csv_buffer(ver, data)
            csv_data.append(["e_vo_end"])

    def _query(node, query_rect):
        if node is None:
            return
        # Record each node traversed
        rp_path.append(node)
        node_rect = [node.border_lng, node.border_lat]
        # If the region range does not match the query range, return verification information for inconsistent longitude/latitude
        if not rectangles_intersect(node_rect, query_rect):
            a = list(rp_path)
            ver = first_vo_hash_collect2(a)
            data = [node.border_lng[0], node.border_lng[1], node.border_lat[0], node.border_lat[1]]
            csv_data.append(["lng_lat_vo_start"])
            write_to_csv_buffer(ver, [data])
            csv_data.append(["lng_lat_vo_end"])
            rp_path.pop()
            return

        if node.is_leaf():
            a = list(rp_path)
            ver = first_vo_hash_collect1(a, True)
            csv_data.append(["lng_lat_vo_start"])
            write_to_csv_buffer(ver)
            if node.adjacent_list:
                for e in node.adjacent_list:
                    process_edge(e, query_rect)
                csv_data.append(["lng_lat_vo_end"])

        if not node.is_leaf():
            p = True
            # If there is no intersection between the query range and the adjacent edge range, individual query of each edge can be omitted
            if not rectangles_intersect(node.linking_box, query_rect):
                p = False
            # First, since this region meets the query range, write the VO (Verification Object) of the first-level index
            a = list(rp_path)
            # If the adjacent edge range intersects with the query range
            if p:
                ver = first_vo_hash_collect1(a, p)
                csv_data.append(["lng_lat_vo_start"])
                write_to_csv_buffer(ver)
                if node.linking:
                    for e in node.linking:
                        process_edge(e, query_rect)
                csv_data.append(["lng_lat_vo_end"])

            else:
                ver = first_vo_hash_collect1(a, p)
                csv_data.append(["lng_lat_vo_start"])
                if node.linking_box:
                    data = [node.linking_box[0][0], node.linking_box[0][1], node.linking_box[1][0],
                            node.linking_box[1][1]]
                    write_to_csv_buffer(ver, data)
                    csv_data.append(["lng_lat_vo_end"])
                # Adjacent edge range does not exist
                else:
                    data = ["", "", "", ""]
                    write_to_csv_buffer(ver, data)
                    csv_data.append(["lng_lat_vo_end"])

            _query(node.left, query_rect)
            _query(node.right, query_rect)
        rp_path.pop()

    # Query longitude/latitude range
    _query(root, query_rect)

    # Write all collected data to CSV file at once
    with open('vo.csv', 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(csv_data)

    return traj_set




def proof_vo(filename):
    # File path
    file_path = filename
    # Open file
    with open(file_path, 'r', encoding='utf-8') as file:
        # Read file content line by line
        flag1 = []
        flag2 = []
        list1 = []
        list2 = []
        reader = csv.reader(file)

        for row in reader:
            # print(row)
            if row[0] == "lng_lat_vo_start":
                flag1 = next(reader)
                # print(flag1)
                # If the length of lng_lat_vo is 1, it means this region intersects with the query range
                if len(flag1) <= 1:
                    flag1 = ast.literal_eval(flag1[0])
                    flag1.reverse()
                    list1 = []
                # If the length of lng_lat_vo is greater than 1, it means this region does not intersect with the query range; fill in the subsequent four longitude/latitude data to calculate the hash
                else:
                    c = flag1[0]
                    b = flag1[1]
                    c = ast.literal_eval(c)
                    c.reverse()
                    a = c[-1]
                    # Read longitude/latitude data to be filled in
                    b = ast.literal_eval(b)
                    # print(b)
                    # b = b[0]
                    # If b is a nested list, it means the large longitude/latitude range is not satisfied
                    if isinstance(b[0], list):
                        b = b[0]

                    # Insert longitude/latitude data
                    j = 0
                    for i in range(len(a)):
                        if a[i] == " ":
                            a[i] = str(b[j])
                            j = j + 1
                            if j > 3:
                                break
                    # print(a)
                    all_hash_join = '+'.join(item for item in a if item)
                    # print(all_hash_join)
                    encoded_data = all_hash_join.encode('utf-8')
                    hash_object = hashlib.sha256(encoded_data)
                    hash_hex = hash_object.hexdigest()
                    # print(hash_hex)
                    t = c.pop()
                    #
                    # t = c.pop()
                    # print(t)
                    # for i in range(len(t)):
                    #     if t[i] == " ":
                    #         t[i] = hash_hex
                    #         break
                    # all_hash_join = '+'.join(item for item in t)
                    # encoded_data = all_hash_join.encode('utf-8')
                    # hash_object = hashlib.sha256(encoded_data)
                    # hash_hex = hash_object.hexdigest()
                    # print(hash_hex)

                    while c:
                        t = c.pop()
                        # print(t)
                        for i in range(len(t)):
                            if t[i] == " ":
                                t[i] = hash_hex
                                break
                        all_hash_join = '+'.join(item for item in t)
                        # print(all_hash_join)
                        encoded_data = all_hash_join.encode('utf-8')
                        hash_object = hashlib.sha256(encoded_data)
                        hash_hex = hash_object.hexdigest()
                        # print(hash_hex)
                    row = next(reader)
                    # print(row)

                    #print(hash_hex)
                    if hash_hex != "85dc2a65b781bcd76a519f46b4c1ae821c5aa7cffa1a0cf9e064cc4407bc962f":
                        return False

            elif row[0] == "e_vo_start":
                flag2 = next(reader)
                # print(flag2)
                # If the length of e_vo is 1, it means this edge intersects with the query range
                if len(flag2) <= 1:
                    flag2 = ast.literal_eval(flag2[0])
                    list2 = []
                # If the length of e_vo is greater than 1, it means this edge is outside the range, so insert the subsequent longitude/latitude data for verification
                else:
                    a = flag2[0]
                    b = flag2[1]
                    a = ast.literal_eval(a)
                    b = ast.literal_eval(b)
                    for i in range(4):
                        a[i] = b[i]
                    # print(a)
                    # print(44)
                    all_hash_join = '+'.join(item for item in a if item)
                    encoded_data = all_hash_join.encode('utf-8')
                    hash_object = hashlib.sha256(encoded_data)
                    hash_hex = hash_object.hexdigest()
                    list1.append(hash_hex)
                    row = next(reader)
                    # print(row)

            elif row[0] ==  "e_vo_end":
                a = flag2
                if list2:
                    a[-1] = list2[0]
                else:
                    a[-1] = ""
                all_hash_join = '+'.join(item for item in a if item)
                encoded_data = all_hash_join.encode('utf-8')
                hash_object = hashlib.sha256(encoded_data)
                hash_hex = hash_object.hexdigest()
                list1.append(hash_hex)
                # print(list1)

            elif row[0] == "lng_lat_vo_end":
                t = flag1.pop()
                j = 0
                for i in range(len(t)):
                    if t[i] == " ":
                        t[i] = list1[j]
                        j = j + 1

                all_hash_join = '+'.join(item for item in t)
                # print(all_hash_join)
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
               # print(hash_hex)
                if hash_hex != "85dc2a65b781bcd76a519f46b4c1ae821c5aa7cffa1a0cf9e064cc4407bc962f":
                    return False
            # Modify here to use accumulator
            else:
                a = row[0]
                a = ast.literal_eval(a)
                if a:
                    traj_list = []
                    for traj in a:
                        traj_list.append(traj)
                    all_traj_str = '+'.join(item for item in traj_list)
                    encoded_data = all_traj_str.encode('utf-8')
                    hash_object = hashlib.sha256(encoded_data)
                    hash_hex = hash_object.hexdigest()
                    list2.append(hash_hex)

    return True
