import ast
import hashlib
import json
import csv


BLACK = 0
RED = 1


class Interval:
    def __init__(self, low, high):
        self.low = low
        self.high = high


class IntervalTNode:
    def __init__(self, key, inte, traj_hash="", merge_hash=""):
        self.key = key
        self.color = RED
        self.parent = None
        self.left = None
        self.right = None
        self.inte = inte
        self.max = inte.high
        self.min = inte.low
        self.traj_hash = traj_hash  # Add string attribute
        self.merge_hash = merge_hash



class IntervalTree:
    def __init__(self):
        self.NIL = IntervalTNode(-1, Interval(-1, -1))
        self.NIL.color = BLACK
        self.NIL.max = -1
        self.NIL.min = float('inf')   # Initialize min value of NIL node
        self.root = self.NIL

    @staticmethod
    def max_of_three(a, b, c):
        return max(a, max(b, c))

    @staticmethod
    def min_of_three(a, b, c):
        return min(a, min(b, c))

    @staticmethod
    def overlap(a, b):
        return not (a.high < b.low or a.low > b.high)

    @staticmethod
    def is_contained(a, b):
        # Check if a is completely contained in b
        return a.low >= b.low and a.high <= b.high

    @staticmethod
    def overlap_coverage(a, b):
        """
        Determine if interval a meets any of the following conditions:
        1. a is completely contained in b (a.low >= b.low and a.high <= b.high)
        2. a is not completely contained in b, but the overlapping part covers more than one-third of b's total length

        Parameters:
            a: Interval object with low and high attributes (e.g., trajectory time range)
            b: Query interval object with low and high attributes

        Returns:
            bool: Returns True if any of the above conditions are met, otherwise False
        """
        # Calculate total length of b; return False directly if b is an invalid interval (low >= high)
        b_length = b.high - b.low
        if b_length <= 0:
            return False

        # Condition 1: Check if a is completely contained in b
        is_a_in_b = (a.low >= b.low) and (a.high <= b.high)
        if is_a_in_b:
            return True

        # Condition 2: If a is not completely contained in b, check if overlapping part exceeds 1/3 of b
        # Calculate overlapping interval
        overlap_low = max(a.low, b.low)
        overlap_high = min(a.high, b.high)

        # Return False if no overlap
        if overlap_low >= overlap_high:
            return False

        # Calculate overlapping length
        overlap_length = overlap_high - overlap_low

        # Check if overlapping length exceeds 1/3 of b's total length
        return overlap_length > (b_length / 3)

    def interval_t_search(self, i):
        x = self.root
        while x != self.NIL:
            if x.inte.low == i.low and x.inte.high == i.high:
                return x
            elif i.low < x.inte.low:
                x = x.left
            else:
                x = x.right
        return self.NIL

    def interval_t_inorder_walk(self, x, count=0):
        if x != self.NIL:
            # Recursively traverse left subtree
            count = self.interval_t_inorder_walk(x.left, count)
            color_str = "Red" if x.color == RED else "Black"
            print(f"[{x.inte.low:3d} {x.inte.high:3d}]     {color_str}     {x.min}   {x.max}  ")
            # Increment counter for each output
            count += 1
            # Recursively traverse right subtree
            count = self.interval_t_inorder_walk(x.right, count)
        return count


    def interval_t_minimum(self, x):
        if x.left:
            while x.left != self.NIL:
                x = x.left
            return x

    def interval_t_successor(self, x):
        if x.right != self.NIL:
            return self.interval_t_minimum(x.right)
        y = x.parent
        while y != self.NIL and x == y.right:
            x = y
            y = y.parent
        return y

    def left_rotate(self, x):
        y = x.right
        x.right = y.left
        if y.left != self.NIL:
            y.left.parent = x
        y.parent = x.parent
        if x.parent == self.NIL:
            self.root = y
        elif x == x.parent.left:
            x.parent.left = y
        else:
            x.parent.right = y
        y.left = x
        x.parent = y
        y.max = x.max
        y.min = x.min
        x.max = self.max_of_three(x.inte.high, x.left.max, x.right.max)
        x.min = self.min_of_three(x.inte.low, x.left.min, x.right.min)

    def right_rotate(self, x):
        y = x.left
        x.left = y.right
        if y.right != self.NIL:
            y.right.parent = x
        y.parent = x.parent
        if x.parent == self.NIL:
            self.root = y
        elif x == x.parent.left:
            x.parent.left = y
        else:
            x.parent.right = y
        y.right = x
        x.parent = y
        y.max = x.max
        y.min = x.min
        x.max = self.max_of_three(x.inte.high, x.left.max, x.right.max)
        x.min = self.min_of_three(x.inte.low, x.left.min, x.right.min)

    def interval_t_insert_fixup(self, z):
        while z.parent.color == RED:
            if z.parent == z.parent.parent.left:
                y = z.parent.parent.right
                if y.color == RED:
                    z.parent.color = BLACK
                    y.color = BLACK
                    z.parent.parent.color = RED
                    z = z.parent.parent
                else:
                    if z == z.parent.right:
                        z = z.parent
                        self.left_rotate(z)
                    z.parent.color = BLACK
                    z.parent.parent.color = RED
                    self.right_rotate(z.parent.parent)
            else:
                y = z.parent.parent.left
                if y.color == RED:
                    z.parent.color = BLACK
                    y.color = BLACK
                    z.parent.parent.color = RED
                    z = z.parent.parent
                else:
                    if z == z.parent.left:
                        z = z.parent
                        self.right_rotate(z)
                    z.parent.color = BLACK
                    z.parent.parent.color = RED
                    self.left_rotate(z.parent.parent)
        self.root.color = BLACK

    def interval_t_insert(self, inte, traj_hash="", merge_hash=""):
        z = IntervalTNode(inte.low, inte, traj_hash, merge_hash)
        y = self.NIL
        x = self.root
        while x != self.NIL:
            x.max = max(x.max, z.max)
            x.min = min(x.min, z.min)
            y = x
            if z.key < x.key:
                x = x.left
            else:
                x = x.right
        z.parent = y
        if y == self.NIL:
            self.root = z
        elif z.key < y.key:
            y.left = z
        else:
            y.right = z
        z.left = self.NIL
        z.right = self.NIL
        self.interval_t_insert_fixup(z)
        # Update min and max of ancestor nodes
        current = z.parent
        while current != self.NIL:
            current.max = self.max_of_three(current.inte.high, current.left.max, current.right.max)
            current.min = self.min_of_three(current.inte.low, current.left.min, current.right.min)
            current = current.parent

    def interval_t_delete_fixup(self, x):
        while x != self.root and x.color == BLACK:
            if x == x.parent.left:
                w = x.parent.right
                if w.color == RED:
                    w.color = BLACK
                    x.parent.color = RED
                    self.left_rotate(x.parent)
                    w = x.parent.right
                if w.left.color == BLACK and w.right.color == BLACK:
                    w.color = RED
                    x = x.parent
                else:
                    if w.right.color == BLACK:
                        w.left.color = BLACK
                        w.color = RED
                        self.right_rotate(w)
                        w = x.parent.right
                    w.color = x.parent.color
                    x.parent.color = BLACK
                    w.right.color = BLACK
                    self.left_rotate(x.parent)
                    x = self.root
            else:
                w = x.parent.left
                if w.color == RED:
                    w.color = BLACK
                    x.parent.color = RED
                    self.right_rotate(x.parent)
                    w = x.parent.left
                if w.left.color == BLACK and w.right.color == BLACK:
                    w.color = RED
                    x = x.parent
                else:
                    if w.left.color == BLACK:
                        w.right.color = BLACK
                        w.color = RED
                        self.left_rotate(w)
                        w = x.parent.left
                    w.color = x.parent.color
                    x.parent.color = BLACK
                    w.left.color = BLACK
                    self.right_rotate(x.parent)
                    x = self.root
        x.color = BLACK

    def interval_t_delete(self, z):
        if z.left == self.NIL or z.right == self.NIL:
            y = z
        else:
            y = self.interval_t_successor(z)

        g = y.parent
        while g != self.NIL and g:
            g.max = self.max_of_three(g.inte.high, g.left.max, g.right.max)
            g.min = self.min_of_three(g.inte.low, g.left.min, g.right.min)
            g = g.parent

        if y.left != self.NIL:
            x = y.left
        else:
            x = y.right
        x.parent = y.parent

        if y.parent == self.NIL:
            self.root = x
        elif y == y.parent.left:
            y.parent.left = x
        else:
            y.parent.right = x

        if y != z:
            z.key = y.key
            z.inte = y.inte
            z.max = y.max
            z.min = y.min
            z.traj_hash = y.traj_hash  # Synchronize string data
            z.merge_hash = y.merge_hash
            z.traj = y.traj

        if y.color == BLACK:
            self.interval_t_delete_fixup(x)

        # Update min and max of ancestor nodes
        current = x.parent
        while current != self.NIL:
            current.max = self.max_of_three(current.inte.high, current.left.max, current.right.max)
            current.min = self.min_of_three(current.inte.low, current.left.min, current.right.min)
            current = current.parent

    # def search_intersecting_intervals(self, query):
    #     tra = []
    #     a = []
    #     traj = set()
    #
    #     with open(f'vo_time.csv', 'a', newline='') as csvfile:
    #         writer = csv.writer(csvfile)
    #
    #         def _search(x):
    #             if x == self.NIL:
    #                 return
    #
    #             tra.append(x)
    #             if x.max < query.low or x.min > query.high:
    #                 a = list(tra)
    #                 ver = self.verify_hash_collect3(a)
    #                 writer.writerow([ver, [str(x.min), str(x.max)]])
    #                 tra.pop()
    #                 return
    #
    #             if self.overlap(x.inte, query):
    #                 traj.add(x.traj_hash)
    #                 a = list(tra)
    #                 ver = self.verify_hash_collect1(a)
    #                 writer.writerow([ver,[x.traj_hash]])
    #                 # print(f"Data successfully written to vo_time.csv file")
    #
    #             else:
    #                 a = list(tra)
    #                 ver = self.verify_hash_collect2(a)
    #                 writer.writerow([ver,[str(x.inte.low),str(x.inte.high)]])
    #                 # print(f"Data successfully written to vo_time.csv file")
    #
    #
    #
    #             if x.left != self.NIL and x.left.max >= query.low:
    #                 _search(x.left)
    #             elif x.left != self.NIL and x.left.max < query.low:
    #                 t = x.left
    #                 tra.append(t)
    #                 a = list(tra)
    #                 ver = self.verify_hash_collect3(a)
    #                 tra.pop()
    #                 writer.writerow([ver,[str(t.min),str(t.max)]])
    #                 # print(f"Data successfully written to vo_time.csv file")
    #
    #             if x.right != self.NIL and x.right.min <= query.high:
    #                 _search(x.right)
    #             elif x.right != self.NIL and x.right.min > query.high:
    #                 t = x.right  # Fixed typo: x.txt.right -> x.right
    #                 tra.append(t)
    #                 a = list(tra)
    #                 ver = self.verify_hash_collect3(a)
    #                 tra.pop()
    #                 writer.writerow([ver,[str(t.min),str(t.max)]])
    #                 # print(f"Data successfully written to vo_time.csv file")
    #             tra.pop()
    #
    #         _search(self.root)
    #     return traj

    import csv

    def search_intersecting_intervals(self, query):
        tra = []
        a = []
        traj = set()
        output_rows = []  # Store all rows to be written

        # Optimization 1: Cache list append method as local variable to reduce attribute lookup overhead
        append_row = output_rows.append

        def _search(x):
            if x == self.NIL:
                return

            tra.append(x)
            # Optimization 2: Cache x's attributes as local variables to reduce repeated lookup (especially in condition checks)
            x_max = x.max
            x_min = x.min
            if x_max < query.low or x_min > query.high:
                a = list(tra)
                ver = self.verify_hash_collect3(a)
                # Use cached append_row instead of output_rows.append
                append_row([ver, [str(x_min), str(x_max)]])
                tra.pop()
                return

            if self.overlap(x.inte, query):
                traj.add(x.traj_hash)
                a = list(tra)
                ver = self.verify_hash_collect1(a)
                append_row([ver, [x.traj_hash]])
            else:
                a = list(tra)
                ver = self.verify_hash_collect2(a)
                # Cache x.inte's attributes to reduce access overhead
                inte_low = x.inte.low
                inte_high = x.inte.high
                append_row([ver, [str(inte_low), str(inte_high)]])

            # Optimization 3: Cache x.left and x.right checks to reduce repeated attribute access
            x_left = x.left
            if x_left != self.NIL:
                if x_left.max >= query.low:
                    _search(x_left)
                else:
                    t = x_left
                    tra.append(t)
                    a = list(tra)
                    ver = self.verify_hash_collect3(a)
                    tra.pop()
                    append_row([ver, [str(t.min), str(t.max)]])

            x_right = x.right
            if x_right != self.NIL:
                if x_right.min <= query.high:
                    _search(x_right)
                else:
                    t = x_right
                    tra.append(t)
                    a = list(tra)
                    ver = self.verify_hash_collect3(a)
                    tra.pop()
                    append_row([ver, [str(t.min), str(t.max)]])
            tra.pop()

        _search(self.root)

        # Optimization 4: Increase file buffering to reduce disk I/O operations (16MB buffer)
        with open(f'vo_time.csv', 'a', newline='', buffering=16 * 1024 * 1024) as csvfile:
            writer = csv.writer(csvfile)
            writer.writerows(output_rows)

        return traj

    # Note
    def node_to_dict(self, node):
        if node == self.NIL:
            return None
        return {
            "color": node.color,
            "key": node.key,
            "interval": [node.inte.low, node.inte.high],
            "max": node.max,
            "min": node.min,  # Include min value
            "traj_hash": node.traj_hash,  # Include string data
            "merge_hash": node.merge_hash,
            "left": self.node_to_dict(node.left),
            "right": self.node_to_dict(node.right)
        }

    def save_to_json(self, filename):
        tree_dict = self.node_to_dict(self.root)
        with open(filename, 'w') as f:
            json.dump(tree_dict, f, indent=4)

    @staticmethod
    def dict_to_node(tree_dict, NIL):
        if tree_dict is None:
            return NIL
        inte = Interval(tree_dict["interval"][0], tree_dict["interval"][1])
        node = IntervalTNode(tree_dict["key"], inte, tree_dict.get("traj_hash", ""), tree_dict.get("merge_hash", ""))
        node.color = tree_dict["color"]
        node.max = tree_dict["max"]
        node.min = tree_dict["min"]
        node.left = IntervalTree.dict_to_node(tree_dict["left"], NIL)
        node.right = IntervalTree.dict_to_node(tree_dict["right"], NIL)
        if node.left != NIL:
            node.left.parent = node
        if node.right != NIL:
            node.right.parent = node
        return node

    @classmethod
    def load_from_json(cls, filename):
        with open(filename, 'r') as f:
            tree_dict = json.load(f)
        tree = cls()
        tree.root = cls.dict_to_node(tree_dict, tree.NIL)
        if tree.root != tree.NIL:
            tree.root.parent = tree.NIL
        return tree

    def trag_hash_merge(self, x):
        if x != self.NIL:
            self.trag_hash_merge(x.left)
            self.trag_hash_merge(x.right)
            # print(f"[{x.txt.inte.low:3d} {x.txt.inte.high:3d}] {x.txt.traj_hash} {x.txt.merage_hash}")
            # If at least one of the tree's left/right children is not empty, it means it's not a leaf node - calculate merge_hash
            if x.left != self.NIL or x.right != self.NIL:
                if x.left != self.NIL:
                    str_interval_node = str(x.left.min) + str(x.left.max) + str(x.left.inte.low) + str(x.left.inte.high) + x.left.traj_hash + x.left.merge_hash# Should have only merge_hash instead of left
                    x.merge_hash = x.merge_hash + str_interval_node
                if x.right != self.NIL:
                    str_interval_node = str(x.right.min) + str(x.right.max) + str(x.right.inte.low) + str(x.right.inte.high) + x.right.traj_hash + x.right.merge_hash
                    x.merge_hash = x.merge_hash + str_interval_node
                encoded_data = x.merge_hash.encode('utf-8')
                hash_object = hashlib.sha256(encoded_data)
                hash_hex = hash_object.hexdigest()
                x.merge_hash = hash_hex


    def root_hash(self):
        # This part will be modified to hash calculation later
        interval_tree_root_hash = str(self.root.min) + str(self.root.max) + str(self.root.inte.low) + str(self.root.inte.high) + self.root.traj_hash + self.root.merge_hash

        encoded_data = interval_tree_root_hash.encode('utf-8')
        hash_object = hashlib.sha256(encoded_data)
        hash_hex = hash_object.hexdigest()
        interval_tree_root_hash = hash_hex
        return interval_tree_root_hash

    # Collect verification results for found trajectories
    def verify_hash_collect1(self, interval_tree_path):
        # Pop target node from stack first
        # print(interval_tree_path)
        stack = []
        flag = interval_tree_path.pop()
        a = []
        if flag == self.root:
            a.append(str(flag.min))
            a.append(str(flag.max))
            a.append(str(flag.inte.low))
            a.append(str(flag.inte.high))
            a.append(" ")
            a.append(flag.merge_hash)
        # This flagged node is the left child of its parent node
        if interval_tree_path:
            if interval_tree_path[-1].left != self.NIL and flag == interval_tree_path[-1].left:
                a.append(str(flag.min))
                a.append(str(flag.max))
                a.append(str(flag.inte.low))
                a.append(str(flag.inte.high))
                a.append(" ")
                a.append(flag.merge_hash)
                if interval_tree_path[-1].right != self.NIL:
                    a.append(str(interval_tree_path[-1].right.min))
                    a.append(str(interval_tree_path[-1].right.max))
                    a.append(str(interval_tree_path[-1].right.inte.low))
                    a.append(str(interval_tree_path[-1].right.inte.high))
                    a.append(interval_tree_path[-1].right.traj_hash)
                    a.append(interval_tree_path[-1].right.merge_hash)

            elif interval_tree_path[-1].right != self.NIL and flag == interval_tree_path[-1].right:
                if interval_tree_path[-1].left != self.NIL:
                    a.append(str(interval_tree_path[-1].left.min))
                    a.append(str(interval_tree_path[-1].left.max))
                    a.append(str(interval_tree_path[-1].left.inte.low))
                    a.append(str(interval_tree_path[-1].left.inte.high))
                    a.append(interval_tree_path[-1].left.traj_hash)
                    a.append(interval_tree_path[-1].left.merge_hash)
                    a.append(str(flag.min))
                    a.append(str(flag.max))
                    a.append(str(flag.inte.low))
                    a.append(str(flag.inte.high))
                    a.append(" ")
                    a.append(flag.merge_hash)
                else:
                    a.append(str(flag.min))
                    a.append(str(flag.max))
                    a.append(str(flag.inte.low))
                    a.append(str(flag.inte.high))
                    a.append(" ")
                    a.append(flag.merge_hash)
        stack.append(a)

        while interval_tree_path:
            b = []
            flag = interval_tree_path.pop()
            # If empty
            if not interval_tree_path:
                b.append(str(flag.min))
                b.append(str(flag.max))
                b.append(str(flag.inte.low))
                b.append(str(flag.inte.high))
                b.append(flag.traj_hash)
                b.append(" ")
            if interval_tree_path:
                if interval_tree_path[-1].left != self.NIL and flag == interval_tree_path[-1].left:
                    b.append(str(flag.min))
                    b.append(str(flag.max))
                    b.append(str(flag.inte.low))
                    b.append(str(flag.inte.high))
                    b.append(flag.traj_hash)
                    b.append(" ")
                    if interval_tree_path[-1].right != self.NIL:
                        b.append(str(interval_tree_path[-1].right.min))
                        b.append(str(interval_tree_path[-1].right.max))
                        b.append(str(interval_tree_path[-1].right.inte.low))
                        b.append(str(interval_tree_path[-1].right.inte.high))
                        b.append(interval_tree_path[-1].right.traj_hash)
                        b.append(interval_tree_path[-1].right.merge_hash)

                if interval_tree_path[-1].right != self.NIL and flag == interval_tree_path[-1].right:
                    if interval_tree_path[-1].left != self.NIL:
                        b.append(str(interval_tree_path[-1].left.min))
                        b.append(str(interval_tree_path[-1].left.max))
                        b.append(str(interval_tree_path[-1].left.inte.low))
                        b.append(str(interval_tree_path[-1].left.inte.high))
                        b.append(interval_tree_path[-1].left.traj_hash)
                        b.append(interval_tree_path[-1].left.merge_hash)
                    b.append(str(flag.min))
                    b.append(str(flag.max))
                    b.append(str(flag.inte.low))
                    b.append(str(flag.inte.high))
                    b.append(flag.traj_hash)
                    b.append(" ")
            stack.append(b)
        return stack


    # Collect verification for timestamps outside query range
    def verify_hash_collect2(self, interval_tree_path):
        # Pop target node from stack first
        # print(interval_tree_path)
        stack = []
        flag = interval_tree_path.pop()
        a = []
        if flag == self.root:
            a.append(str(flag.min))
            a.append(str(flag.max))
            a.append(" ")
            a.append(" ")
            a.append(flag.traj_hash)  # Keep original type
            a.append(flag.merge_hash)  # Keep original type
        # This flagged node is the left child of its parent node
        if interval_tree_path:
            if interval_tree_path[-1].left != self.NIL and flag == interval_tree_path[-1].left:
                a.append(str(flag.min))
                a.append(str(flag.max))
                a.append(" ")
                a.append(" ")
                a.append(flag.traj_hash)  # Keep original type
                a.append(flag.merge_hash)  # Keep original type
                if interval_tree_path[-1].right != self.NIL:
                    a.append(str(interval_tree_path[-1].right.min))
                    a.append(str(interval_tree_path[-1].right.max))
                    a.append(str(interval_tree_path[-1].right.inte.low))
                    a.append(str(interval_tree_path[-1].right.inte.high))
                    a.append(interval_tree_path[-1].right.traj_hash)  # Keep original type
                    a.append(interval_tree_path[-1].right.merge_hash)  # Keep original type

            elif interval_tree_path[-1].right != self.NIL and flag == interval_tree_path[-1].right:
                if interval_tree_path[-1].left != self.NIL:
                    a.append(str(interval_tree_path[-1].left.min))
                    a.append(str(interval_tree_path[-1].left.max))
                    a.append(str(interval_tree_path[-1].left.inte.low))
                    a.append(str(interval_tree_path[-1].left.inte.high))
                    a.append(interval_tree_path[-1].left.traj_hash)  # Keep original type
                    a.append(interval_tree_path[-1].left.merge_hash)  # Keep original type
                    a.append(str(flag.min))
                    a.append(str(flag.max))
                    a.append(" ")
                    a.append(" ")
                    a.append(flag.traj_hash)  # Keep original type
                    a.append(flag.merge_hash)  # Keep original type
                else:
                    a.append(str(flag.min))
                    a.append(str(flag.max))
                    a.append(" ")
                    a.append(" ")
                    a.append(flag.traj_hash)  # Keep original type
                    a.append(flag.merge_hash)  # Keep original type
        stack.append(a)

        while interval_tree_path:
            b = []
            flag = interval_tree_path.pop()
            # If empty
            if not interval_tree_path:
                b.append(str(flag.min))
                b.append(str(flag.max))
                b.append(str(flag.inte.low))
                b.append(str(flag.inte.high))
                b.append(flag.traj_hash)  # Keep original type
                b.append(" ")
            if interval_tree_path:
                if interval_tree_path[-1].left != self.NIL and flag == interval_tree_path[-1].left:
                    b.append(str(flag.min))
                    b.append(str(flag.max))
                    b.append(str(flag.inte.low))
                    b.append(str(flag.inte.high))
                    b.append(flag.traj_hash)  # Keep original type
                    b.append(" ")
                    if interval_tree_path[-1].right != self.NIL:
                        b.append(str(interval_tree_path[-1].right.min))
                        b.append(str(interval_tree_path[-1].right.max))
                        b.append(str(interval_tree_path[-1].right.inte.low))
                        b.append(str(interval_tree_path[-1].right.inte.high))
                        b.append(interval_tree_path[-1].right.traj_hash)  # Keep original type
                        b.append(interval_tree_path[-1].right.merge_hash)  # Keep original type

                if interval_tree_path[-1].right != self.NIL and flag == interval_tree_path[-1].right:
                    if interval_tree_path[-1].left != self.NIL:
                        b.append(str(interval_tree_path[-1].left.min))
                        b.append(str(interval_tree_path[-1].left.max))
                        b.append(str(interval_tree_path[-1].left.inte.low))
                        b.append(str(interval_tree_path[-1].left.inte.high))
                        b.append(interval_tree_path[-1].left.traj_hash)  # Keep original type
                        b.append(interval_tree_path[-1].left.merge_hash)  # Keep original type
                    b.append(str(flag.min))
                    b.append(str(flag.max))
                    b.append(str(flag.inte.low))
                    b.append(str(flag.inte.high))
                    b.append(flag.traj_hash)  # Keep original type
                    b.append(" ")
            stack.append(b)
        return stack

    # Collect query proof for pruning during query
    def verify_hash_collect3(self, interval_tree_path):
        # Pop target node from stack first
        # print(interval_tree_path)
        stack = []
        flag = interval_tree_path.pop()
        a = []
        if flag == self.root:
            a.append(" ")
            a.append(" ")
            a.append(str(flag.inte.low))
            a.append(str(flag.inte.high))
            a.append(flag.traj_hash)  # Keep original type
            a.append(flag.merge_hash)  # Keep original type
        # This flagged node is the left child of its parent node
        if interval_tree_path:
            if interval_tree_path[-1].left != self.NIL and flag == interval_tree_path[-1].left:
                a.append(" ")
                a.append(" ")
                a.append(str(flag.inte.low))
                a.append(str(flag.inte.high))
                a.append(flag.traj_hash)  # Keep original type
                a.append(flag.merge_hash)  # Keep original type
                if interval_tree_path[-1].right != self.NIL:
                    a.append(str(interval_tree_path[-1].right.min))
                    a.append(str(interval_tree_path[-1].right.max))
                    a.append(str(interval_tree_path[-1].right.inte.low))
                    a.append(str(interval_tree_path[-1].right.inte.high))
                    a.append(interval_tree_path[-1].right.traj_hash)  # Keep original type
                    a.append(interval_tree_path[-1].right.merge_hash)  # Keep original type

            elif interval_tree_path[-1].right != self.NIL and flag == interval_tree_path[-1].right:
                if interval_tree_path[-1].left != self.NIL:
                    a.append(str(interval_tree_path[-1].left.min))
                    a.append(str(interval_tree_path[-1].left.max))
                    a.append(str(interval_tree_path[-1].left.inte.low))
                    a.append(str(interval_tree_path[-1].left.inte.high))
                    a.append(interval_tree_path[-1].left.traj_hash)  # Keep original type
                    a.append(interval_tree_path[-1].left.merge_hash)  # Keep original type
                    a.append(" ")
                    a.append(" ")
                    a.append(str(flag.inte.low))
                    a.append(str(flag.inte.high))
                    a.append(flag.traj_hash)  # Keep original type
                    a.append(flag.merge_hash)  # Keep original type
                else:
                    a.append(" ")
                    a.append(" ")
                    a.append(str(flag.inte.low))
                    a.append(str(flag.inte.high))
                    a.append(flag.traj_hash)  # Keep original type
                    a.append(flag.merge_hash)  # Keep original type
        stack.append(a)

        while interval_tree_path:
            b = []
            flag = interval_tree_path.pop()
            # If empty
            if not interval_tree_path:
                b.append(str(flag.min))
                b.append(str(flag.max))
                b.append(str(flag.inte.low))
                b.append(str(flag.inte.high))
                b.append(flag.traj_hash)  # Keep original type
                b.append(" ")
            if interval_tree_path:
                if interval_tree_path[-1].left != self.NIL and flag == interval_tree_path[-1].left:
                    b.append(str(flag.min))
                    b.append(str(flag.max))
                    b.append(str(flag.inte.low))
                    b.append(str(flag.inte.high))
                    b.append(flag.traj_hash)  # Keep original type
                    b.append(" ")
                    if interval_tree_path[-1].right != self.NIL:
                        b.append(str(interval_tree_path[-1].right.min))
                        b.append(str(interval_tree_path[-1].right.max))
                        b.append(str(interval_tree_path[-1].right.inte.low))
                        b.append(str(interval_tree_path[-1].right.inte.high))
                        b.append(interval_tree_path[-1].right.traj_hash)  # Keep original type
                        b.append(interval_tree_path[-1].right.merge_hash)  # Keep original type

                if interval_tree_path[-1].right != self.NIL and flag == interval_tree_path[-1].right:
                    if interval_tree_path[-1].left != self.NIL:
                        b.append(str(interval_tree_path[-1].left.min))
                        b.append(str(interval_tree_path[-1].left.max))
                        b.append(str(interval_tree_path[-1].left.inte.low))
                        b.append(str(interval_tree_path[-1].left.inte.high))
                        b.append(interval_tree_path[-1].left.traj_hash)  # Keep original type
                        b.append(interval_tree_path[-1].left.merge_hash)  # Keep original type
                    b.append(str(flag.min))
                    b.append(str(flag.max))
                    b.append(str(flag.inte.low))
                    b.append(str(flag.inte.high))
                    b.append(flag.traj_hash)  # Keep original type
                    b.append(" ")
            stack.append(b)
        return stack


def proof_root_hash(verify_hash_list):
        # print(verify_hash_list)
        verify_hash_list.reverse()
        a = verify_hash_list.pop()
        str1 = ""
        for i in a:
            str1 = str1 + i
        encoded_data = str1.encode('utf-8')
        hash_object = hashlib.sha256(encoded_data)
        str1 = hash_object.hexdigest()
        # print(str1)

        while verify_hash_list:
            a = verify_hash_list.pop()
            str2 = ""
            for i in a:
                if i == " ":
                     str2 = str2 + str1
                else:
                    str2 = str2 + i
            encoded_data = str2.encode('utf-8')
            hash_object = hashlib.sha256(encoded_data)
            str2 = hash_object.hexdigest()
            str1 = str2
        # print(str1)
        return str1


def proof_interval(filename):
    file_path = filename
    with open(file_path, 'r', encoding='utf-8') as file:
        reader = csv.reader(file)
        for row in reader:
            col1 = ast.literal_eval(row[0])
            col2 = ast.literal_eval(row[1])
            if len(col2) == 1:
                i = 0
                for j in col1[0]:
                    if j == " ":
                         col1[0][i] = col2[0]
                         break
                    i = i + 1
                if proof_root_hash(col1) != "92b4c3028562946191fee2910582b772283733a16e41eaae107b325aa71705d0":  #select the special root hash value 

                    print("Verification failed")
                    return
            if len(col2) == 2:
                i = 0
                p = 0
                for j in col1[0]:
                    if j == " ":
                         col1[0][i] = col2[p]
                         p = p + 1
                         if p == 2:
                             break
                    i = i + 1
                    # Replace with root hash of this index
                if proof_root_hash(col1) != "cb058e3cb21c55a9690141546d33a96efe068c08174a09e2ae9c02479975f7bc":
                    print("Verification failed")
                    return
    return True

