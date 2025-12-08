# ------------------------------ First-level VO Hash Collection ------------------------------
def first_vo_hash_collect1(rp_path):
    """
    First-level Verification Object (VO) hash collection - for regions that satisfy query range
    
    Args:
        rp_path (list): Path of RP-Tree nodes traversed during query (from leaf to root)
    
    Returns:
        list: Stack of hash components for first-level verification (each element is a list of hash parts)
    """
    stack = []
    # Store verification data for each level
    a = []
    flag = rp_path.pop()  # Get current RP-Tree node (start from leaf node)
    
    # Add node boundary coordinates to hash components
    str_border_lng0 = str(flag.border_lng[0])
    a.append(str_border_lng0)
    str_border_lng1 = str(flag.border_lng[1])
    a.append(str_border_lng1)
    str_border_lat0 = str(flag.border_lat[0])
    a.append(str_border_lat0)
    str_border_lat1 = str(flag.border_lat[1])
    a.append(str_border_lat1)

    # Handle leaf node: add placeholders for adjacent edges (to be filled during verification)
    if flag.is_leaf():
        if flag.adjacent_list:
            for edge in flag.adjacent_list:
                a.append(" ")
    # Handle internal node: add placeholders for linking edges (to be filled during verification)
    else:
        if flag.linking:
            for edge in flag.linking:
                a.append(" ")

    # Add left child's merged hash if exists
    if flag.left:
        a.append(flag.left.rp_hash_merge)
    # Add right child's merged hash if exists
    if flag.right:
        a.append(flag.right.rp_hash_merge)
    
    stack.append(a)

    # Traverse up the RP-Tree path (from leaf to root)
    while rp_path:
        b = []
        flag1 = rp_path.pop()  # Get parent node
        
        # Add parent node boundary coordinates
        str_border_lng0 = str(flag1.border_lng[0])
        b.append(str_border_lng0)
        str_border_lng1 = str(flag1.border_lng[1])
        b.append(str_border_lng1)
        str_border_lat0 = str(flag1.border_lat[0])
        b.append(str_border_lat0)
        str_border_lat1 = str(flag1.border_lat[1])
        b.append(str_border_lat1)

        # Handle leaf node: add merged hashes of adjacent edges
        if flag1.is_leaf():
            if flag1.adjacent_list:
                for edge in flag1.adjacent_list:
                    b.append(edge.edge_merge)
        # Handle internal node: add merged hashes of linking edges
        else:
            if flag1.linking:
                for edge in flag1.linking:
                    b.append(edge.edge_merge)
        
        # Case 1: Left child is the node to be verified (add placeholder)
        if flag1.left and flag1.left == flag:
            b.append(" ")
            # Add right child's merged hash if exists
            if flag1.right:
                b.append(flag1.right.rp_hash_merge)

        # Case 2: Right child is the node to be verified (add placeholder)
        if flag1.right and flag1.right == flag:
            # Add left child's merged hash if exists
            if flag1.left:
                b.append(flag1.left.rp_hash_merge)
            b.append(" ")
        
        stack.append(b)
        flag = flag1  # Move up to parent node for next iteration

    return stack


def first_vo_hash_collect2(rp_path):
    """
    First-level Verification Object (VO) hash collection - for regions that DO NOT satisfy query range
    
    Args:
        rp_path (list): Path of RP-Tree nodes traversed during query (from leaf to root)
    
    Returns:
        list: Stack of hash components for first-level verification (each element is a list of hash parts)
    """
    stack = []
    # Store verification data for each level
    a = []
    flag = rp_path.pop()  # Get current RP-Tree node (start from leaf node)
    
    # Add placeholders for node boundary coordinates (region not in query range)
    a.append(" ")
    a.append(" ")
    a.append(" ")
    a.append(" ")

    # Handle leaf node: add merged hashes of adjacent edges
    if flag.is_leaf():
        if flag.adjacent_list:
            for edge in flag.adjacent_list:
                a.append(edge.edge_merge)
    # Handle internal node: add merged hashes of linking edges
    else:
        if flag.linking:
            for edge in flag.linking:
                a.append(edge.edge_merge)

    # Add left child's merged hash if exists
    if flag.left:
        a.append(flag.left.rp_hash_merge)
    # Add right child's merged hash if exists
    if flag.right:
        a.append(flag.right.rp_hash_merge)
    
    stack.append(a)

    # Traverse up the RP-Tree path (from leaf to root)
    while rp_path:
        b = []
        flag1 = rp_path.pop()  # Get parent node
        
        # Add parent node boundary coordinates
        str_border_lng0 = str(flag1.border_lng[0])
        b.append(str_border_lng0)
        str_border_lng1 = str(flag1.border_lng[1])
        b.append(str_border_lng1)
        str_border_lat0 = str(flag1.border_lat[0])
        b.append(str_border_lat0)
        str_border_lat1 = str(flag1.border_lat[1])
        b.append(str_border_lat1)

        # Handle leaf node: add merged hashes of adjacent edges
        if flag1.is_leaf():
            if flag1.adjacent_list:
                for edge in flag1.adjacent_list:
                    b.append(edge.edge_merge)
        # Handle internal node: add merged hashes of linking edges
        else:
            if flag1.linking:
                for edge in flag1.linking:
                    b.append(edge.edge_merge)
        
        # Case 1: Left child is the node to be verified (add placeholder)
        if flag1.left and flag1.left == flag:
            b.append(" ")
            # Add right child's merged hash if exists
            if flag1.right:
                b.append(flag1.right.rp_hash_merge)

        # Case 2: Right child is the node to be verified (add placeholder)
        if flag1.right and flag1.right == flag:
            # Add left child's merged hash if exists
            if flag1.left:
                b.append(flag1.left.rp_hash_merge)
            b.append(" ")
        
        stack.append(b)
        flag = flag1  # Move up to parent node for next iteration

    return stack

# Example time range: 1538461200 (start), 1538464800 (end) (Unix timestamps)

# ------------------------------ Second-level VO Hash Collection ------------------------------
def second_vo_hash_collect_find(traj, start_time, end_time):
    """
    Second-level VO hash collection - verify trajectory against time range
    
    Args:
        traj (Trajectory): Trajectory object to verify
        start_time (int): Start timestamp of query time range (Unix timestamp)
        end_time (int): End timestamp of query time range (Unix timestamp)
    
    Returns:
        tuple: 
            - list: Hash components for second-level verification
            - int: Flag (1 = trajectory matches time range, 0 = no match)
    """
    flag = 1  # Default: assume trajectory matches time range
    a = []
    
    # Case 1: Trajectory overlaps with query time range (return trajectory ID for verification)
    if start_time <= traj.end_time and traj.start_time <= end_time:
        start_time_str = str(traj.start_time)
        end_time_str = str(traj.end_time)
        a.append(start_time_str)
        a.append(end_time_str)
        a.append(" ")  # Placeholder for trajectory hash (to be filled during verification)
    # Case 2: Trajectory does NOT overlap with query time range
    else:
        a.append(" ")  # Placeholder for start time
        a.append(" ")  # Placeholder for end time
        a.append(traj.traj_hash)  # Add trajectory hash for verification
        flag = 0  # Mark trajectory as non-matching

    return a, flag


def no_edge_vo_hash(edge):
    """
    VO hash collection for edges OUTSIDE the query range
    (Edges may fall outside query range due to longitude/latitude constraints and still need verification)
    
    Args:
        edge (Edge): Edge object outside the query range
    
    Returns:
        list: Hash components for edge verification (placeholders + trajectory hashes)
    """
    a = []
    # Add placeholders for edge coordinates (edge not in query range)
    a.append(" ")
    a.append(" ")
    a.append(" ")
    a.append(" ")
    
    # Add merged hashes of all trajectories on this edge
    for traj in edge.traj_hashList:
        a.append(traj.merge_hash)
    
    return a


def edge_vo_hash(edge):
    """
    VO hash collection for edges WITHIN the query range
    
    Args:
        edge (Edge): Edge object inside the query range
    
    Returns:
        list: Hash components for edge verification (coordinates + placeholders for trajectory hashes)
    """
    a = []
    # Add edge coordinates (start/end vertices' longitude/latitude)
    str1 = str(edge.start.lng)
    str2 = str(edge.start.lat)
    str3 = str(edge.end.lng)
    str4 = str(edge.end.lat)
    a.append(str1)
    a.append(str2)
    a.append(str3)
    a.append(str4)
    
    # Add placeholders for trajectory hashes (to be filled during verification)
    for traj in edge.traj_hashList:
        a.append(" ")
    
    return a
