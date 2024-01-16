import numpy as np
from utils import load_env, get_collision_fn_PR2, execute_trajectory
from pybullet_tools.utils import connect, disconnect, wait_if_gui, joint_from_name, get_joint_positions, set_joint_positions, get_joint_info, get_link_pose, link_from_name
import random
### YOUR IMPORTS HERE ###
import time
from utils import draw_sphere_marker
closest_dist = 100
#########################


joint_names =('l_shoulder_pan_joint','l_shoulder_lift_joint','l_elbow_flex_joint','l_upper_arm_roll_joint','l_forearm_roll_joint','l_wrist_flex_joint')

def main(screenshot=False):
    # initialize PyBullet
    connect(use_gui=True)
    # load robot and obstacle resources
    robots, obstacles = load_env('pr2table.json')

    # define active DoFs
    joint_names =('l_shoulder_pan_joint','l_shoulder_lift_joint','l_elbow_flex_joint','l_upper_arm_roll_joint','l_forearm_roll_joint','l_wrist_flex_joint')
    joint_idx = [joint_from_name(robots['pr2'], jn) for jn in joint_names]

    # parse active DoF joint limits
    joint_limits = {joint_names[i] : (get_joint_info(robots['pr2'], joint_idx[i]).jointLowerLimit, get_joint_info(robots['pr2'], joint_idx[i]).jointUpperLimit) for i in range(len(joint_idx))}

    collision_fn = get_collision_fn_PR2(robots['pr2'], joint_idx, list(obstacles.values()))
    # Example use of collision checking
    # print("Robot colliding? ", collision_fn((0.5, 1.19, -1.548, 1.557, -1.32, -0.1928)))

    start_config = tuple(get_joint_positions(robots['pr2'], joint_idx))
    goal_config = (0.5, 0.33, -1.548, 1.557, -1.32, -0.1928)
    path = []
    ### YOUR CODE HERE ###

    def sample_config(joint_limits):
        q = []
        for joint_name, (lower, upper) in joint_limits.items():
            q.append(random.uniform(lower, upper))
        return tuple(q)

    def nearest(tree, q):
        min_dist = float('inf')
        nearest_node = []

        for q_node in tree.keys():
            dist = np.linalg.norm(np.array(q_node) - np.array(q))
            
            if dist < min_dist:
                min_dist = dist
                nearest_node = q_node 

        node_norms = [(node, np.linalg.norm(np.array(node) - np.array(q))) for node in tree.keys()]
        sorted_nodes = sorted(node_norms, key=lambda x: x[1], reverse=True)
        if(len(sorted_nodes) > 11):
            # Get the 10th node (index 5 since indexing starts at 0)
            nearest_node = sorted_nodes[9][0]
        global closest_dist
        closest_dist = min(closest_dist,min_dist)
        print("min_dist",closest_dist)
        return nearest_node

    def connect_tree(tree, q_near, q_target, step_size,collision_fn):
        direction =  np.array(q_target) - np.array(q_near)
        d_norm = np.linalg.norm(direction)
        if d_norm > step_size:
            direction_step = (direction/d_norm) * step_size
        else:
            direction_step = direction
            # print("came here fortunately")
        q_new = tuple(np.array(q_near) + direction_step)
        isCollision = False
        while True:    
            if collision_fn(q_new):
                isCollision = True
                # print("breaking now because of collision")
                break
            if q_near not in tree:
                tree[q_near] = []
            tree[q_near].append(q_new)
            if q_new == q_target:
                if q_target not in tree:
                    tree[q_target] = []
                tree[q_target].append(q_new)
                # print("reached q_rand", q_new, q_target)
                break
            
            q_near = q_new
            direction = np.array(q_target) - np.array(q_near)
            d_norm = np.linalg.norm(direction)
            direction_step = direction if d_norm <= step_size else (direction / d_norm) * step_size
            q_new = tuple(np.array(q_near) + direction_step)
        # print("Check: ",q_target in tree,", ",isCollision)
        return tree

    def backtrace(tree, start, end):
        path = [end]
        current = end
        while current != start:
            for node, children in tree.items():
                if current in children:
                    path.append(node)
                    current = node
                    break
        path.reverse() 
        return path

    def smoothing(path, collision_fn, iter_smooth=150):
        sm_path = path[:]
        for _ in range(iter_smooth):
            n = len(sm_path)
            if n <= 2:
                return sm_path
            i, j = sorted(random.sample(range(n), 2))
            if j - i <= 1:
                continue
            
            temp_tree = {sm_path[i]: []}
            temp_tree = connect_tree(temp_tree, sm_path[i], sm_path[j], step_size=0.05, collision_fn=collision_fn)
            
            # Check if the goal of the shortcut is in the resulting temp_tree
            if sm_path[j] in temp_tree:
                shortcut = backtrace(temp_tree, sm_path[i], sm_path[j])
                sm_path = sm_path[:i] + shortcut + sm_path[j + 1:]

        return sm_path

    PR2 = robots['pr2']
    iter = 100000
    t = time.time()
    step_size = 0.05
    goal_bias = 0.1
    tree = {start_config:[]}
    
    tik = time.time()
    for i in range(iter):
        start = start_config
        goal = goal_config
        p = random.random()
        # print("p is ",p)
        if p < goal_bias:
            # print(p)
            q_rand = goal
            # print("taking goal")
        else:
            q_rand = sample_config(joint_limits)
        
        q_nearest = nearest(tree=tree, q=q_rand)
        tree = connect_tree(tree=tree, q_near=q_nearest, q_target=q_rand, step_size=step_size,collision_fn=collision_fn)

        if goal_config in tree:
            path = backtrace(tree, start_config, goal_config)
            break

    for point in path:
        collision_fn(point)
        ee_pose = get_link_pose(PR2, link_from_name(PR2, 'l_gripper_tool_frame'))
        draw_sphere_marker(ee_pose[0], 0.02, (1, 0, 0, 1))

    # smoothing the path
    path = smoothing(path, collision_fn, 150)
    tok = time.time()

    for pi in path:
        set_joint_positions(PR2, joint_idx, pi)
        ee_pose = get_link_pose(PR2, link_from_name(PR2, 'l_gripper_tool_frame'))
        draw_sphere_marker(ee_pose[0], 0.02, (0, 0, 1, 1))

    if path is None:
        print("No Path Found")
    
    print(f"time taken with smoothing: {tok - tik}")
    ######################
    # Execute planned path
    execute_trajectory(robots['pr2'], joint_idx, path, sleep=0.1)
    # Keep graphics window opened
    wait_if_gui()
    disconnect()

if __name__ == '__main__':
    main()