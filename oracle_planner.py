import cv2
import numpy as np
import pickle
import math
import heapq

experiment_dir = "/scratch/luisamao/all_terrain/experiment_logs/"
# gt_map = cv2.imread('gt_maps/DJI_0469_gt.png')
# gt_map = cv2.imread('gt_maps/EER1_gt.png')
# gt_map = cv2.imread('gt_maps/MLK_gt.png')
# convert to rgb

def is_red(pixel):
    return pixel[0] >128 and pixel[1] <= 128 and pixel[2] <= 128
def is_black(pixel):
    return pixel[0] <= 128 and pixel[1] <= 128 and pixel[2] <= 128
def is_magenta(pixel):
    return pixel[0] >128 and pixel[1] <= 128 and pixel[2] >128
def is_white(pixel):
    return pixel[0] >128 and pixel[1] >128 and pixel[2] >128
def is_yellow(pixel):
    return pixel[0] >128 and pixel[1] >128 and pixel[2] <= 128


def heuristic(a, b):
    return math.dist(a, b)

def astar(start, goal, costmap):        
    neighbors = [(0,1),(1,0),(-1,0),(0,-1),(1,1),(-1,-1),(1,-1),(-1,1)]
    neighbors = [(i*5, j*5) for (i,j) in neighbors]
    
    g_score = {start: 0}
    f_score = {start: heuristic(start, goal)}
    open_heap = [(f_score[start], start)]
    came_from = {}

    while open_heap:
        current = heapq.heappop(open_heap)[1]

        if heuristic(current, goal) < 5:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(current)

            return path[::-1]

        for idx, (i, j) in enumerate(neighbors):
            neighbor = current[0] + i, current[1] + j

            if 0 <= neighbor[0] < costmap.shape[0]:
                if 0 <= neighbor[1] < costmap.shape[1]:
                    pass                
                else:
                    continue
            else:
                continue
            
            tentative_g_score = g_score[current] + math.dist(current, neighbor) * (costmap[neighbor] / 255.0) * 5
                
            if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g_score
                f_score[neighbor] = tentative_g_score + heuristic(neighbor, goal)
                heapq.heappush(open_heap, (f_score[neighbor], neighbor))
    return []


def oracle_plan(i, realistic=True):
    log = pickle.load(open(experiment_dir+f'log_{i}.pkl', 'rb'))
    start = log['start']
    goal = log['goal']
    map_path = log['map_name']
    gt_map_path = map_path.split('/')[-1]
    # gt_map_path = "gt_maps/MLK_gt.png"
    gt_map = cv2.imread(f'gt_maps/{gt_map_path[:-4]}_gt.png')
    # print the map name
    print(gt_map_path, gt_map.shape)
    # gt_map = cv2.imread(gt_map_path)
    gt_map = cv2.cvtColor(gt_map, cv2.COLOR_BGR2RGB)
    # load the map
    map = cv2.imread(map_path)
    # resize the gtmap to the size of the map
    gt_map = cv2.resize(gt_map, (map.shape[1], map.shape[0]))

    if realistic:
        # create a 1 channel costmap the same size as the gt_map of ones
        costmap = np.ones((gt_map.shape[0], gt_map.shape[1]), dtype=np.float32)
        # if the gtmap pixel is red, magenta, or black, set the costmap pixel to 0
        for i in range(gt_map.shape[0]):
            for j in range(gt_map.shape[1]):
                if is_red(gt_map[i][j]) or is_black(gt_map[i][j]) or is_yellow(gt_map[i][j]):
                    costmap[i][j] = .2
                elif is_white(gt_map[i][j]) or is_magenta(gt_map[i][j]):
                    costmap[i][j] = 0.5

    else:
        # create a 1 channel costmap the same size as the gt_map of ones
        costmap = np.ones((gt_map.shape[0], gt_map.shape[1]), dtype=np.float32) * .2
        # if the gtmap pixel is red, magenta, or black, set the costmap pixel to 0
        for i in range(gt_map.shape[0]):
            for j in range(gt_map.shape[1]):
                if is_red(gt_map[i][j]) or is_black(gt_map[i][j]) or is_yellow(gt_map[i][j]):
                    costmap[i][j] = 1
                elif is_white(gt_map[i][j]) or is_magenta(gt_map[i][j]):
                    costmap[i][j] = 0.5



    # print start and goal
    # print(start, goal)
    # print the shape of the map
    # print(map.shape)
    costmap *= 255

    path = astar(start, goal, costmap)

    # print the number of pixels that go on low, medium, and high cost
    print("Low cost pixels:", sum([1 for p in path if costmap[p] == .2 * 255]))
    print("Medium cost pixels:", sum([1 for p in path if costmap[p] == 0.5 * 255]))
    print("High cost pixels:", sum([1 for p in path if costmap[p] == 1 * 255]))

    # draw the path on the costmap
    # cv2 grayscale to rgb
    costmap = cv2.cvtColor(costmap, cv2.COLOR_GRAY2RGB)
    for p in path:
        # use cvt circle to draw the path
        cv2.circle(costmap, (p[1],p[0]), 5, (0, 0, 255), -1)
    # plot the start and end
    cv2.circle(costmap, (start[1], start[0]), 5, (0, 255, 0), -1)
    cv2.circle(costmap, (goal[1], goal[0]), 5, (0, 255, 0), -1)

    # save the costmap
    cv2.imwrite('oracle_costmap.png', costmap)

    log = {
        "map_name": map_path,
        "start": start,
        "goal": goal,
        "path": path,
        "costmap": costmap
    }

    # save the log to /scratch/luisamao/all_terrain/upper_bound/log_14.pkl
    with open(f'/scratch/luisamao/all_terrain/upper_bound/log_{i}.pkl', 'wb') as f:
        pickle.dump(log, f)

# oracle_plan(14)

preferences = [True, False, True, False, True, False, True, False,
               True, False, True, False, True, False, True, False, True, False,
               True, False, True, False, True, True, False, True]

# for i in range(12, 26):
#     oracle_plan(i, preferences[i])
#     print("Done with", i)
#     print("__________")
# for i in range(8, 12):
#     oracle_plan(i, preferences[i])
#     print("Done with", i)
#     print("__________")

oracle_plan(7, False)
