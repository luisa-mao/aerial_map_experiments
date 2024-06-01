import cv2
import numpy as np
import pickle

experiment_dir = "/scratch/luisamao/all_terrain/experiment_logs/"
experiment_dir = "/scratch/luisamao/all_terrain/sterling_baseline/"


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




def eval(i, realistic=True):
    log = pickle.load(open(experiment_dir+f'log_{i}.pkl', 'rb'))
    start = log['start']
    goal = log['goal']
    map_path = log['map_name']
    gt_map_path = map_path.split('/')[-1]
    # gt_map_path = "gt_maps/MLK_gt.png"
    gt_map = cv2.imread(f'gt_maps/{gt_map_path[:-4]}_gt.png')

    if gt_map is None:
        gt_map_path = "gt_maps/MLK_gt.png"
        gt_map = cv2.imread(gt_map_path)
    # print the map name
    # print(gt_map_path, gt_map.shape)
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
                elif is_white(gt_map[i][j])  or is_magenta(gt_map[i][j]) :
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

    path = log['path']

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
    cv2.imwrite('test_costmap.png', costmap)

    # log = {
    #     "map_name": map_path,
    #     "start": start,
    #     "goal": goal,
    #     "path": path,
    #     "costmap": costmap
    # }

    # # save the log to /scratch/luisamao/all_terrain/upper_bound/log_14.pkl
    # with open(f'/scratch/luisamao/all_terrain/upper_bound/log_{i}.pkl', 'wb') as f:
    #     pickle.dump(log, f)

# oracle_plan(14)

preferences = [True, False, True, False, True, False, True, False,
               True, False, True, False, True, False, True, False, True, False,
               True, False, True, False, True, True, False, True]

# for i in range(0+26, 6+26):
#     if i == 7:
#         continue
#     eval(i, preferences[i%26])
#     print("Done with", i, preferences[i%26])
#     print("__________")
# for i in range(8+26, 18+26):
#     if i == 7:
#         continue
#     eval(i, preferences[i%26])
#     print("Done with", i, preferences[i%26])
#     print("__________")
# for i in range(18+26, 26+26):
#     if i == 7:
#         continue
#     eval(i, preferences[i%26])
#     print("Done with", i, preferences[i%26])
#     print("__________")
# for i in range(0, 6):
#     if i == 7:
#         continue
#     eval(i, preferences[i])
#     print("Done with", i)
#     print("__________")
# for i in range(18, 26):
#     if i == 7:
#         continue
#     eval(i, preferences[i])
#     print("Done with", i)
#     print("__________")
for i in range(6, 8):
    eval(i, preferences[i%26])
    print("Done with", i, preferences[i%26])
    print("__________")

