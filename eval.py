import cv2
import numpy as np
import pickle

experiment_dir = "/scratch/luisamao/all_terrain/ablation1/"
experiment_dir2 = "/scratch/luisamao/all_terrain/ablation2/"
experiment_dir3 = "/scratch/luisamao/all_terrain/ablation2.1/"
dirs = [experiment_dir, experiment_dir2, experiment_dir3]


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

# make a file
f1 = open("ablation1.txt", "w")
f2 = open("ablation2.txt", "w")
f3 = open("ablation2.1.txt", "w")
files = [f1, f2, f3]




def eval(log_idx, realistic=True):
    log = pickle.load(open(dirs[0]+f'log_{log_idx}.pkl', 'rb'))
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

    if log_idx == 7 or log_idx == 6 or log_idx == 7+26 or log_idx == 6+26:
        def is_set1(pixel):
            return is_red(pixel) or is_black(pixel) or is_yellow(pixel)
        def is_set2(pixel):
            return is_white(pixel) or is_magenta(pixel)
        
    else:
        def is_set1(pixel):
            return is_red(pixel) or is_black(pixel) or is_yellow(pixel) or is_magenta(pixel)
        def is_set2(pixel):
            return is_white(pixel)

    if realistic:
        # create a 1 channel costmap the same size as the gt_map of ones
        costmap = np.ones((gt_map.shape[0], gt_map.shape[1]), dtype=np.float32)
        # if the gtmap pixel is red, magenta, or black, set the costmap pixel to 0
        for i in range(gt_map.shape[0]):
            for j in range(gt_map.shape[1]):
                if is_set1(gt_map[i][j]):
                    costmap[i][j] = .2
                elif is_set2(gt_map[i][j]) :
                    costmap[i][j] = 0.5

    else:
        # create a 1 channel costmap the same size as the gt_map of ones
        costmap = np.ones((gt_map.shape[0], gt_map.shape[1]), dtype=np.float32) * .2
        # if the gtmap pixel is red, magenta, or black, set the costmap pixel to 0
        for i in range(gt_map.shape[0]):
            for j in range(gt_map.shape[1]):
                if is_set1(gt_map[i][j]):
                    costmap[i][j] = 1
                elif is_set2(gt_map[i][j]):
                    costmap[i][j] = 0.5



    # print start and goal
    # print(start, goal)
    # print the shape of the map
    # print(map.shape)
    costmap *= 255

    for idx in range(3):
        log = pickle.load(open(dirs[idx]+f'log_{log_idx}.pkl', 'rb'))
        # print the name
        # print(dirs[idx]+f'log_{log_idx}.pkl', log['map_name'])
        # print(costmap.shape)
        start = log['start']
        goal = log['goal']
        path = log['path']
        # print("Path length:", len(path))
        # print(start, goal)

        # print the number of pixels that go on low, medium, and high cost
        print("Low cost pixels:", sum([1 for p in path if costmap[p] == .2 * 255]))
        print("Medium cost pixels:", sum([1 for p in path if costmap[p] == 0.5 * 255]))
        print("High cost pixels:", sum([1 for p in path if costmap[p] == 1 * 255]))

        # write to the file
        files[idx].write(f"{sum([1 for p in path if costmap[p] == .2 * 255])}, {sum([1 for p in path if costmap[p] == 0.5 * 255])}, {sum([1 for p in path if costmap[p] == 1 * 255])} \n")
        files[idx].flush()


        # draw the path on the costmap
        # cv2 grayscale to rgb
        costmaprgb = cv2.cvtColor(costmap, cv2.COLOR_GRAY2RGB)
        for p in path:
            # use cvt circle to draw the path
            cv2.circle(costmaprgb, (p[1],p[0]), 5, (0, 0, 255), -1)
        # plot the start and end
        cv2.circle(costmaprgb, (start[1], start[0]), 5, (0, 255, 0), -1)
        cv2.circle(costmaprgb, (goal[1], goal[0]), 5, (0, 255, 0), -1)

        # save the costmap
        cv2.imwrite('test_costmap.png', costmaprgb)


preferences = [True, False, True, False, True, False, True, False,
               True, False, True, False, True, False, True, False, True, False,
               True, False, True, False, True, True, False, True]

for i in range(0, 26):
    eval(i, preferences[i])
    print("Done with", i, preferences[i])
    print("__________")

for i in range(0+26, 26+26):
    eval(i, True)
    print("Done with", i, True)
    print("__________")