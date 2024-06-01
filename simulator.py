import argparse
import cv2
import numpy as np
import sys
import math
import pickle as pkl
import heapq
import os
from sim_utils import increase_contrast, heuristic, histogram_mode_separation

sys.path.append("..")
# from bev_to_costmap import bev_img_to_tensor, bev_to_costmap_w_context, get_context_embedding
from utils import patches_to_tensor

# sys.path.append("/home/luisamao/classifier_baseline/")
# from scripts.local_costmap import image_to_costmap
sys.path.append("/home/luisamao/sterling/")
from costmap_scripts.local_costmap import image_to_costmap


class Simulator:
    def __init__(self, command_line_args, gui_args = None):
        if gui_args is not None:
            self.map_name = gui_args["map_path"]
            # self.robot_height = gui_args["robot_height"]
            # self.drone_height = gui_args["drone_height"]
            self.scale = gui_args["scale"]
            self.log = gui_args["log"]
            self.contrast = gui_args["contrast"]
            if self.log:
                self.create_log_dirs()
            self.mask = None
            self.aerial_map = None
            self.context = gui_args["context"]
            self.load_map_and_mask(gui_args["mask_path"], gui_args["map_path"])
            self.known_map = None
            self.full_costmap = None

        else:
            self.map_name = command_line_args["map_path"]
            self.robot_height = command_line_args["robot_height"]
            self.drone_height = command_line_args["drone_height"]
            self.scale = command_line_args["scale"]
            self.log = command_line_args["log"]
            self.contrast = command_line_args["contrast"]
            if self.log:
                self.create_log_dirs()
            self.mask = None
            self.aerial_map = None
            self.context = None
            self.load_map_and_mask(command_line_args["mask_path"], command_line_args["map_path"])
            self.load_context(command_line_args["context_path"])
            self.known_map = None
            self.full_costmap = None
    
    def create_log_dirs(self):
        os.makedirs("logs", exist_ok=True)
        os.makedirs("logs/views", exist_ok=True)
        os.makedirs("logs/costmaps", exist_ok=True)

    def load_context(self, context_path):
        with open(context_path, 'rb') as f:
            self.context = pkl.load(f)

    def load_map_and_mask(self, mask_path, map_path):
        # load the mask as an np array
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        # resize the mask to 256x128
        mask = cv2.resize(mask, (256, 128))
        self.mask = np.array(mask)[:106, :]
        aerial_map = cv2.imread(map_path)
        # resize 4x smaller
        s = self.scale
        h, w = aerial_map.shape[:2]
        self.aerial_map = cv2.resize(aerial_map, (int(w / s), int(h / s)))
        # h, w = h // 4, w // 4

    '''
    very very simple function to get the robot's bev view given its position
    '''
    def get_view(self, x, y, yaw):
        # Create a copy of the aerial_map
        aerial_map_copy = self.aerial_map.copy()

        # Define the patch size
        patch_size = self.mask.shape[:2]

        # Rotate the copy of the aerial_map
        M = cv2.getRotationMatrix2D((x, y), 90-yaw, 1)
        rotated_map = cv2.warpAffine(aerial_map_copy, M, (aerial_map_copy.shape[1], aerial_map_copy.shape[0]))

        # Extract the patch
        start_y = int(max(y - patch_size[0], 0))
        end_y = int(min(y, rotated_map.shape[0]))
        start_x = int(max(x - patch_size[1] // 2, 0))
        end_x = int(min(x + patch_size[1] // 2, rotated_map.shape[1]))
        patch = rotated_map[start_y:end_y, start_x:end_x]

        sx = int(max(0-(x - patch_size[1] // 2), 0))
        ex = int(min(self.mask.shape[1] - ((x + patch_size[1] // 2) - rotated_map.shape[1]), self.mask.shape[1]))
        sy = int(max(0-(y - patch_size[0]), 0))
        ey = int(min(self.mask.shape[0] - (y - rotated_map.shape[0]), self.mask.shape[0]))

        for i in range(sx, ex):
            for j in range(sy, ey):
                if self.mask[j][i] == 0:
                    patch[j-sy][i-sx] = [0,0,0]

        padding = ((sy-0, self.mask.shape[0]-ey),(sx-0, self.mask.shape[1]-ex),(0,0))
        patch = np.pad(patch, padding, mode='constant', constant_values=0)
        return patch
    
    '''
    plots "known bev" area on the known map and plots the local self.full_costmap on the full self.full_costmap
    '''
    def expand_known_map(self, local_costmap, x,y, yaw):
        # facing right is 0 degrees
        yaw -= 90
        # enforce angles betwen -180 and 180
        yaw = yaw % 360
        if yaw > 180:
            yaw = yaw - 360

        flip = False
        # if yaw is outside -90, 90, x flip and y flip needed
        if abs(yaw) > 90:
            # yaw += (-90 if yaw > 0 else 90)
            if yaw > 0:
                yaw = yaw - 180
            else:
                yaw = yaw + 180
            flip = True

        # convert yaw from degrees to radians
        yaw = math.radians(yaw)

        # Calculate the shear values
        shear_x = -np.tan(yaw / 2)
        shear_y = np.sin(yaw)

        # create an x and y flip matrix
        flip_xy = np.float32([[-1, 0], [0, -1]])

        # Create the shear matrices
        M1 = np.float32([[1, shear_x], [0, 1]])
        M2 = np.float32([[1, 0], [shear_y, 1]])
        M3 = np.float32([[1, shear_x], [0, 1]])

        for j in range(self.mask.shape[0]):
            for i in range(self.mask.shape[1]):
                if self.mask[j][i] > 0:
                    # Calculate the relative coordinates
                    r = j - self.mask.shape[0] 
                    c = i - self.mask.shape[1] / 2

                    pixel = [r, c]
                    if flip:
                        # apply the flip matrix
                        pixel = np.dot(flip_xy, pixel)
                        pixel = pixel.astype(int)
                    
                    # apply the three shears to the pixel
                    pixel = np.dot(M1, pixel)
                    pixel = pixel.astype(int)
                    pixel = np.dot(M2, pixel)
                    pixel = pixel.astype(int)
                    pixel = np.dot(M3, pixel)
                    pixel = pixel.astype(int)
                    pixel  = [pixel[0]+y, pixel[1]+x]

                    if pixel[0] >= 0 and pixel[0] < self.aerial_map.shape[0] and pixel[1] >=0 and pixel[1] < self.aerial_map.shape[1]:
                        # transform the pixel and draw it on the known map
                        self.known_map[pixel[0], pixel[1]] = 255
                        if local_costmap[j][i] > 0:
                            self.full_costmap[pixel[0], pixel[1]] = local_costmap[j][i]


    def astar(self, start, goal):        
        count = 0
        neighbors = [(0,1),(1,0),(-1,0),(0,-1),(1,1),(-1,-1),(1,-1),(-1,1)]
        neighbors = [(i*5, j*5) for (i,j) in neighbors]
        yaws = [0, -90, -270, 180, -45, (90+45), -(90+45), 45]
        
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
                # debugging
                print(len(path))
                print("Cost of the path:", sum(self.full_costmap[p[0], p[1]] for p in path))
                print(f_score[path[0]])

                # path2 = [(384, i) for i in range(448, 1024, 5)]
                # print("Cost of the path:", sum(self.full_costmap[p[0], p[1]] for p in path2))
                # print(len(path2))

                # # print the distances between the start and goal
                # print("Distance between start and goal:", math.sqrt((start[0] - goal[0])**2 + (start[1] - goal[1])**2))



                return path[::-1]

            for idx, (i, j) in enumerate(neighbors):
                neighbor = current[0] + i, current[1] + j

                if 0 <= neighbor[0] < self.full_costmap.shape[0]:
                    if 0 <= neighbor[1] < self.full_costmap.shape[1]:                
                        if self.known_map[neighbor] == 0:
                            count += 1
                            x = current[1]
                            y = current[0]
                            yaw = yaws[idx]
                            view = self.get_view(x, y, yaw)
                            # add rows of zeros to get 256 rows
                            padding = ((0, 22), (0, 0), (0,0))
                            view = np.pad(view, padding, mode='constant', constant_values=0)
                            # bev_tensor = bev_img_to_tensor(view)

                            # print the shape of bev_tensor and self.context
                            # print(bev_tensor.shape, self.context.shape)

                            # out = bev_to_costmap_w_context(bev_tensor, self.context)[:106,:]
                            
                            # reshape view to 749 x 1476
                            view = cv2.resize(view, (1476, 749))
                            out = image_to_costmap(view)
                            # resize view back down again
                            # view = cv2.resize(view, (256, 128))
                            # resize to 256 x 128
                            out = cv2.resize(out, (256, 128))[:106,:]
                            if self.contrast and False:
                                # out = increase_contrast(out, self.mask)
                                out = histogram_mode_separation(out, self.mask)
                            out *= 255
                            self.expand_known_map(out, x,y,yaw)

                            if self.log:
                                cv2.imwrite( f"logs/views/view_{count}.png", view)
                                cv2.imwrite(f"logs/costmaps/costmap_{count}.png", self.full_costmap)
                    else:
                        continue
                else:
                    continue
                
                tentative_g_score = g_score[current] + math.dist(current, neighbor) * (self.full_costmap[neighbor] / 255.0) * 5
                    
                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = tentative_g_score + heuristic(neighbor, goal)
                    heapq.heappush(open_heap, (f_score[neighbor], neighbor))
        return []


    # start and goal are in column, row order
    def run(self, start=(320,440), goal=(192, 572), patch_centers = None, filename = None):
        context_tensor = None
        PATCH_WIDTH = 32
        patches = []
        if patch_centers is not None:
            for i in range(len(patch_centers)):
                x_min = patch_centers[i][0] - PATCH_WIDTH // 2
                x_max = patch_centers[i][0] + PATCH_WIDTH // 2
                y_min = patch_centers[i][1] - PATCH_WIDTH // 2
                y_max = patch_centers[i][1] + PATCH_WIDTH // 2

                patch = self.aerial_map[y_min:y_max, x_min:x_max]
                # change each patch from bgr to rgb
                patches.append(patch)
            context_tensor = patches_to_tensor(patches)
            self.context = get_context_embedding(context_tensor)
            print("set new self.context", self.context.shape)

            # # save the patches
            # from torchvision.utils import make_grid
            # import matplotlib.pyplot as plt
            # context = self.context.reshape(3, 3, 64, 32)
            # grid = make_grid(context, normalize=True, nrow=3)
            # grid = grid.permute(1, 2, 0).numpy()
            # plt.imshow(grid)
            # plt.savefig("patches.png")
            # exit()
                


        # print start and end
        print("Start:", start)
        print("Goal:", goal)

        self.known_map = np.zeros(self.aerial_map.shape[:2])
        self.full_costmap = np.zeros(self.aerial_map.shape[:2])
        path = self.astar(start, goal)
        # if self.log:
        cost_map = np.stack([self.full_costmap, self.full_costmap, self.full_costmap], axis=-1)
        for p in path:
            cv2.circle(cost_map, (p[1], p[0]), 2, (0, 0, 255), -1)
        # for p in path2:
        #     cv2.circle(cost_map, (p[1], p[0]), 2, (0,255, 0), -1)

        # cv2 circle plots it backwards
        cv2.circle(cost_map, (start[1], start[0]), 3, (0, 255, 0), -1)
        cv2.circle(cost_map, (goal[1],goal[0]), 3, (255, 0, 0), -1)
        cv2.imwrite("full_costmap.png", cost_map)
        print("saved to:", "full_costmap.png")


        # write to a log file
        log_dict = {
            "map_name": self.map_name,
            "context_tensor": context_tensor,
            "start": start,
            "goal": goal,
            "path": path,
            "costmap": cost_map,
        }

        # pickle this file
        directory = "/scratch/luisamao/all_terrain/sterling_baseline/"
        # count the number of files in this dir
        num_files = len(os.listdir(directory))+8 
        # pickle the file
        # with open(f"{directory}log_{num_files}.pkl", 'wb') as f:
        with open(f"{directory}{filename}.pkl", 'wb') as f:
            pkl.dump(log_dict, f)


        return path


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--robot_height', type=float, default=0)
    parser.add_argument('--drone_height', type=float, default=0)
    parser.add_argument('--scale', type=float, default=5)
    parser.add_argument('--mask_path', type=str, default='../clean_mask.png')
    parser.add_argument('--map_path', type=str, default='maps/DJI_0455.JPG')
    parser.add_argument('--context_path', type=str, default='contexts/concrete_grass_bush_mulch.pkl')
    parser.add_argument('--contrast', type=bool, default=True)
    parser.add_argument('--log', type=bool, default=False)
    args = parser.parse_args()

    command_line_args = {
        "robot_height": args.robot_height,
        "drone_height": args.drone_height,
        "scale": args.scale,
        "mask_path": args.mask_path,
        "map_path": args.map_path,
        "context_path": args.context_path,
        "contrast": args.contrast,
        "log": args.log
    }

    simulator = Simulator(command_line_args, gui_args = None)
    simulator.run((189, 450), (177, 1026))
    # simulator.run((384, 448), (384, 1024))
    # simulator.run((128, 512), (512, 64))
    # simulator.run((704, 256), (448, 256))

