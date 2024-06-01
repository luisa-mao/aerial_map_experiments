import pickle
from simulator import Simulator
import os


experiment_dir = "/scratch/luisamao/all_terrain/experiment_logs"

# iterate over the directories and make a list of filenames
filenames = []
for root, dirs, files in os.walk(experiment_dir):
    for file in files:
        if file.endswith(".pkl"):
            filenames.append(os.path.join
                                (root, file))
print(filenames)
print(len(filenames))



# the filenames are log_0.pkl, log_1.pkl, log_2.pkl, etc.
# sort the filenames by the number at the end
filenames.sort(key=lambda x: int(x.split("_")[-1].split(".")[0]))

print(filenames)

with open("contexts/concrete_grass_bush_mulch.pkl", "rb") as f:
    context_tensor = pickle.load(f)
with open("contexts/c1.pkl", "rb") as f:
    context_tensor1 = pickle.load(f)
# print(context_tensor.shape)

def run_simulator(i, context = None):
    log = filenames[i]
    with open(log, "rb") as f:
        log = pickle.load(f)

    # print map name, start, goal, path length, and costmap shape
    print(log["map_name"], log["start"], log["goal"], len(log["path"]), log["costmap"].shape)
    args = {
        "map_path": log["map_name"],
        "mask_path": '../clean_mask.png',
        "scale": 1,
        "contrast": False,
        "log": False,
        "context": context_tensor
    }
    simulator = Simulator(None, args)
    path = simulator.run(log["start"], log["goal"], filename = "log_"+str(i))
    # print the path length
    print(len(path))
    print(log["map_name"])
    print(filenames[i])
    print(i)

# for i in range(0,8):
#     run_simulator(i, context_tensor)
#     print("Done with", i)
#     print("__________")
# for i in range(12,18):
#     run_simulator(i, context_tensor)
#     print("Done with", i)
#     print("__________")
# for i in range(18,26):
#     run_simulator(i, context_tensor)
#     print("Done with", i)
#     print("__________")
run_simulator(5)
run_simulator(7)
