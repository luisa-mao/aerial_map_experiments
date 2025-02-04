import sys
import os
from simulator import Simulator
import pickle as pkl

# Add the directory containing simulator.py to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

# open log_14 from /scratch/luisamao/all_terrain/experiment_logs
with open('/scratch/luisamao/all_terrain/experiment_logs/log_14.pkl', 'rb') as f:
    log_14 = pkl.load(f)
    print(log_14.keys())
args = {
    "map_path": log_14['map_name'],
    "mask_path": '../clean_mask.png',
    "scale": 1,
    "contrast": False,
    "log": False,
    "context": None
}
simulator = Simulator(None, args)
simulator.image_to_costmap(log_14['context_tensor'])
# simulator.run(start = log_14['start'], goal = log_14['goal'], context_tensor = log_14['context_tensor'])