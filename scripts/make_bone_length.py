import os
import numpy as np
import json
from glob import glob

with open("data/meta/joints_mapping.json") as f:
    bone_graph = json.load(f)
edges = bone_graph["edges"]

j3d_files = glob("data/annotations/joints_3d/*.npy")

bone_lengths_all = []

for f in j3d_files:
    joints = np.load(f)  # (J,3)
    lengths = [np.linalg.norm(joints[a] - joints[b])
               for (a, b) in edges]
    bone_lengths_all.append(lengths)

bone_lengths_all = np.array(bone_lengths_all)
bone_lengths = bone_lengths_all.mean(axis=0)

os.makedirs("data/meta", exist_ok=True)
np.save("data/meta/bone_lengths.npy", bone_lengths)

print(f"Saved bone_lengths.npy with {len(bone_lengths)} bones.")
