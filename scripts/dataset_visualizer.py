import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from torch.utils.data import DataLoader


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
sys.path.insert(0, PROJECT_ROOT)

from src.data.synthetic_pose_dataset import SyntheticPoseDataset
bone_graph_path = os.path.join(PROJECT_ROOT, "data", "meta", "joints_mapping.json")
with open(bone_graph_path, "r") as f:
    bone_graph = json.load(f)

BONES = bone_graph["edges"]

def visualize_sample(sample):
    for k in ("rgb", "joints_3d_world", "joints_3d_centered", "joints_2d", "joints_2d_mpii", "K", "R", "t"):
        if k in sample:
            v = sample[k]
            sample[k] = v.cpu().numpy() if hasattr(v, "cpu") else v

    img    = sample["rgb"]
    j3d    = sample["joints_3d_world"]
    j3d_centered    = sample["joints_3d_centered"]
    K, R, t= sample["K"], sample["R"], sample["t"]

    cam_pts = (R @ j3d.T) + t.reshape(3, 1)
    proj    = (K @ cam_pts).T
    proj2d  = proj[:, :2] / proj[:, 2:3]

    fig = plt.figure(figsize=(12, 6))

    ax1 = fig.add_subplot(1, 2, 1)
    ax1.imshow(img.astype(np.uint8))
    ax1.axis("off")
    ax1.set_title("3D→2D Projection Check")

    ax1.scatter(proj2d[:, 0], proj2d[:, 1], c="r", s=30, label="proj 3D")

    if "joints_2d_mpii" in sample:
        j2d = sample["joints_2d_mpii"]
        ax1.scatter(j2d[:, 0], j2d[:, 1], c="b", marker="x", s=25, label="GT 2D")
    ax1.legend()

    ax2 = fig.add_subplot(1, 2, 2, projection="3d")
    ax2.set_title("3D Skeleton")

    ax2.scatter(j3d_centered[:, 0], j3d_centered[:, 1], j3d_centered[:, 2], c="k", s=25)

    for parent, child in BONES:
        xs = [j3d_centered[parent, 0], j3d_centered[child, 0]]
        ys = [j3d_centered[parent, 1], j3d_centered[child, 1]]
        zs = [j3d_centered[parent, 2], j3d_centered[child, 2]]
        ax2.plot(xs, ys, zs, c="blue", lw=2)

    for idx, (x, y, z) in enumerate(j3d_centered):
        ax2.text(x, y, z, str(idx), fontsize=8)
    ax2.set_xlabel("X (m)")
    ax2.set_ylabel("Y (m)")
    ax2.set_zlabel("Z (m)")

    plt.tight_layout()
    plt.show()

def main():
    data_root = os.path.join(PROJECT_ROOT, "data")
    split_txt = os.path.join(data_root, "splits", "val.txt")

    ds     = SyntheticPoseDataset(data_root=data_root, split_txt=split_txt)
    loader = DataLoader(ds, batch_size=1, shuffle=True, num_workers=2)

    batch  = next(iter(loader))
    sample = {k: v[0] for k, v in batch.items()}

    visualize_sample(sample)

if __name__ == "__main__":
    main()
