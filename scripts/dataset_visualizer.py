import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from torch.utils.data import DataLoader

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
sys.path.insert(0, PROJECT_ROOT)

from src.data.synthetic_pose_dataset import SyntheticPoseDataset

def visualize_sample(sample):
    print(sample)
    img    = sample["rgb"]
    j3d    = sample["joints_3d"]
    K, R, t= sample["K"], sample["R"], sample["t"]

    for arr_name in ["rgb", "joints_3d", "joints_2d", "K", "R", "t"]:
        v = sample.get(arr_name)
        if v is None: continue
        if hasattr(v, "cpu"):
            sample[arr_name] = v.cpu().numpy()

    img = sample["rgb"]
    j3d = sample["joints_3d"]
    K, R, t = sample["K"], sample["R"], sample["t"]

    cam_pts = (R @ j3d.T) + t.reshape(3, 1)
    proj    = (K @ cam_pts).T
    proj2d  = proj[:, :2] / proj[:, 2:3]

    plt.figure(figsize=(8, 8 * img.shape[0] / img.shape[1]))
    plt.imshow(img.astype(np.uint8))
    plt.scatter(proj2d[:, 0], proj2d[:, 1], c="r", s=30, label="proj 3D")
    if "joints_2d" in sample:
        j2d = sample["joints_2d"]
        plt.scatter(j2d[:, 0], j2d[:, 1], c="b", marker="x", s=25, label="GT 2D")
    plt.axis("off")
    plt.legend()
    plt.title("3D→2D Projection check")
    plt.show()

    fig = plt.figure(figsize=(6, 6))
    ax  = fig.add_subplot(111, projection="3d")
    ax.scatter(j3d[:, 0], j3d[:, 1], j3d[:, 2], s=25)
    for idx, (x, y, z) in enumerate(j3d):
        ax.text(x, y, z, str(idx), fontsize=8)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_title("3D Joint Positions")
    plt.show()

def main():
    data_root = os.path.join(PROJECT_ROOT, "data")
    split_txt = os.path.join(data_root, "splits", "val.txt")

    ds = SyntheticPoseDataset(data_root=data_root, split_txt=split_txt)

    loader = DataLoader(ds, batch_size=1, shuffle=True, num_workers=2)
    batch  = next(iter(loader))

    sample = {k: v[0] for k, v in batch.items()}

    visualize_sample(sample)

if __name__ == "__main__":
    main()
