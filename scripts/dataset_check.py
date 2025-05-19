import os, numpy as np, matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from PIL import Image
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "data"))
FRAME_ID  = "rp_carla_rigged_001_FBX_hdr0_camera30_pos10"

paths = {
    "rgb": os.path.join(DATA_ROOT, "raw", "rgb",          f"{FRAME_ID}.png"),
    "j3d": os.path.join(DATA_ROOT, "annotations", "joints_3d", f"{FRAME_ID}.npy"),
    "j2d": os.path.join(DATA_ROOT, "annotations", "joints_2d", f"{FRAME_ID}.npy"),
    "K":   os.path.join(DATA_ROOT, "annotations", "K",         f"{FRAME_ID}.npy"),
    "R":   os.path.join(DATA_ROOT, "annotations", "R",         f"{FRAME_ID}.npy"),
    "t":   os.path.join(DATA_ROOT, "annotations", "t",         f"{FRAME_ID}.npy"),
}

img  = np.array(Image.open(paths["rgb"]))
j3d  = np.load(paths["j3d"])
K    = np.load(paths["K"])
R    = np.load(paths["R"])
t    = np.load(paths["t"])

cam_pts = (R @ j3d.T + t.reshape(3, 1))       # (3,J)
proj_2d = (K @ cam_pts).T
proj_2d = proj_2d[:, :2] / proj_2d[:, 2:3]

plt.figure(figsize=(8, 8 * img.shape[0] / img.shape[1]))
plt.imshow(img)
plt.scatter(proj_2d[:, 0], proj_2d[:, 1], s=30, label="projected 3-D")

if os.path.exists(paths["j2d"]):
    j2d = np.load(paths["j2d"])
    plt.scatter(j2d[:, 0], j2d[:, 1], marker="x", s=25, label="stored 2-D")
plt.axis("off")
plt.title(f"{FRAME_ID} – 2-D projection check")
plt.legend()
plt.show()

fig = plt.figure(figsize=(6, 6))
ax = fig.add_subplot(111, projection="3d")
ax.scatter(j3d[:, 0], j3d[:, 1], j3d[:, 2], s=25)
for i, (x, y, z) in enumerate(j3d):
    ax.text(x, y, z, str(i), fontsize=8)
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.set_title(f"{FRAME_ID} – 3-D joint positions")
plt.show()

