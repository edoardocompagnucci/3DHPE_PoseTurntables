import os
import sys
import random
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
sys.path.insert(0, PROJECT_ROOT)

from data.mixed_pose_dataset import create_mixed_dataset

SMPL_SKELETON = [
    (0, 1), (0, 2), (0, 3),  # Root to hips and spine
    (1, 4), (2, 5), (3, 6),  # Hips to knees, spine to spine1
    (4, 7), (5, 8), (6, 9),  # Knees to ankles, spine1 to spine2
    (7, 10), (8, 11), (9, 12),  # Ankles to feet, spine2 to neck
    (9, 13), (9, 14),          # Neck to shoulders
    (12, 15),                  # Neck to head
    (13, 16), (14, 17),        # Shoulders to elbows
    (16, 18), (17, 19),        # Elbows to wrists
    (18, 20), (19, 21),        # Wrists to hands
    (20, 22), (21, 23),        # Hands to finger tips
]

# ---------------------------------------------------------------------------
# Helper functions – 2‑D and 3‑D plotting
# ---------------------------------------------------------------------------

def plot_2d_keypoints(ax, keypoints, title, pixel_space=True, image_size=512):
    """Scatter‑plot keypoints (and skeleton) in 2‑D space."""
    ax.clear()

    if pixel_space:
        # Pixel coordinates (origin top‑left)
        ax.scatter(keypoints[:, 0], keypoints[:, 1], c="red", s=50, zorder=2)

        for parent, child in SMPL_SKELETON:
            if parent < len(keypoints) and child < len(keypoints):
                ax.plot(
                    [keypoints[parent, 0], keypoints[child, 0]],
                    [keypoints[parent, 1], keypoints[child, 1]],
                    "b-", linewidth=1.5, zorder=1,
                )

        ax.set_xlim(0, image_size)
        ax.set_ylim(image_size, 0)  # invert y‑axis
        ax.set_aspect("equal")

    else:  # normalised 0‑1 coordinates
        keypoints_norm = keypoints / image_size
        ax.scatter(keypoints_norm[:, 0], keypoints_norm[:, 1], c="red", s=50, zorder=2)

        for parent, child in SMPL_SKELETON:
            if parent < len(keypoints_norm) and child < len(keypoints_norm):
                ax.plot(
                    [keypoints_norm[parent, 0], keypoints_norm[child, 0]],
                    [keypoints_norm[parent, 1], keypoints_norm[child, 1]],
                    "b-", linewidth=1.5, zorder=1,
                )

        ax.set_xlim(0, 1)
        ax.set_ylim(1, 0)
        ax.set_aspect("equal")

    ax.set_title(title)
    ax.grid(True, alpha=0.3)


def plot_3d_skeleton(ax, joints_3d, title):
    """Render a SMPL skeleton in 3‑D."""
    ax.clear()

    ax.scatter(joints_3d[:, 0], joints_3d[:, 1], joints_3d[:, 2], c="red", s=50, zorder=2)

    for parent, child in SMPL_SKELETON:
        if parent < len(joints_3d) and child < len(joints_3d):
            ax.plot(
                [joints_3d[parent, 0], joints_3d[child, 0]],
                [joints_3d[parent, 1], joints_3d[child, 1]],
                [joints_3d[parent, 2], joints_3d[child, 2]],
                "b-", linewidth=1.5, zorder=1,
            )

    # Equal aspect ratio for all axes
    max_range = (
        np.array(
            [
                joints_3d[:, 0].max() - joints_3d[:, 0].min(),
                joints_3d[:, 1].max() - joints_3d[:, 1].min(),
                joints_3d[:, 2].max() - joints_3d[:, 2].min(),
            ]
        ).max()
        / 2.0
    )

    mid_x = (joints_3d[:, 0].max() + joints_3d[:, 0].min()) * 0.5
    mid_y = (joints_3d[:, 1].max() + joints_3d[:, 1].min()) * 0.5
    mid_z = (joints_3d[:, 2].max() + joints_3d[:, 2].min()) * 0.5

    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(title)

# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main():
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # 1. Data root – edit this if your data lives elsewhere.
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    DATA_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "data"))

    if not os.path.exists(DATA_ROOT):
        print(f"Error: Data root path does not exist: {DATA_ROOT}")
        print("Please update the DATA_ROOT variable to point to your data directory.")
        return

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # 2. Build the combined dataset (100 % synthetic + 100 % real)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    dataset = create_mixed_dataset(
        data_root=DATA_ROOT,
        split="train",
        synthetic_ratio=1.0,
        real_ratio=1.0,
    )

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # 3. Pick *random* samples so every run looks different
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    synthetic_indices = [idx for idx in range(len(dataset)) if dataset[idx]["data_type"] == "synthetic"]
    real_indices = [idx for idx in range(len(dataset)) if dataset[idx]["data_type"] == "real"]

    if not synthetic_indices or not real_indices:
        print("Could not find both synthetic and real samples!")
        return

    synthetic_sample = dataset[random.choice(synthetic_indices)]
    real_sample = dataset[random.choice(real_indices)]

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # 4. Plotting – same as before, only the data changes
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    fig = plt.figure(figsize=(15, 10))

    # ----- Real sample ------------------------------------------------
    ax1 = fig.add_subplot(2, 3, 1)
    plot_2d_keypoints(ax1, real_sample["joints_2d"].numpy(), "2‑D Keypoints (Pixel) – Real", pixel_space=True)

    ax2 = fig.add_subplot(2, 3, 2)
    plot_2d_keypoints(ax2, real_sample["joints_2d"].numpy(), "2‑D Keypoints (Norm.) – Real", pixel_space=False)

    ax3 = fig.add_subplot(2, 3, 3, projection="3d")
    plot_3d_skeleton(ax3, real_sample["joints_3d_centered"].numpy(), "3‑D Skeleton – Real")

    # ----- Synthetic sample ------------------------------------------
    ax4 = fig.add_subplot(2, 3, 4)
    plot_2d_keypoints(ax4, synthetic_sample["joints_2d"].numpy(), "2‑D Keypoints (Pixel) – Synthetic", pixel_space=True)

    ax5 = fig.add_subplot(2, 3, 5)
    plot_2d_keypoints(ax5, synthetic_sample["joints_2d"].numpy(), "2‑D Keypoints (Norm.) – Synthetic", pixel_space=False)

    ax6 = fig.add_subplot(2, 3, 6, projection="3d")
    plot_3d_skeleton(ax6, synthetic_sample["joints_3d_centered"].numpy(), "3‑D Skeleton – Synthetic")

    plt.tight_layout()
    plt.show()

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # 5. Console summary (optional but handy for debugging)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    print("\nSample Information:")

    print("Real sample – data type:", real_sample["data_type"])
    print("  joints_2d shape:", real_sample["joints_2d"].shape)
    print("  joints_3d_centered shape:", real_sample["joints_3d_centered"].shape)
    print("  rot_mats shape:", real_sample["rot_mats"].shape)
    print("  rot_6d shape:", real_sample["rot_6d"].shape)

    print("\nSynthetic sample – data type:", synthetic_sample["data_type"])
    print("  joints_2d shape:", synthetic_sample["joints_2d"].shape)
    print("  joints_3d_centered shape:", synthetic_sample["joints_3d_centered"].shape)
    print("  rot_mats shape:", synthetic_sample["rot_mats"].shape)
    print("  rot_6d shape:", synthetic_sample["rot_6d"].shape)

# ---------------------------------------------------------------------------
# Run as a script                                                               
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    main()
