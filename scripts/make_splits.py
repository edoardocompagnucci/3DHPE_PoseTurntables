import os
import glob
import random

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "data"))
ANN_DIR = os.path.join(DATA_ROOT, "annotations", "joints_3d")
SPLITS_DIR = os.path.join(DATA_ROOT, "splits")
os.makedirs(SPLITS_DIR, exist_ok=True)

files = glob.glob(os.path.join(ANN_DIR, "*.npy"))
frame_ids = [os.path.splitext(os.path.basename(f))[0] for f in files]

if not frame_ids:
    raise RuntimeError(f"No .npy files found in {ANN_DIR}")

print(f"Total available samples: {len(frame_ids)}")

# Set random seed for reproducibility
random.seed(42)

# Shuffle all frame IDs first
random.shuffle(frame_ids)

# Take only 20,000 samples
reduced_frame_ids = frame_ids[:20000]

# Split 80/20 for train/val
split_idx = int(0.8 * len(reduced_frame_ids))

train_ids = reduced_frame_ids[:split_idx]
val_ids = reduced_frame_ids[split_idx:]

train_path = os.path.join(SPLITS_DIR, "train.txt")
val_path = os.path.join(SPLITS_DIR, "val.txt")

with open(train_path, "w") as f:
    f.write("\n".join(train_ids) + "\n")
with open(val_path, "w") as f:
    f.write("\n".join(val_ids) + "\n")

print(f"Reduced from {len(frame_ids)} to {len(reduced_frame_ids)} samples")
print(f"Saved {len(train_ids)} train IDs → {train_path}")
print(f"Saved {len(val_ids)} val IDs → {val_path}")