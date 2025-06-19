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

random.seed(42)
random.shuffle(frame_ids)
#split_idx = int(0.8 * len(frame_ids))

#train_ids = frame_ids[:split_idx]
#val_ids   = frame_ids[split_idx:]

train_path = os.path.join(SPLITS_DIR, "train.txt")
#val_path   = os.path.join(SPLITS_DIR, "val.txt")

with open(train_path, "w") as f:
    f.write("\n".join(frame_ids) + "\n")
#with open(val_path, "w") as f:
#    f.write("\n".join(val_ids) + "\n")

print(f"Saved {len(frame_ids)} train IDs → {train_path}")
#print(f"Saved {len(val_ids)} val IDs   → {val_path}")
