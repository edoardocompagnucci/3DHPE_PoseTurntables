# evaluate_3dpw_fixed.py
"""
evaluate_3dpw_fixed
=================================
Single‑image evaluation on 3DPW **without any argparse/CLI code**.

Usage (from anywhere in your project)
-------------------------------------
```python
from evaluate_3dpw_fixed import evaluate_3dpw
stats = evaluate_3dpw('/datasets/3DPW', split='validation', workers=12)
print(stats)  # {'mpjpe_mm': 54.7, 'pa_mpjpe_mm': 38.3}
```

* Evaluates **every frame** of the requested split, discovering images by folder, not by .pkl path list.
* Computes **MPJPE** and **PA‑MPJPE** (mm, pelvis‑centred).
* Models (MMPose detector + MLP lifter) are loaded **once globally**.
* Thread pool is used only for JPEG decoding; network forward passes happen in the main thread (safe for CUDA).
"""
from __future__ import annotations

import pickle
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import torch
from tqdm import tqdm
from scipy.linalg import orthogonal_procrustes
from mmpose.apis import MMPoseInferencer

# Project imports (adjust path if needed)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
import sys; sys.path.insert(0, str(PROJECT_ROOT))
from src.models.rot_head import MLPLifterRotationHead
from src.utils.transforms import NormalizerJoints2d

# Constants
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
IMG_SIZE = 512
DETECTOR_CFG = 'rtmpose-m_8xb64-210e_mpii-256x256'

# Load models
detector = MMPoseInferencer(pose2d=DETECTOR_CFG, device=device.type)
lifter = MLPLifterRotationHead(num_joints=24, dropout=0.1).to(device)
ckpt = PROJECT_ROOT / 'checkpoints' / 'mlp_lifter_rotation_poseaug_20250609_224512' / 'final_model.pth'
lifter.load_state_dict(torch.load(ckpt, map_location=device)['model_state'])
lifter.eval()
normalizer = NormalizerJoints2d(img_size=IMG_SIZE)

# MPII->SMPL mapping
MPII_TO_SMPL = [6,3,2,None,4,1,None,5,0,7,None,None,8,None,None,9,13,12,14,11,15,10,None,None]

# Helpers
def square_resize(img: np.ndarray, size: int=IMG_SIZE) -> np.ndarray:
    h, w = img.shape[:2]
    pad = max(h, w)
    canvas = np.zeros((pad, pad, 3), dtype=img.dtype)
    y0, x0 = (pad-h)//2, (pad-w)//2
    canvas[y0:y0+h, x0:x0+w] = img
    return cv2.resize(canvas, (size, size), cv2.INTER_LINEAR)

def map_mpii_to_smpl(kp16: np.ndarray) -> np.ndarray:
    kp24 = np.zeros((24,2), dtype=np.float32)
    for i, m in enumerate(MPII_TO_SMPL):
        if m is not None and m < len(kp16):
            kp24[i] = kp16[m]
    return kp24

def detect_keypoints(img: np.ndarray) -> np.ndarray | None:
    inp = square_resize(img)
    result = next(detector(inp, show=False))
    preds = result['predictions']
    while isinstance(preds, list): preds = preds[0]
    if isinstance(preds, dict):
        return np.asarray(preds['keypoints'], dtype=np.float32)
    return preds.pred_instances.keypoints[0].cpu().numpy().astype(np.float32)

@torch.no_grad()
def predict_joints(img: np.ndarray) -> np.ndarray | None:
    kp16 = detect_keypoints(img)
    if kp16 is None: return None
    kp24 = map_mpii_to_smpl(kp16)
    tensor2d = torch.from_numpy(kp24).unsqueeze(0)
    normed = normalizer({'joints_2d': tensor2d})['joints_2d'].to(device)
    p3_flat, _ = lifter(normed)
    p3 = p3_flat.view(24,3).cpu().numpy()
    # root-centre
    p3 = p3 - p3[0]
    # TEST: flip the x-axis
    p3[:,2] *= -1
    return p3


# Metrics
def procrustes(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a_c = a - a.mean(0)
    b_c = b - b.mean(0)
    R, _ = orthogonal_procrustes(b_c, a_c)
    return (b_c @ R) + a.mean(0)

def mpjpe(pred: np.ndarray, gt: np.ndarray) -> float:
    return np.linalg.norm(pred-gt, axis=1).mean()*1000

def pa_mpjpe(pred: np.ndarray, gt: np.ndarray) -> float:
    return mpjpe(procrustes(gt, pred), gt)

# Evaluation
def load_joints(pkl: Path) -> np.ndarray:
    data = pickle.load(open(pkl,'rb'), encoding='latin1', fix_imports=True)
    joints = data.get('jointPositions') or data.get('joints3d') or data.get('J') or data.get('joints')
    return np.asarray(joints).reshape(-1,24,3)

def evaluate_3dpw(root: str|Path, split: str='validation', workers: int=8) -> dict[str,float]:
    root = Path(root)
    seq_dir = root/'sequenceFiles'/'sequenceFiles'/split
    pkls = sorted(seq_dir.glob('*.pkl'))
    total_mp=0.0; total_pa=0.0; cnt=0
    pool = ThreadPoolExecutor(workers) if workers>0 else None
    for pkl in pkls:
        gt_seq = load_joints(pkl)
        folder = root/'imageFiles'/'imageFiles'/pkl.stem
        imgs = sorted([p for p in folder.iterdir() if p.suffix.lower() in ['.jpg','.png']])
        for gt24, img_path in zip(gt_seq, imgs):
            img = pool.submit(cv2.imread, str(img_path)).result() if pool else cv2.imread(str(img_path))
            if img is None: continue
            pred = predict_joints(img)
            if pred is None: continue
            gt_c = gt24 - gt24[0]
            total_mp += mpjpe(pred, gt_c)
            total_pa += pa_mpjpe(pred, gt_c)
            cnt +=1
    if pool: pool.shutdown()
    stats={'mpjpe_mm': total_mp/cnt, 'pa_mpjpe_mm': total_pa/cnt}
    print(f"3DPW {split.upper()} frames={cnt}: MPJPE={stats['mpjpe_mm']:.2f} mm, PA-MPJPE={stats['pa_mpjpe_mm']:.2f} mm")
    return stats


# -----------------------------------------------------------------------------
# Optional: run directly (edit the path before using!)
# -----------------------------------------------------------------------------
if __name__ == '__main__':
    # MODIFY THIS PATH TO YOUR DATASET LOCATION OR COMMENT OUT
    DATASET_ROOT = r'C:\Users\Drako\Downloads\3DPW'  # <- change me or comment
    if Path(DATASET_ROOT).exists():
        evaluate_3dpw(DATASET_ROOT, split='validation', workers=12)
    else:
        print('Please edit DATASET_ROOT before running as a script.')
