import os, sys, json, cv2, torch, numpy as np, matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mmpose.apis import MMPoseInferencer

IMAGE_PATH = r"assets\demoimage.jpg"
CKPT_PATH = r"checkpoints\mlp_lifter_regularized_20250529_083604\final_model.pth"
IMG_SIZE = 512

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
sys.path.insert(0, PROJECT_ROOT)

from src.models.lifter import MLPLifter, MLPLifter_v2
from src.utils.transforms import NormalizerJoints2d


JOINT_MAPPING_JFN = os.path.join(PROJECT_ROOT, "data/meta/joints_mapping.json")

with open(JOINT_MAPPING_JFN, "r") as f:
    joint_map = json.load(f)
    
smpl2mpii = np.array([-1 if v is None else v for v in joint_map["smpl2mpii"]], dtype=np.int16)
smpl_edges = joint_map["edges"]
mpii_edges = joint_map["mpii_edges"]

def pad_mpii_to_smpl(kp_mpii, cof_mpii):
    kp_smpl = np.zeros((24, 2), dtype=np.float32)
    cof_smpl = np.zeros(24, dtype=np.float32)
    for smpl_idx, mpii_idx in enumerate(smpl2mpii):
        if mpii_idx >= 0:
            kp_smpl[smpl_idx] = kp_mpii[mpii_idx]
            cof_smpl[smpl_idx] = cof_mpii[mpii_idx]
    return kp_smpl, cof_smpl

def draw_skeleton(img, kps, conf, edges, thr=0.3):
    for i, (x, y) in enumerate(kps):
        if conf[i] < thr:
            continue
        cv2.circle(img, (int(x), int(y)), 4, (0, 0, 255), -1)
    for p, c in edges:
        if p >= len(kps) or c >= len(kps):
            continue
        if conf[p] < thr or conf[c] < thr:
            continue
        cv2.line(img, (int(kps[p, 0]), int(kps[p, 1])), (int(kps[c, 0]), int(kps[c, 1])), (0, 255, 0), 2)
    return img

def first_person_keypoints(det_result):
    obj = det_result["predictions"]
    while isinstance(obj, list) and len(obj):
        obj = obj[0]
    if isinstance(obj, dict):
        kp16 = np.asarray(obj["keypoints"], dtype=np.float32)
        cof16 = np.asarray(obj["keypoint_scores"], dtype=np.float32)
    else:
        inst = obj.pred_instances
        kp16 = inst.keypoints[0].cpu().numpy().astype(np.float32)
        cof16 = inst.keypoint_scores[0].cpu().numpy().astype(np.float32)
    return kp16, cof16

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pose_detector = MMPoseInferencer(pose2d="rtmpose-m_8xb64-210e_mpii-256x256", device=device.type)
lifter = MLPLifter_v2(num_joints=24, dropout_rate=0.25).to(device)
lifter.load_state_dict(torch.load(CKPT_PATH, map_location=device)["model_state"])
lifter.eval()
normaliser = NormalizerJoints2d(img_size=IMG_SIZE)

img_bgr = cv2.imread(IMAGE_PATH)
assert img_bgr is not None, f"cannot read {IMAGE_PATH}"
img_bgr = cv2.resize(img_bgr, (IMG_SIZE, IMG_SIZE))

det_result = next(pose_detector(img_bgr, show=False))
kp16, cof16 = first_person_keypoints(det_result)
kp24, cof24 = pad_mpii_to_smpl(kp16, cof16)

kp_tensor = torch.from_numpy(kp24).unsqueeze(0)
kp_norm = normaliser({"joints_2d": kp_tensor})["joints_2d"]
with torch.no_grad():
    pose3d = lifter(kp_norm.to(device)).cpu().numpy().reshape(24, 3)

overlay = draw_skeleton(img_bgr.copy(), kp16, cof16, mpii_edges)

fig = plt.figure(figsize=(20, 5))

# 2-D overlay, cell 1 of a 1×2 grid
ax_img = fig.add_subplot(1, 2, 1)
ax_img.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
ax_img.set_title("2-D overlay")
ax_img.axis("off")

# 3-D view, cell 2 of the same 1×2 grid
ax = fig.add_subplot(1, 2, 2, projection="3d")
ax.set_title("front view")

X, Y, Z = pose3d[:, 0], pose3d[:, 1], pose3d[:, 2]
ax.scatter(X, Y, Z, s=20)
for p, c in smpl_edges:
    ax.plot([X[p], X[c]], [Y[p], Y[c]], [Z[p], Z[c]], "b-", lw=2)
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.view_init(elev=5, azim=-85)

plt.show()