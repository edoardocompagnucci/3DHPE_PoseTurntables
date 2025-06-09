import os, sys, json, cv2, torch, numpy as np, matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mmpose.apis import MMPoseInferencer

IMAGE_PATH = r"assets\demo_images\Lorenzo\1000067779.jpg"
CKPT_PATH = r"checkpoints\mlp_lifter_rotation_20250609_133841\final_model.pth"
IMG_SIZE = 512
OUTPUT_DIR = "outputs/inference"

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
sys.path.insert(0, PROJECT_ROOT)

from src.models.rot_head import MLPLifterRotationHead
from src.utils.transforms import NormalizerJoints2d

JOINT_MAPPING_PATH = os.path.join(PROJECT_ROOT, "data/meta/joints_mapping.json")
with open(JOINT_MAPPING_PATH, "r") as f:
    joint_map = json.load(f)

smpl2mpii = np.array([-1 if v is None else v for v in joint_map["smpl2mpii"]], dtype=np.int16)
smpl_edges = joint_map["edges"]
mpii_edges = joint_map["mpii_edges"]

def preprocess_image(image_path, target_size=512):
    img = cv2.imread(image_path)
    h, w = img.shape[:2]
    max_dim = max(h, w)
    square_img = np.zeros((max_dim, max_dim, 3), dtype=img.dtype)
    y_offset = (max_dim - h) // 2
    x_offset = (max_dim - w) // 2
    square_img[y_offset:y_offset+h, x_offset:x_offset+w] = img
    resized_img = cv2.resize(square_img, (target_size, target_size))
    return resized_img

def pad_mpii_to_smpl(kp_mpii, cof_mpii):
    kp_smpl = np.zeros((24, 2), dtype=np.float32)
    cof_smpl = np.zeros(24, dtype=np.float32)
    for smpl_idx, mpii_idx in enumerate(smpl2mpii):
        if mpii_idx >= 0:
            kp_smpl[smpl_idx] = kp_mpii[mpii_idx]
            cof_smpl[smpl_idx] = cof_mpii[mpii_idx]
    return kp_smpl, cof_smpl

def draw_skeleton(img, kps, conf, edges, thr=0.3):
    img = img.copy()
    for i, (x, y) in enumerate(kps):
        if conf[i] < thr:
            continue
        cv2.circle(img, (int(x), int(y)), 4, (0, 0, 255), -1)
    for p, c in edges:
        if p >= len(kps) or c >= len(kps) or conf[p] < thr or conf[c] < thr:
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

def run_inference():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    checkpoint = torch.load(CKPT_PATH, map_location=device)
    model_state = checkpoint["model_state"]
    
    model = MLPLifterRotationHead(num_joints=24, dropout=0.1).to(device)
    model.load_state_dict(model_state)
    model.eval()
    
    pose_detector = MMPoseInferencer(pose2d="rtmpose-m_8xb64-210e_mpii-256x256", device=device.type)
    normalizer = NormalizerJoints2d(img_size=IMG_SIZE)
    
    img_bgr = preprocess_image(IMAGE_PATH, IMG_SIZE)
    
    det_result = next(pose_detector(img_bgr, show=False))
    kp16, cof16 = first_person_keypoints(det_result)
    
    if kp16 is None:
        print("No person detected")
        return
    
    kp24, cof24 = pad_mpii_to_smpl(kp16, cof16)
    kp_tensor = torch.from_numpy(kp24).unsqueeze(0)
    kp_norm = normalizer({"joints_2d": kp_tensor})["joints_2d"]
    
    with torch.no_grad():
        model_output = model(kp_norm.to(device))
        pos, rot = model_output
        pose3d = pos.cpu().numpy().reshape(24, 3)
        rotations = rot.cpu().numpy().reshape(24, 6)
    
    create_visualization(img_bgr, kp16, cof16, pose3d)
    save_results(pose3d, rotations)

def create_visualization(img_bgr, kp16, cof16, pose3d):
    fig = plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    overlay = draw_skeleton(img_bgr.copy(), kp16, cof16, mpii_edges)
    plt.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
    plt.title("2D Detection")
    plt.axis("off")
    
    ax = fig.add_subplot(1, 2, 2, projection='3d')
    X, Y, Z = pose3d[:, 0], pose3d[:, 1], pose3d[:, 2]
    ax.scatter(X, Y, Z, s=30, c='blue')
    for p, c in smpl_edges:
        ax.plot([X[p], X[c]], [Y[p], Y[c]], [Z[p], Z[c]], "b-", lw=2)
    ax.set_title("3D Pose")
    ax.view_init(elev=5, azim=-85)
    
    plt.tight_layout()
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    plt.savefig(os.path.join(OUTPUT_DIR, "pose_result.png"), dpi=150, bbox_inches='tight')
    plt.show()

def save_results(pose3d, rotations):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    results = {
        "pose_3d": pose3d.tolist(),
        "rotations": rotations.tolist()
    }
    
    output_file = os.path.join(OUTPUT_DIR, "pose_result.json")
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to {output_file}")

if __name__ == "__main__":
    run_inference()