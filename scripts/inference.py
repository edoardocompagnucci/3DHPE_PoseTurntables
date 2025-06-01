import os, sys, json, cv2, torch, numpy as np, matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mmpose.apis import MMPoseInferencer

IMAGE_PATH = r"assets\demo_images\image_00059.jpg"
CKPT_PATH = r"checkpoints\mlp_lifter_rotation_20250529_232201\final_model.pth"
IMG_SIZE = 512
SAVE_RESULTS = False
OUTPUT_DIR = "outputs/week6_test"

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
sys.path.insert(0, PROJECT_ROOT)

from src.models.lifter import MLPLifter_v2
from src.models.rot_head import MLPLifterRotationHead
from src.utils.transforms import NormalizerJoints2d
from src.utils import rotation_utils

JOINT_MAPPING_PATH = os.path.join(PROJECT_ROOT, "data/meta/joints_mapping.json")
with open(JOINT_MAPPING_PATH, "r") as f:
    joint_map = json.load(f)

smpl2mpii = np.array([-1 if v is None else v for v in joint_map["smpl2mpii"]], dtype=np.int16)
smpl_edges = joint_map["edges"]
mpii_edges = joint_map["mpii_edges"]
joint_names = joint_map["smpl_names"]

# ============================================================================
# ADDED: Proper aspect ratio preservation
# ============================================================================
def preprocess_image_proper(image_path, target_size=512):
    """
    Properly preprocess image maintaining aspect ratio
    - Pad to square instead of stretching
    - Return scaling info for reference
    """
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Cannot load image: {image_path}")
    
    h, w = img.shape[:2]
    print(f"Original image size: {w}x{h}")
    
    # Calculate padding to make square
    max_dim = max(h, w)
    
    # Create square canvas with padding
    square_img = np.zeros((max_dim, max_dim, 3), dtype=img.dtype)
    
    # Calculate offsets to center the image
    y_offset = (max_dim - h) // 2
    x_offset = (max_dim - w) // 2
    
    # Place original image in center
    square_img[y_offset:y_offset+h, x_offset:x_offset+w] = img
    
    # Resize to target size
    resized_img = cv2.resize(square_img, (target_size, target_size))
    
    # Calculate scaling factors for reference
    scale_factor = target_size / max_dim
    
    preprocessing_info = {
        'original_size': (w, h),
        'square_size': max_dim,
        'target_size': target_size,
        'x_offset': x_offset,
        'y_offset': y_offset,
        'scale_factor': scale_factor
    }
    
    print(f"Preprocessed to: {target_size}x{target_size} (scale: {scale_factor:.3f}, padding: {x_offset}, {y_offset})")
    
    return resized_img, preprocessing_info

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

def analyze_rotations(rotations):
    """FIXED: Proper rotation analysis with correct validation"""
    issues = []
    if rotations is None:
        return issues, "No rotation data (position-only model)"
    
    # Convert 6D rotations to matrices
    if rotations.shape[-1] == 144:  # Flattened (1, 144)
        rot_6d = rotations.reshape(-1, 6)  # (24, 6)
    elif rotations.shape[-1] == 6:  # Already (24, 6)
        rot_6d = rotations
    else:
        return ["Unexpected rotation format"], f"Invalid shape: {rotations.shape}"
    
    try:
        rot_6d_tensor = torch.tensor(rot_6d, dtype=torch.float32)
        rot_matrices = rotation_utils.rot_6d_to_matrix(rot_6d_tensor).numpy()
    except Exception as e:
        return [f"6D to 3x3 conversion failed: {e}"], "Conversion error"
    
    valid_rotations = 0
    determinants = []
    orth_errors = []
    
    # FIXED: Proper validation thresholds
    det_threshold = 1e-3  # Very tight but reasonable
    orth_threshold = 1e-3  # Very tight but reasonable
    
    for i, rot_mat in enumerate(rot_matrices):
        det = np.linalg.det(rot_mat)
        orth_error = np.linalg.norm(rot_mat @ rot_mat.T - np.eye(3))
        
        determinants.append(det)
        orth_errors.append(orth_error)
        
        # Check if rotation matrix is valid
        if abs(det - 1.0) < det_threshold and orth_error < orth_threshold:
            valid_rotations += 1
        else:
            issues.append(f"Joint {i} ({joint_names[i]}): det={det:.6f}, orth_err={orth_error:.6f}")
    
    # Calculate quality metrics
    avg_det = np.mean(determinants)
    avg_orth = np.mean(orth_errors)
    
    status = f"Valid: {valid_rotations}/24 (det_avg={avg_det:.6f}, orth_avg={avg_orth:.6f})"
    return issues, status

def run_inference():
    print(f"Week 6 Real-World Testing (FIXED with Aspect Ratio)")
    print("="*50)
    
    # Setup device and models
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Load model - auto-detect type
    print(f"Loading model: {CKPT_PATH}")
    checkpoint = torch.load(CKPT_PATH, map_location=device)
    model_state = checkpoint["model_state"]
    
    # Check if rotation head model
    has_rotation_head = any("rotation_head" in key for key in model_state.keys())
    
    if has_rotation_head:
        print("Detected: Rotation Head Model")
        model = MLPLifterRotationHead(num_joints=24, dropout_rate=0.25).to(device)
    else:
        print("Detected: Position-Only Model")
        model = MLPLifter_v2(num_joints=24, dropout_rate=0.25).to(device)
    
    model.load_state_dict(model_state)
    model.eval()
    
    # Initialize components
    pose_detector = MMPoseInferencer(pose2d="rtmpose-m_8xb64-210e_mpii-256x256", device=device.type)
    normalizer = NormalizerJoints2d(img_size=IMG_SIZE)
    
    # FIXED: Load and process image with proper aspect ratio preservation
    print(f"Processing: {IMAGE_PATH}")
    img_bgr, preprocessing_info = preprocess_image_proper(IMAGE_PATH, IMG_SIZE)
    
    # 2D pose detection
    print("Detecting 2D pose...")
    det_result = next(pose_detector(img_bgr, show=False))
    kp16, cof16 = first_person_keypoints(det_result)
    
    if kp16 is None:
        print("No person detected")
        return
    
    print(f"Person detected (confidence: {cof16.mean():.3f})")
    
    # Convert to SMPL and normalize
    kp24, cof24 = pad_mpii_to_smpl(kp16, cof16)
    kp_tensor = torch.from_numpy(kp24).unsqueeze(0)
    kp_norm = normalizer({"joints_2d": kp_tensor})["joints_2d"]
    
    # 3D prediction
    print("Predicting 3D pose...")
    with torch.no_grad():
        model_output = model(kp_norm.to(device))
        
        # Handle different model outputs
        if isinstance(model_output, dict):
            # Rotation head model
            pose3d = model_output["positions"].cpu().numpy().reshape(24, 3)
            rotations = model_output["rotations"].cpu().numpy() if "rotations" in model_output else None
            model_type = "Rotation Head"
        else:
            # Legacy model
            pose3d = model_output.cpu().numpy().reshape(24, 3)
            rotations = None
            model_type = "Position Only"
    
    print(f"Model type: {model_type}")
    
    # Analyze results
    print("Analyzing quality...")
    rotation_issues, rotation_status = analyze_rotations(rotations)
    
    # Print results
    print(f"\nRESULTS:")
    print(f"  Image preprocessing: {preprocessing_info['original_size']} -> {IMG_SIZE}x{IMG_SIZE}")
    print(f"  Scale factor: {preprocessing_info['scale_factor']:.3f}")
    print(f"  2D Detection: {cof16.mean():.3f} confidence")
    print(f"  Visible joints: {(cof16 > 0.3).sum()}/16")
    print(f"  Rotation status: {rotation_status}")
    print(f"  Issues found: {len(rotation_issues)}")
    
    if rotation_issues:
        print(f"  First few issues:")
        for issue in rotation_issues[:3]:
            print(f"    {issue}")
    
    # Quality assessment
    detection_good = cof16.mean() > 0.5
    rotation_good = len(rotation_issues) < 3  # Very strict criteria
    
    if detection_good and rotation_good:
        assessment = "EXCELLENT"
        print(f"  WEEK 6 STATUS: {assessment} - Ready for benchmarking!")
    elif detection_good or rotation_good:
        assessment = "GOOD"
        print(f"  WEEK 6 STATUS: {assessment} - Minor improvements possible")
    else:
        assessment = "FAIR"
        print(f"  WEEK 6 STATUS: {assessment} - Some domain gap exists")
    
    # Create visualization (simplified, no emojis)
    create_clean_visualization(img_bgr, kp16, cof16, pose3d, rotations, rotation_issues, assessment, preprocessing_info)
    
    # Save results if requested
    if SAVE_RESULTS:
        save_results(pose3d, rotations, cof16, rotation_issues, rotation_status, assessment, preprocessing_info)

def create_clean_visualization(img_bgr, kp16, cof16, pose3d, rotations, issues, assessment, preprocessing_info):
    """Clean visualization without emoji characters"""
    fig = plt.figure(figsize=(15, 8))
    
    # 2D overlay
    plt.subplot(2, 3, 1)
    overlay = draw_skeleton(img_bgr.copy(), kp16, cof16, mpii_edges)
    plt.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
    title_text = f"2D Detection (Processed)\nConf: {cof16.mean():.3f}\nScale: {preprocessing_info['scale_factor']:.3f}"
    plt.title(title_text)
    plt.axis("off")
    
    # 3D pose - front view
    ax1 = fig.add_subplot(2, 3, 2, projection='3d')
    X, Y, Z = pose3d[:, 0], pose3d[:, 1], pose3d[:, 2]
    ax1.scatter(X, Y, Z, s=30, c='red')
    for p, c in smpl_edges:
        ax1.plot([X[p], X[c]], [Y[p], Y[c]], [Z[p], Z[c]], "b-", lw=2)
    ax1.set_title("3D Pose - Front")
    ax1.view_init(elev=5, azim=-85)
    
    # 3D pose - side view
    ax2 = fig.add_subplot(2, 3, 3, projection='3d')
    ax2.scatter(X, Y, Z, s=30, c='red')
    for p, c in smpl_edges:
        ax2.plot([X[p], X[c]], [Y[p], Y[c]], [Z[p], Z[c]], "b-", lw=2)
    ax2.set_title("3D Pose - Side")
    ax2.view_init(elev=5, azim=5)
    
    # Confidence histogram
    plt.subplot(2, 3, 4)
    plt.hist(cof16, bins=10, alpha=0.7)
    plt.axvline(cof16.mean(), color='red', linestyle='--', label=f'Mean: {cof16.mean():.3f}')
    plt.title('2D Confidence')
    plt.xlabel('Confidence')
    plt.ylabel('Count')
    plt.legend()
    
    # Rotation analysis
    plt.subplot(2, 3, 5)
    if rotations is not None:
        # Convert to matrices for analysis
        rot_6d = rotations.reshape(-1, 6)
        rot_6d_tensor = torch.tensor(rot_6d, dtype=torch.float32)
        rot_matrices = rotation_utils.rot_6d_to_matrix(rot_6d_tensor).numpy()
        
        # Show determinants
        determinants = [np.linalg.det(mat) for mat in rot_matrices]
        plt.bar(range(24), determinants, alpha=0.7)
        plt.axhline(1.0, color='red', linestyle='--', label='Perfect (1.0)')
        plt.title('Rotation Determinants')
        plt.xlabel('Joint Index')
        plt.ylabel('Determinant')
        plt.legend()
        plt.ylim(0.98, 1.02)  # Zoom in to see small variations
    else:
        plt.text(0.5, 0.5, 'No Rotation Data\n(Position-Only Model)', 
                ha='center', va='center', transform=plt.gca().transAxes)
        plt.title('Rotations: N/A')
    
    # Week 6 Summary
    plt.subplot(2, 3, 6)
    plt.axis('off')
    summary_text = f"WEEK 6 ASSESSMENT:\n\n"
    summary_text += f"Original size: {preprocessing_info['original_size']}\n"
    summary_text += f"Scale factor: {preprocessing_info['scale_factor']:.3f}\n"
    summary_text += f"2D Detection: {cof16.mean():.3f}\n"
    summary_text += f"Visible joints: {(cof16 > 0.3).sum()}/16\n"
    summary_text += f"Rotation issues: {len(issues)}\n\n"
    summary_text += f"STATUS: {assessment}\n\n"
    
    if assessment == "EXCELLENT":
        summary_text += "Ready for benchmarking!\nAspect ratio preserved!"
        color = "lightgreen"
    elif assessment == "GOOD":
        summary_text += "Very promising results!\nAspect ratio fixed."
        color = "lightblue"
    else:
        summary_text += "Some domain gap exists.\nBut aspect ratio OK."
        color = "lightyellow"
    
    plt.text(0.1, 0.9, summary_text, transform=plt.gca().transAxes, 
            verticalalignment='top', fontsize=10,
            bbox=dict(boxstyle="round,pad=0.5", facecolor=color))
    plt.title('Week 6 Summary')
    
    plt.tight_layout()
    
    if SAVE_RESULTS:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        plt.savefig(os.path.join(OUTPUT_DIR, "week6_analysis_aspect_fixed.png"), dpi=150, bbox_inches='tight')
        print(f"Visualization saved to {OUTPUT_DIR}/week6_analysis_aspect_fixed.png")
    
    plt.show()

def save_results(pose3d, rotations, cof16, issues, rotation_status, assessment, preprocessing_info):
    """Save numerical results"""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Convert rotations to both formats for completeness
    rotation_matrices = None
    if rotations is not None:
        rot_6d = rotations.reshape(-1, 6)
        rot_6d_tensor = torch.tensor(rot_6d, dtype=torch.float32)
        rotation_matrices = rotation_utils.rot_6d_to_matrix(rot_6d_tensor).numpy()
    
    results = {
        "image_path": IMAGE_PATH,
        "model_path": CKPT_PATH,
        "preprocessing_info": preprocessing_info,  # ADDED: Include preprocessing details
        "pose_3d": pose3d.tolist(),
        "rotations_6d": rotations.tolist() if rotations is not None else None,
        "rotations_3x3": rotation_matrices.tolist() if rotation_matrices is not None else None,
        "detection_confidence": {
            "mean": float(cof16.mean()),
            "std": float(cof16.std()),
            "visible_joints": int((cof16 > 0.3).sum())
        },
        "rotation_analysis": {
            "status": rotation_status,
            "issues": issues,
            "issue_count": len(issues)
        },
        "week6_assessment": {
            "overall_status": assessment,
            "detection_quality": "excellent" if cof16.mean() > 0.6 else "good" if cof16.mean() > 0.4 else "fair",
            "rotation_quality": "excellent" if len(issues) == 0 else "good" if len(issues) < 3 else "fair",
            "ready_for_benchmarking": assessment in ["EXCELLENT", "GOOD"],
            "synthetic_to_real_transfer": "working" if len(issues) < 5 else "needs_improvement",
            "aspect_ratio_preserved": True  # ADDED: Flag that aspect ratio was preserved
        }
    }
    
    output_file = os.path.join(OUTPUT_DIR, "week6_results_aspect_fixed.json")
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to {output_file}")

if __name__ == "__main__":
    run_inference()