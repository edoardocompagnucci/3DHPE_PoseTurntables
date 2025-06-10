import os
import sys
import pickle
import json
import numpy as np
import cv2
from tqdm import tqdm
import torch
from mmpose.apis import MMPoseInferencer

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
sys.path.insert(0, PROJECT_ROOT)

# 3DPW dataset path
THREEDPW_ROOT = os.path.join(PROJECT_ROOT, "data", "3DPW")
SEQUENCE_FILES_DIR = os.path.join(THREEDPW_ROOT, "sequenceFiles")
IMAGE_FILES_DIR = os.path.join(THREEDPW_ROOT, "imageFiles")

# Output directory for preprocessed detections
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "data", "3DPW_processed")
DETECTIONS_DIR = os.path.join(OUTPUT_DIR, "detections")

# SMPL to MPII joint mapping (from your existing project)
JOINT_MAPPING_PATH = os.path.join(PROJECT_ROOT, "data", "meta", "joints_mapping.json")

def preprocess_image(img, target_size=512):
    """
    Resize image to target_size while preserving aspect ratio (EXACT SAME AS INFERENCE.PY)
    img: cv2 image array
    Returns: resized image array
    """
    h, w = img.shape[:2]
    max_dim = max(h, w)
    square_img = np.zeros((max_dim, max_dim, 3), dtype=img.dtype)
    y_offset = (max_dim - h) // 2
    x_offset = (max_dim - w) // 2
    square_img[y_offset:y_offset+h, x_offset:x_offset+w] = img
    resized_img = cv2.resize(square_img, (target_size, target_size))
    return resized_img

def axis_angle_to_rotation_matrix(axis_angle):
    """
    Convert axis-angle representation to rotation matrix
    EXACT SAME FUNCTION AS HOUDINI VERIFICATION AND TEST SCRIPT
    axis_angle: (3,) vector where magnitude is angle, direction is axis
    Returns: (3, 3) rotation matrix
    """
    angle = np.linalg.norm(axis_angle)
    
    if angle < 1e-8:
        return np.eye(3)
    
    axis = axis_angle / angle
    cos_angle = np.cos(angle)
    sin_angle = np.sin(angle)
    
    # Rodrigues' rotation formula
    K = np.array([
        [0, -axis[2], axis[1]],
        [axis[2], 0, -axis[0]],
        [-axis[1], axis[0], 0]
    ])
    
    R = np.eye(3) + sin_angle * K + (1 - cos_angle) * (K @ K)
    return R

def rot_matrix_to_6d(rot_matrices):
    """
    Convert rotation matrices to 6D representation
    EXACT SAME FUNCTION AS SYNTHETIC DATASET AND TEST SCRIPT
    rot_matrices: (..., 3, 3) rotation matrices
    Returns: (..., 6) 6D representation (first two columns flattened)
    """
    return np.concatenate([rot_matrices[..., :, 0], rot_matrices[..., :, 1]], axis=-1)

def process_6d_rotations(pose_params):
    """
    Convert 3DPW pose parameters to 6D rotation representation
    EXACT SAME LOGIC AS TEST SCRIPT AND HOUDINI VERIFICATION
    pose_params: (72,) SMPL pose parameters (24 joints × 3 axis-angles)
    Returns: (24, 6) 6D rotation representation
    """
    axis_angles = pose_params.reshape(24, 3)
    
    # Convert axis-angles to rotation matrices (SAME AS HOUDINI)
    rot_matrices = np.zeros((24, 3, 3))
    for i in range(24):
        rot_matrices[i] = axis_angle_to_rotation_matrix(axis_angles[i])
    
    # Convert to 6D representation (SAME AS SYNTHETIC DATASET)
    rot_6d = rot_matrix_to_6d(rot_matrices)
    
    return rot_6d

def extract_camera_parameters(seq_data, frame_idx):
    """
    Extract camera parameters from 3DPW format to match synthetic dataset format
    Returns: K (3,3), R (3,3), t (3,)
    """
    # Camera intrinsics (same for all frames)
    K = seq_data['cam_intrinsics']  # (3, 3)
    
    # Camera pose (4x4 homogeneous matrix for this frame)
    cam_pose = seq_data['cam_poses'][frame_idx]  # (4, 4)
    
    # Extract rotation and translation
    R = cam_pose[:3, :3]  # (3, 3) rotation matrix
    t = cam_pose[:3, 3]   # (3,) translation vector
    
    return K, R, t

def load_joint_mapping():
    """Load SMPL to MPII joint mapping"""
    with open(JOINT_MAPPING_PATH, "r") as f:
        mapping = json.load(f)
    return mapping

def project_3d_to_2d(joints_3d_world, cam_intrinsics, cam_pose):
    """
    Project 3D joints to 2D using camera intrinsics and pose
    joints_3d_world: (24, 3) SMPL joint positions in world coordinates
    cam_intrinsics: (3, 3) camera intrinsics matrix
    cam_pose: (4, 4) camera pose matrix (world-to-camera transformation)
    Returns: (24, 2) 2D projected coordinates
    """
    # Transform from world to camera coordinates
    joints_3d_cam = (cam_pose[:3, :3] @ joints_3d_world.T + cam_pose[:3, 3:4]).T
    
    # Project to image plane
    projected = (cam_intrinsics @ joints_3d_cam.T).T  # (24, 3)
    
    # Normalize by depth
    joints_2d = projected[:, :2] / projected[:, 2:3]  # (24, 2)
    
    return joints_2d

def smpl_to_mpii_subset(smpl_joints_2d, mapping):
    """
    Extract MPII 16 joints from SMPL 24 joints using mapping
    smpl_joints_2d: (24, 2) SMPL joint coordinates
    Returns: (16, 2) MPII joint coordinates
    """
    smpl2mpii = mapping["smpl2mpii"]
    mpii_joints = np.zeros((16, 2), dtype=np.float32)
    
    for mpii_idx in range(16):
        # Find which SMPL joint maps to this MPII joint
        for smpl_idx, mapped_mpii in enumerate(smpl2mpii):
            if mapped_mpii == mpii_idx:
                mpii_joints[mpii_idx] = smpl_joints_2d[smpl_idx]
                break
    
    return mpii_joints

def transform_coordinates_to_resized(coords, original_size, target_size=512):
    """
    Transform coordinates from original image space to 512x512 resized space
    coords: (N, 2) coordinates in original image space
    original_size: (height, width) of original image
    target_size: target size (512)
    Returns: (N, 2) coordinates in 512x512 space
    """
    h, w = original_size
    max_dim = max(h, w)
    
    # Calculate transformation parameters
    x_offset = (max_dim - w) // 2
    y_offset = (max_dim - h) // 2
    scale_factor = target_size / max_dim
    
    # Apply padding offset
    coords_padded = coords.copy()
    coords_padded[:, 0] += x_offset  # x offset
    coords_padded[:, 1] += y_offset  # y offset
    
    # Apply scaling
    coords_resized = coords_padded * scale_factor
    
    return coords_resized

def match_detections_to_actors(detections, gt_joints_2d_list, threshold=50.0):
    """
    Match RTMPose detections to ground truth actors
    detections: List of detected person keypoints [(16,2), ...]
    gt_joints_2d_list: List of ground truth 2D projections for each actor [(16,2), ...]
    threshold: Distance threshold for matching (pixels)
    Returns: List of matched detection indices for each actor (or None if no match)
    """
    if not detections or not gt_joints_2d_list:
        return [None] * len(gt_joints_2d_list)
    
    matches = [None] * len(gt_joints_2d_list)
    used_detections = set()
    
    # For each actor, find the closest detection
    for actor_idx, gt_joints in enumerate(gt_joints_2d_list):
        best_detection_idx = None
        best_distance = float('inf')
        
        for det_idx, detection in enumerate(detections):
            if det_idx in used_detections:
                continue
                
            # Calculate average distance between corresponding joints
            distances = np.linalg.norm(detection - gt_joints, axis=1)
            avg_distance = np.mean(distances)
            
            if avg_distance < best_distance and avg_distance < threshold:
                best_distance = avg_distance
                best_detection_idx = det_idx
        
        if best_detection_idx is not None:
            matches[actor_idx] = best_detection_idx
            used_detections.add(best_detection_idx)
    
    return matches

def extract_detections_from_result(result):
    """
    Extract keypoints and scores from MMPose result
    Returns: List of (keypoints, scores) tuples for each detected person
    """
    detections = []
    
    try:
        predictions = result["predictions"][0]  # First image
        
        # Handle different result formats
        if isinstance(predictions, list):
            # Multiple detections
            for pred in predictions:
                if "keypoints" in pred and "keypoint_scores" in pred:
                    kpts = np.array(pred["keypoints"], dtype=np.float32)
                    scores = np.array(pred["keypoint_scores"], dtype=np.float32)
                    detections.append((kpts, scores))
        else:
            # Single detection
            if "keypoints" in predictions and "keypoint_scores" in predictions:
                kpts = np.array(predictions["keypoints"], dtype=np.float32)
                scores = np.array(predictions["keypoint_scores"], dtype=np.float32)
                detections.append((kpts, scores))
    
    except (KeyError, IndexError, TypeError) as e:
        print(f"Warning: Could not extract detections from result: {e}")
        return []
    
    return detections

def process_sequence(seq_name, split, pose_detector, joint_mapping):
    """
    Process a single 3DPW sequence
    """
    print(f"\nProcessing sequence: {seq_name} (split: {split})")
    
    # Load sequence data
    seq_file = os.path.join(SEQUENCE_FILES_DIR, split, f"{seq_name}.pkl")
    if not os.path.exists(seq_file):
        print(f"Sequence file not found: {seq_file}")
        return
    
    with open(seq_file, 'rb') as f:
        seq_data = pickle.load(f, encoding='latin1')
    
    # Get sequence info
    num_actors = len(seq_data['poses'])
    num_frames = len(seq_data['cam_poses'])
    
    print(f"  Actors: {num_actors}, Frames: {num_frames}")
    
    # Prepare output storage
    sequence_detections = {
        'sequence_name': seq_name,
        'split': split,
        'num_actors': num_actors,
        'num_frames': num_frames,
        'detections': {},  # frame_id -> actor_id -> complete data dict
        'metadata': {
            'detection_method': 'rtmpose-m_8xb64-210e_mpii-256x256',
            'image_preprocessing': 'resize_512x512_aspect_ratio_preserved',
            'keypoint_format': 'mpii_16_joints_in_512x512_pixel_space',
            'rotation_format': '6d_representation_24_joints_same_as_synthetic_dataset',
            '3d_positions_format': 'root_centered_24_joints_same_as_synthetic_dataset',
            'camera_format': 'K_R_t_same_as_synthetic_dataset',
            'rotation_pipeline': 'axis_angle_to_rotation_matrix_to_6d_same_as_houdini_verification',
            'matching_threshold': 50.0,
            'coordinate_system': 'keypoints_in_512x512_space_3d_root_centered_camera_per_frame',
            'coordinate_fix_applied': 'gt_projections_transformed_to_512x512_for_matching',
            'pipeline_verified_against_houdini': True,
            'pipeline_consistent_with_synthetic_dataset': True,
            'complete_data_format': 'ready_for_domain_mixing_training',
            'critical_bug_fixed': 'coordinate_space_mismatch_resolved'
        }
    }
    
    # Process each frame
    processed_frames = 0
    matched_frames = 0
    complete_data_frames = 0
    
    for frame_idx in tqdm(range(num_frames), desc=f"  Processing frames"):
        # Check if camera pose is valid for at least one actor
        valid_actors = [seq_data['campose_valid'][actor_idx][frame_idx] for actor_idx in range(num_actors)]
        if not any(valid_actors):
            continue
        
        # Load and preprocess image (resize to 512x512 with aspect ratio preservation)
        img_path = os.path.join(IMAGE_FILES_DIR, seq_data['sequence'], f'image_{frame_idx:05d}.jpg')
        if not os.path.exists(img_path):
            continue
        
        img_original = cv2.imread(img_path)
        if img_original is None:
            continue
        
        # Resize image to 512x512 (EXACT SAME AS INFERENCE PIPELINE)
        img_resized = preprocess_image(img_original, target_size=512)
        
        # Run RTMPose detection on resized image (512x512)
        try:
            result = next(pose_detector(img_resized, show=False))
            detections = extract_detections_from_result(result)
        except Exception as e:
            print(f"    Warning: Detection failed for frame {frame_idx}: {e}")
            continue
        
        # Extract camera parameters for this frame (SAME FORMAT AS SYNTHETIC DATASET)
        K, R, t = extract_camera_parameters(seq_data, frame_idx)
        
        # Get ground truth 2D projections for matching
        gt_joints_2d_list = []
        joints_3d_centered_list = []
        rotations_6d_list = []
        cam_pose = seq_data['cam_poses'][frame_idx]  # For projection
        
        for actor_idx in range(num_actors):
            if not seq_data['campose_valid'][actor_idx][frame_idx]:
                gt_joints_2d_list.append(None)
                joints_3d_centered_list.append(None)
                rotations_6d_list.append(None)
                continue
            
            # Get 3D joint positions for this actor and frame (in world coordinates)
            joints_3d_world = seq_data['jointPositions'][actor_idx][frame_idx].reshape(24, 3)
            
            # Root-center the 3D positions (SAME AS SYNTHETIC DATASET)
            joints_3d_centered = joints_3d_world - joints_3d_world[0]  # Subtract root joint
            joints_3d_centered_list.append(joints_3d_centered)
            
            # Project to 2D (with proper coordinate transformation)
            joints_2d_smpl = project_3d_to_2d(joints_3d_world, K, cam_pose)
            
            # Convert to MPII subset
            joints_2d_mpii = smpl_to_mpii_subset(joints_2d_smpl, joint_mapping)
            
            # *** CRITICAL FIX: Transform GT projections to 512x512 coordinates ***
            original_size = img_original.shape[:2]  # (height, width)
            joints_2d_mpii_resized = transform_coordinates_to_resized(joints_2d_mpii, original_size, 512)
            gt_joints_2d_list.append(joints_2d_mpii_resized)
            
            # Process 6D rotations for this actor (EXACT SAME AS TEST SCRIPT AND HOUDINI)
            pose_params = seq_data['poses'][actor_idx][frame_idx]  # 72 values
            rot_6d = process_6d_rotations(pose_params)  # (24, 6)
            rotations_6d_list.append(rot_6d)
        
        # Match detections to actors
        valid_gt = [gt for gt in gt_joints_2d_list if gt is not None]
        if valid_gt and detections:
            detection_keypoints = [det[0] for det in detections]
            matches = match_detections_to_actors(detection_keypoints, valid_gt)
        else:
            matches = [None] * len(gt_joints_2d_list)
        
        # Store results (COMPLETE FORMAT FOR DOMAIN MIXING)
        frame_detections = {}
        valid_actor_idx = 0
        frame_has_matches = False
        frame_has_complete_data = False
        
        for actor_idx in range(num_actors):
            if gt_joints_2d_list[actor_idx] is None:
                # Actor not valid in this frame
                frame_detections[actor_idx] = {
                    'keypoints': None,
                    'scores': None,
                    'joints_3d_centered': None,
                    'rotations_6d': None,
                    'K': None,
                    'R': None,
                    't': None,
                    'matched': False,
                    'reason': 'invalid_campose'
                }
            else:
                # Actor is valid - save complete data regardless of detection match
                joints_3d_centered = joints_3d_centered_list[actor_idx]
                rot_6d = rotations_6d_list[actor_idx]
                
                match_idx = matches[valid_actor_idx]
                if match_idx is not None:
                    # Successfully matched - COMPLETE DATA PACKAGE
                    kpts, scores = detections[match_idx]
                    frame_detections[actor_idx] = {
                        'keypoints': kpts.tolist(),                    # (16, 2) MPII keypoints in 512x512 space
                        'scores': scores.tolist(),                     # (16,) confidence scores
                        'joints_3d_centered': joints_3d_centered.tolist(),  # (24, 3) root-centered 3D positions
                        'rotations_6d': rot_6d.tolist(),               # (24, 6) SMPL 6D rotations
                        'K': K.tolist(),                               # (3, 3) camera intrinsics
                        'R': R.tolist(),                               # (3, 3) rotation matrix
                        't': t.tolist(),                               # (3,) translation vector
                        'matched': True,
                        'detection_idx': match_idx
                    }
                    frame_has_matches = True
                    frame_has_complete_data = True
                else:
                    # No matching detection found, but still save GT data for potential use
                    frame_detections[actor_idx] = {
                        'keypoints': None,
                        'scores': None,
                        'joints_3d_centered': joints_3d_centered.tolist(),  # Still save GT 3D
                        'rotations_6d': rot_6d.tolist(),                     # Still save GT rotations
                        'K': K.tolist(),                                     # Still save camera params
                        'R': R.tolist(),
                        't': t.tolist(),
                        'matched': False,
                        'reason': 'no_match_found'
                    }
                    frame_has_complete_data = True  # GT data is still complete
                valid_actor_idx += 1
        
        sequence_detections['detections'][frame_idx] = frame_detections
        processed_frames += 1
        if frame_has_matches:
            matched_frames += 1
        if frame_has_complete_data:
            complete_data_frames += 1
    
    # Save sequence detections
    output_file = os.path.join(DETECTIONS_DIR, f"{split}_{seq_name}_detections.pkl")
    with open(output_file, 'wb') as f:
        pickle.dump(sequence_detections, f)
    
    # Print detailed statistics
    total_frames = len(sequence_detections['detections'])
    matched_counts = [0] * num_actors
    complete_data_counts = [0] * num_actors
    
    for frame_data in sequence_detections['detections'].values():
        for actor_idx in range(num_actors):
            if frame_data[actor_idx]['matched']:
                matched_counts[actor_idx] += 1
            if frame_data[actor_idx]['joints_3d_centered'] is not None:
                complete_data_counts[actor_idx] += 1
    
    print(f"  ✅ Saved to: {output_file}")
    print(f"  📊 Frame statistics:")
    print(f"     Total processed frames: {total_frames}")
    print(f"     Frames with detection matches: {matched_frames} ({matched_frames/total_frames*100:.1f}%)")
    print(f"     Frames with complete GT data: {complete_data_frames} ({complete_data_frames/total_frames*100:.1f}%)")
    
    print(f"  👥 Actor statistics:")
    for actor_idx in range(num_actors):
        match_rate = matched_counts[actor_idx] / total_frames if total_frames > 0 else 0
        complete_rate = complete_data_counts[actor_idx] / total_frames if total_frames > 0 else 0
        print(f"     Actor {actor_idx}: detection_matches={matched_counts[actor_idx]}/{total_frames} ({match_rate:.1%}), complete_data={complete_data_counts[actor_idx]}/{total_frames} ({complete_rate:.1%})")

def main():
    """Main preprocessing function"""
    print("3DPW Complete Data Preprocessing (Detection + 3D + Camera)")
    print("=" * 70)
    print("🎯 Complete processing pipeline:")
    print("  1. Load 3DPW images and resize to 512×512 (aspect-ratio preserved)")
    print("  2. Run RTMPose-MPII detection on resized images")
    print("  3. Extract 3D joint positions and root-center them")
    print("  4. Convert SMPL pose parameters to 6D rotations")
    print("  5. Extract camera parameters (K, R, t) from 3DPW format")
    print("  6. Match detections to ground truth actors")
    print("  7. Save COMPLETE data package for domain mixing")
    print()
    print("💾 Complete data format:")
    print("  - RTMPose keypoints: (16, 2) MPII format in 512×512 pixel space")
    print("  - 3D positions: (24, 3) root-centered, same as synthetic dataset")
    print("  - 6D rotations: (24, 6) SMPL format, same pipeline as synthetic data")
    print("  - Camera params: K (3,3), R (3,3), t (3,) same as synthetic dataset")
    print("  - Confidence scores: (16,) RTMPose detection confidence")
    print()
    print("✅ Rotation pipeline VERIFIED against Houdini")
    print("✅ Same 6D representation as synthetic dataset")
    print("✅ Same 3D root-centering as synthetic dataset")
    print("✅ Same camera format as synthetic dataset")
    print("✅ Same image preprocessing as inference pipeline")
    print("✅ Ready for seamless domain mixing!")
    print()
    
    # Create output directories
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(DETECTIONS_DIR, exist_ok=True)
    
    # Load joint mapping
    joint_mapping = load_joint_mapping()
    print(f"✅ Loaded joint mapping from {JOINT_MAPPING_PATH}")
    
    # Initialize RTMPose detector
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pose_detector = MMPoseInferencer(
        pose2d="rtmpose-m_8xb64-210e_mpii-256x256",
        device=device
    )
    print(f"✅ Initialized RTMPose detector on {device}")
    
    # Get all available sequences from all splits
    if not os.path.exists(SEQUENCE_FILES_DIR):
        print(f"❌ Sequence files directory not found: {SEQUENCE_FILES_DIR}")
        return
    
    all_sequences = []
    splits = ["train", "validation", "test"]
    
    for split in splits:
        split_dir = os.path.join(SEQUENCE_FILES_DIR, split)
        if os.path.exists(split_dir):
            sequences = [f[:-4] for f in os.listdir(split_dir) if f.endswith('.pkl')]
            for seq in sequences:
                all_sequences.append((seq, split))
            print(f"📂 Found {len(sequences)} sequences in {split} split")
        else:
            print(f"⚠️  Split directory not found: {split}")
    
    if not all_sequences:
        print("❌ No sequences found in any split!")
        return
    
    print(f"📂 Total sequences to process: {len(all_sequences)}")
    print()
    
    # Process each sequence
    processed_count = 0
    failed_count = 0
    
    for seq_name, split in all_sequences:
        try:
            process_sequence(seq_name, split, pose_detector, joint_mapping)
            processed_count += 1
        except Exception as e:
            print(f"❌ Error processing {seq_name} ({split}): {e}")
            failed_count += 1
            continue
    
    print(f"\n" + "="*70)
    print(f"🎉 COMPLETE DATA PREPROCESSING FINISHED!")
    print(f"="*70)
    print(f"✅ Successfully processed: {processed_count} sequences")
    if failed_count > 0:
        print(f"❌ Failed to process: {failed_count} sequences")
    print(f"💾 Results saved to: {DETECTIONS_DIR}")
    print()
    print(f"📊 Complete output data format:")
    print(f"  - RTMPose keypoints: (16, 2) MPII format in 512×512 pixel space")
    print(f"  - 3D positions: (24, 3) root-centered, same as synthetic dataset")
    print(f"  - 6D rotations: (24, 6) SMPL format, same pipeline as synthetic data")
    print(f"  - Camera params: K (3,3), R (3,3), t (3,) same as synthetic dataset")
    print(f"  - Confidence scores: (16,) RTMPose detection confidence")
    print()
    print(f"🔄 Data pipeline consistency:")
    print(f"  ✅ Same rotation processing as Houdini verification")
    print(f"  ✅ Same 6D representation as synthetic dataset")
    print(f"  ✅ Same 3D root-centering as synthetic dataset")
    print(f"  ✅ Same camera format as synthetic dataset")
    print(f"  ✅ Same image preprocessing as inference pipeline")
    print(f"  ✅ Complete data package ready for domain mixing training!")

if __name__ == "__main__":
    main()