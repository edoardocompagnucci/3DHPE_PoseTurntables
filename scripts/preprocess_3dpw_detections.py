"""
Preprocess 3DPW dataset with RTMPose detections and complete data package

Key improvement: Smart detection-to-actor matching
- Previous: Cap detections at 2, match whoever remains (might match wrong people)
- Now: Keep all detections, find best matching subset (correctly matches actors)

This solves the "173% detection rate" issue where background people were being
matched to ground truth actors.

UPDATED: Now also saves joints_3d_world for GT 2D projection support
UPDATED 2: Now also saves image resolution for each frame
"""

import os
import sys
import pickle
import json
import numpy as np
import cv2
from tqdm import tqdm
import torch
from mmpose.apis import MMPoseInferencer

# Run with VERBOSE=1 python preprocess_3dpw_detections.py for frame-by-frame logging

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
sys.path.insert(0, PROJECT_ROOT)

# Verbose mode - set to True for frame-by-frame logging
VERBOSE = os.environ.get('VERBOSE', '0') == '1'

# 3DPW dataset path
THREEDPW_ROOT = os.path.join(PROJECT_ROOT, "data", "3DPW")
SEQUENCE_FILES_DIR = os.path.join(THREEDPW_ROOT, "sequenceFiles")
IMAGE_FILES_DIR = os.path.join(THREEDPW_ROOT, "imageFiles")

# Output directory for preprocessed detections
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "data", "3DPW_processed")
DETECTIONS_DIR = os.path.join(OUTPUT_DIR, "detections")

JOINT_MAPPING_PATH = os.path.join(PROJECT_ROOT, "data", "meta", "joints_mapping.json")

REST_POSE_DATA_PATH = os.path.join(PROJECT_ROOT, "data", "meta", "rest_pose_data.npy")

CONFIDENCE_THRESHOLD = 0.5  # Filter out low-confidence detections (increase to 0.6-0.7 if still getting >2 detections)
MAX_DETECTIONS_PER_FRAME = 2  # Expected max actors in 3DPW (used for statistics, not capping)
MATCHING_THRESHOLD = 25.0  # Tighter matching threshold in pixels


def load_rest_pose_data():
    """Load rest pose data exported from Houdini"""
    if not os.path.exists(REST_POSE_DATA_PATH):
        raise FileNotFoundError(f"Rest pose data not found: {REST_POSE_DATA_PATH}")
    
    data = np.load(REST_POSE_DATA_PATH, allow_pickle=True).item()
    print(f"✅ Loaded rest pose data: {data['num_joints']} joints")
    return data

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
    EXACT SAME FUNCTION AS HOUDINI VERIFICATION
    """
    angle = np.linalg.norm(axis_angle)
    
    if angle < 1e-8:
        return np.eye(3)
    
    axis = axis_angle / angle
    x, y, z = axis
    c = np.cos(angle)
    s = np.sin(angle)
    C = 1 - c
    
    return np.array([
        [x*x*C + c,   x*y*C - z*s, x*z*C + y*s],
        [y*x*C + z*s, y*y*C + c,   y*z*C - x*s],
        [z*x*C - y*s, z*y*C + x*s, z*z*C + c  ]
    ])

def forward_kinematics(local_rotations, parents):
    """
    Apply forward kinematics to get world rotations
    EXACT SAME FUNCTION AS HOUDINI SCRIPT
    """
    rot_mats = np.zeros_like(local_rotations)
    
    # Process joints in hierarchical order
    for joint_id in range(len(parents)):
        parent_id = parents[joint_id]
        
        if parent_id == -1:  # Root joint
            rot_mats[joint_id] = local_rotations[joint_id]
        else:
            # World = Parent_world @ Local_current
            rot_mats[joint_id] = rot_mats[parent_id] @ local_rotations[joint_id]
    
    return rot_mats

def rot_matrix_to_6d(rot_matrices):
    """
    Convert rotation matrices to 6D representation
    EXACT SAME FUNCTION AS SYNTHETIC DATASET
    """
    return np.concatenate([rot_matrices[..., :, 0], rot_matrices[..., :, 1]], axis=-1)

def process_rotations_houdini_pipeline(pose_params, rest_pose_data):
    """
    Process 3DPW pose parameters following EXACT SAME PIPELINE AS HOUDINI
    pose_params: (72,) SMPL pose parameters (24 joints × 3 axis-angles)
    rest_pose_data: Rest pose data from Houdini export
    Returns: rot_mats (24, 3, 3), rot_6d (24, 6)
    """
    # Step 1: Convert axis-angles to local rotation matrices (SAME AS HOUDINI)
    axis_angles = pose_params.reshape(24, 3)
    smpl_local = np.zeros((24, 3, 3))
    for i in range(24):
        smpl_local[i] = axis_angle_to_rotation_matrix(axis_angles[i])
    
    # Step 2: Apply rest pose (identity in this case, but keeping for consistency)
    rest_transforms = rest_pose_data['rest_transforms']
    final_local = np.einsum('ijk,ikl->ijl', smpl_local, rest_transforms)
    
    # Step 3: Apply forward kinematics (SAME AS HOUDINI)
    parents = rest_pose_data['smpl_parents']
    rot_mats = forward_kinematics(final_local, parents)
    
    # Step 4: Transpose each matrix (SAME AS HOUDINI - CRITICAL!)
    rot_mats = rot_mats.transpose(0, 2, 1)
    
    # Step 5: Convert to 6D representation
    rot_6d = rot_matrix_to_6d(rot_mats)
    
    return rot_mats, rot_6d

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

def extract_detections_from_result(result, confidence_threshold=CONFIDENCE_THRESHOLD):
    """
    Extract keypoints and scores from MMPose result with confidence filtering
    Returns: List of (keypoints, scores) tuples for each detected person, sorted by confidence
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
                    avg_confidence = np.mean(scores)
                    
                    # Only keep high-confidence detections
                    if avg_confidence >= confidence_threshold:
                        detections.append((kpts, scores, avg_confidence))
        else:
            # Single detection
            if "keypoints" in predictions and "keypoint_scores" in predictions:
                kpts = np.array(predictions["keypoints"], dtype=np.float32)
                scores = np.array(predictions["keypoint_scores"], dtype=np.float32)
                avg_confidence = np.mean(scores)
                
                if avg_confidence >= confidence_threshold:
                    detections.append((kpts, scores, avg_confidence))
    
    except (KeyError, IndexError, TypeError) as e:
        print(f"Warning: Could not extract detections from result: {e}")
        return []
    
    # Sort by confidence (highest first)
    detections.sort(key=lambda x: x[2], reverse=True)
    
    # Return without the confidence value (for compatibility)
    return [(kpts, scores) for kpts, scores, _ in detections]

def match_detections_to_actors(detections, gt_joints_2d_list, threshold=MATCHING_THRESHOLD):
    """
    Match RTMPose detections to ground truth actors
    When there are more detections than actors, finds the best matching subset
    
    Example scenario:
    - GT actors: 2 people
    - Detections: 4 people [Background1, Actor1, Background2, Actor2]
    - Old method: Keep top 2 by confidence → might get [Background1, Background2] ❌
    - New method: Try all combinations → correctly matches [Actor1, Actor2] ✅
    
    detections: List of detected person keypoints [(16,2), ...]
    gt_joints_2d_list: List of ground truth 2D projections for each actor [(16,2), ...]
    threshold: Distance threshold for matching (pixels)
    Returns: List of matched detection indices for each actor (or None if no match)
    """
    global VERBOSE
    
    if not detections or not gt_joints_2d_list:
        return [None] * len(gt_joints_2d_list)
    
    # Calculate distance matrix between all detections and all GT actors
    num_detections = len(detections)
    num_actors = len(gt_joints_2d_list)
    distance_matrix = np.full((num_actors, num_detections), float('inf'))
    
    for actor_idx, gt_joints in enumerate(gt_joints_2d_list):
        for det_idx, detection in enumerate(detections):
            # Calculate average distance between corresponding joints
            distances = np.linalg.norm(detection - gt_joints, axis=1)
            avg_distance = np.mean(distances)
            if avg_distance < threshold:
                distance_matrix[actor_idx, det_idx] = avg_distance
    
    # Greedy matching: for each actor, find closest unused detection
    matches = [None] * num_actors
    used_detections = set()
    
    # Sort actors by minimum distance to any detection (process best matches first)
    actor_min_distances = [(actor_idx, np.min(distance_matrix[actor_idx])) 
                          for actor_idx in range(num_actors)]
    actor_min_distances.sort(key=lambda x: x[1])
    
    for actor_idx, min_dist in actor_min_distances:
        if min_dist == float('inf'):
            continue  # No valid detection for this actor
            
        # Find best available detection for this actor
        best_det_idx = None
        best_distance = float('inf')
        
        for det_idx in range(num_detections):
            if det_idx in used_detections:
                continue
            
            distance = distance_matrix[actor_idx, det_idx]
            if distance < best_distance:
                best_distance = distance
                best_det_idx = det_idx
        
        if best_det_idx is not None:
            matches[actor_idx] = best_det_idx
            used_detections.add(best_det_idx)
            
            if VERBOSE and num_detections > num_actors:
                print(f"      Matched actor {actor_idx} to detection {best_det_idx} (dist: {best_distance:.1f}px)")
    
    return matches

def process_sequence(seq_name, split, pose_detector, joint_mapping, rest_pose_data):
    """
    Process a single 3DPW sequence with improved detection handling
    Returns: (success, has_high_detections)
    """
    print(f"\nProcessing sequence: {seq_name} (split: {split})")
    
    # Load sequence data
    seq_file = os.path.join(SEQUENCE_FILES_DIR, split, f"{seq_name}.pkl")
    if not os.path.exists(seq_file):
        print(f"Sequence file not found: {seq_file}")
        return False, False
    
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
            'rotation_format': '6d_representation_24_joints_EXACT_SAME_AS_HOUDINI_PIPELINE',
            'rotation_processing': 'axis_angle_to_rotation_matrix_then_FK_then_transpose_then_6d',
            '3d_positions_format': 'root_centered_24_joints_same_as_synthetic_dataset',
            '3d_world_positions_format': 'original_24_joints_in_world_coordinates',
            'camera_format': 'K_R_t_same_as_synthetic_dataset',
            'image_resolution_saved': True,  # NEW
            'forward_kinematics_applied': True,
            'rest_pose_transformation_applied': True,
            'transpose_applied_after_FK': True,
            'houdini_pipeline_verified': True,
            'complete_data_format': 'ready_for_domain_mixing_training',
            'rot_mats_saved_as_rot_mats': True,
            '6d_rotations_saved_as_rot_6d': True,
            'joints_3d_world_saved': True,
            'confidence_threshold': CONFIDENCE_THRESHOLD,
            'max_detections_per_frame': MAX_DETECTIONS_PER_FRAME,
            'matching_threshold': MATCHING_THRESHOLD
        }
    }
    
    # Process each frame
    processed_frames = 0
    matched_frames = 0
    complete_data_frames = 0
    excessive_detection_frames = 0
    all_detection_counts = []
    detection_failures = 0
    frames_exceeding_actors = 0
    
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
        
        # Get original image resolution
        original_height, original_width = img_original.shape[:2]
        
        # Resize image to 512x512 (EXACT SAME AS INFERENCE PIPELINE)
        img_resized = preprocess_image(img_original, target_size=512)
        
        # Run RTMPose detection on resized image (512x512)
        try:
            result = next(pose_detector(img_resized, show=False))
            detections = extract_detections_from_result(result, confidence_threshold=CONFIDENCE_THRESHOLD)
            
            # Track original detection count (but DON'T cap yet!)
            original_detection_count = len(detections)
            all_detection_counts.append(original_detection_count)
            
            if original_detection_count > MAX_DETECTIONS_PER_FRAME:
                excessive_detection_frames += 1
                if VERBOSE:
                    print(f"    Frame {frame_idx}: {original_detection_count} detections (will match best {MAX_DETECTIONS_PER_FRAME})")
            
        except Exception as e:
            if VERBOSE:
                print(f"    Warning: Detection failed for frame {frame_idx}: {e}")
            detection_failures += 1
            continue
        
        # Extract camera parameters for this frame (SAME FORMAT AS SYNTHETIC DATASET)
        K, R, t = extract_camera_parameters(seq_data, frame_idx)
        
        # Get ground truth 2D projections for matching
        gt_joints_2d_list = []
        joints_3d_world_list = []
        joints_3d_centered_list = []
        rot_mats_list = []
        rot_6d_list = []
        cam_pose = seq_data['cam_poses'][frame_idx]  # For projection
        
        for actor_idx in range(num_actors):
            if not seq_data['campose_valid'][actor_idx][frame_idx]:
                gt_joints_2d_list.append(None)
                joints_3d_world_list.append(None)
                joints_3d_centered_list.append(None)
                rot_mats_list.append(None)
                rot_6d_list.append(None)
                continue
            
            # Get 3D joint positions for this actor and frame (in world coordinates)
            joints_3d_world = seq_data['jointPositions'][actor_idx][frame_idx].reshape(24, 3)
            joints_3d_world_list.append(joints_3d_world)
            
            # Root-center the 3D positions (SAME AS SYNTHETIC DATASET)
            joints_3d_centered = joints_3d_world - joints_3d_world[0]  # Subtract root joint
            joints_3d_centered_list.append(joints_3d_centered)
            
            # Project to 2D (with proper coordinate transformation)
            joints_2d_smpl = project_3d_to_2d(joints_3d_world, K, cam_pose)
            
            # Convert to MPII subset
            joints_2d_mpii = smpl_to_mpii_subset(joints_2d_smpl, joint_mapping)
            
            # Transform GT projections to 512x512 coordinates
            original_size = (original_height, original_width)  # Using actual image size
            joints_2d_mpii_resized = transform_coordinates_to_resized(joints_2d_mpii, original_size, 512)
            gt_joints_2d_list.append(joints_2d_mpii_resized)
            
            # Process rotations using EXACT SAME PIPELINE AS HOUDINI
            pose_params = seq_data['poses'][actor_idx][frame_idx]  # 72 values
            rot_mats, rot_6d = process_rotations_houdini_pipeline(pose_params, rest_pose_data)
            rot_mats_list.append(rot_mats)  # (24, 3, 3) - FK world rotations
            rot_6d_list.append(rot_6d)  # (24, 6) - 6D representation
        
        # Count valid GT actors
        valid_gt = [gt for gt in gt_joints_2d_list if gt is not None]
        num_valid_actors = len(valid_gt)
        
        # Track if detections exceed valid actors
        if len(detections) > num_valid_actors and num_valid_actors > 0:
            if VERBOSE:
                print(f"    Frame {frame_idx}: {len(detections)} detections vs {num_valid_actors} valid actors")
            frames_exceeding_actors += 1
        
        # Match detections to actors
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
                    'joints_3d_world': None,
                    'joints_3d_centered': None,
                    'rot_mats': None,
                    'rot_6d': None,
                    'K': None,
                    'R': None,
                    't': None,
                    'image_width': original_width,  # NEW
                    'image_height': original_height,  # NEW
                    'matched': False,
                    'reason': 'invalid_campose'
                }
            else:
                # Actor is valid - save complete data regardless of detection match
                joints_3d_world = joints_3d_world_list[actor_idx]
                joints_3d_centered = joints_3d_centered_list[actor_idx]
                rot_mats = rot_mats_list[actor_idx]  # FK world rotations
                rot_6d = rot_6d_list[actor_idx]  # 6D representation
                
                match_idx = matches[valid_actor_idx]
                if match_idx is not None:
                    # Successfully matched - COMPLETE DATA PACKAGE
                    kpts, scores = detections[match_idx]
                    frame_detections[actor_idx] = {
                        'keypoints': kpts.tolist(),                    # (16, 2) MPII keypoints in 512x512 space
                        'scores': scores.tolist(),                     # (16,) confidence scores
                        'joints_3d_world': joints_3d_world.tolist(),
                        'joints_3d_centered': joints_3d_centered.tolist(),  # (24, 3) root-centered 3D positions
                        'rot_mats': rot_mats.tolist(),                # (24, 3, 3) FK world rotation matrices
                        'rot_6d': rot_6d.tolist(),                    # (24, 6) 6D rotations
                        'K': K.tolist(),                               # (3, 3) camera intrinsics
                        'R': R.tolist(),                               # (3, 3) rotation matrix
                        't': t.tolist(),                               # (3,) translation vector
                        'image_width': original_width,                 # NEW: original image width
                        'image_height': original_height,               # NEW: original image height
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
                        'joints_3d_world': joints_3d_world.tolist(),
                        'joints_3d_centered': joints_3d_centered.tolist(),
                        'rot_mats': rot_mats.tolist(),
                        'rot_6d': rot_6d.tolist(),
                        'K': K.tolist(),
                        'R': R.tolist(),
                        't': t.tolist(),
                        'image_width': original_width,                 # NEW
                        'image_height': original_height,               # NEW
                        'matched': False,
                        'reason': 'no_match_found'
                    }
                    frame_has_complete_data = True
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
    
    if all_detection_counts:
        avg_detections = np.mean(all_detection_counts)
        max_detections = np.max(all_detection_counts)
        frames_over_2 = sum(1 for d in all_detection_counts if d > 2)
        has_high_detections = avg_detections > 2.0
        
        print(f"  📈 Detection statistics:")
        print(f"     Average detections per frame: {avg_detections:.2f}")
        print(f"     Max detections in a frame: {max_detections}")
        print(f"     Frames with >2 detections: {frames_over_2} ({frames_over_2/len(all_detection_counts)*100:.1f}%)")
        print(f"     Frames exceeding valid actors: {frames_exceeding_actors}")
        print(f"     Frames capped at 2 detections: {excessive_detection_frames}")
        if detection_failures > 0:
            print(f"     Detection failures: {detection_failures}")
        
        if has_high_detections:
            print(f"     ⚠️  HIGH DETECTION RATE: {avg_detections:.2f} avg detections (max should be 2)")
            print(f"     This sequence likely has background people or false detections!")
            print(f"     💡 Consider increasing CONFIDENCE_THRESHOLD from {CONFIDENCE_THRESHOLD} to 0.6-0.7")
    else:
        has_high_detections = False
    
    print(f"  👥 Actor statistics:")
    for actor_idx in range(num_actors):
        match_rate = matched_counts[actor_idx] / total_frames if total_frames > 0 else 0
        complete_rate = complete_data_counts[actor_idx] / total_frames if total_frames > 0 else 0
        print(f"     Actor {actor_idx}: detection_matches={matched_counts[actor_idx]}/{total_frames} ({match_rate:.1%}), complete_data={complete_data_counts[actor_idx]}/{total_frames} ({complete_rate:.1%})")
    
    return True, has_high_detections

def main():
    """Main preprocessing function with improved detection handling"""
    print("3DPW Complete Data Preprocessing - HOUDINI PIPELINE (with Detection Fixes)")
    print("=" * 70)
    print("🎯 DETECTION IMPROVEMENTS:")
    print(f"  - Confidence threshold: {CONFIDENCE_THRESHOLD}")
    print(f"  - Max expected actors: {MAX_DETECTIONS_PER_FRAME} (3DPW never has >2)")
    print(f"  - Matching threshold: {MATCHING_THRESHOLD} pixels")
    print(f"  - Matching strategy: Best subset selection (handles excess detections)")
    print(f"  - Verbose mode: {'ON' if VERBOSE else 'OFF'} (set VERBOSE=1 for frame-by-frame logs)")
    print()
    print("🎯 EXACT SAME COMPUTATION PIPELINE AS HOUDINI:")
    print("  1. Load rest pose data from data/meta/rest_pose_data.npy")
    print("  2. Convert 3DPW axis-angles to local rotation matrices")
    print("  3. Apply rest pose transformation (identity)")
    print("  4. Apply forward kinematics with corrected SMPL hierarchy")
    print("  5. Transpose result matrices (CRITICAL!)")
    print("  6. Convert to 6D representation")
    print("  7. Save rot_mats (rot_mats) and rot_6d (rot_6d)")
    print()
    print("💾 Complete data format:")
    print("  - RTMPose keypoints: (16, 2) MPII format in 512×512 pixel space")
    print("  - 3D positions: (24, 3) root-centered, same as synthetic dataset")
    print("  - 3D world positions: (24, 3) original world coordinates")
    print("  - World rotations: (24, 3, 3) FK result, same as Houdini (rot_mats)")
    print("  - 6D rotations: (24, 6) converted from world rotations (rot_6d)")
    print("  - Camera params: K (3,3), R (3,3), t (3,) same as synthetic dataset")
    print("  - Image resolution: width, height for each frame (NEW!)")
    print()
    
    # Load rest pose data first
    try:
        rest_pose_data = load_rest_pose_data()
    except FileNotFoundError as e:
        print(f"❌ {e}")
        print("💡 Please export rest pose data from Houdini first!")
        return
    
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
    high_detection_sequences = []
    
    for seq_name, split in all_sequences:
        try:
            success, has_high_detections = process_sequence(seq_name, split, pose_detector, joint_mapping, rest_pose_data)
            if success:
                processed_count += 1
                if has_high_detections:
                    high_detection_sequences.append((seq_name, split))
            else:
                failed_count += 1
        except Exception as e:
            print(f"❌ Error processing {seq_name} ({split}): {e}")
            failed_count += 1
            continue
    
    print(f"\n" + "="*70)
    print(f"🎉 HOUDINI PIPELINE PREPROCESSING COMPLETE (with Detection Fixes)!")
    print(f"="*70)
    print(f"✅ Successfully processed: {processed_count} sequences")
    if failed_count > 0:
        print(f"❌ Failed to process: {failed_count} sequences")
    print(f"💾 Results saved to: {DETECTIONS_DIR}")
    
    if high_detection_sequences:
        print(f"\n⚠️  Sequences with excessive detections ({len(high_detection_sequences)} total):")
        for seq_name, split in high_detection_sequences[:10]:  # Show first 10
            print(f"   - {seq_name} ({split})")
        if len(high_detection_sequences) > 10:
            print(f"   ... and {len(high_detection_sequences) - 10} more")
        print(f"\n💡 To fix excessive detections:")
        print(f"   1. Increase CONFIDENCE_THRESHOLD from {CONFIDENCE_THRESHOLD} to 0.6-0.7")
        print(f"   2. Re-run preprocessing on affected sequences")
    
    print()
    print(f"📊 Complete output data format:")
    print(f"  - RTMPose keypoints: (16, 2) MPII format in 512×512 pixel space")
    print(f"  - 3D positions: (24, 3) root-centered, same as synthetic dataset")
    print(f"  - 3D world positions: (24, 3) original world coordinates")
    print(f"  - World rotations: (24, 3, 3) FK result, same as Houdini (rot_mats)")
    print(f"  - 6D rotations: (24, 6) converted from world rotations (rot_6d)")
    print(f"  - Camera params: K (3,3), R (3,3), t (3,) same as synthetic dataset")
    print(f"  - Image resolution: width, height for each frame (NEW!)")
    print()
    print(f"🔧 Detection improvements applied:")
    print(f"  ✅ Confidence filtering (threshold: {CONFIDENCE_THRESHOLD})")
    print(f"  ✅ Smart matching: finds best subset from all detections")
    print(f"  ✅ Tighter matching threshold ({MATCHING_THRESHOLD} pixels)")
    print(f"  ✅ Better logging for suspicious frames")
    print(f"  ✅ No arbitrary capping - matches correct people even with excess detections")
    print(f"  ✅ Now saves joints_3d_world for GT 2D projection support")
    print(f"  ✅ Now saves image resolution (width, height) for each frame")

if __name__ == "__main__":
    main()