# Save as: scripts/debug_predictions.py

import os
import sys
import torch
import numpy as np
import pickle
import json
from collections import defaultdict
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import time

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.rot_head import MLPLifterRotationHead
from src.utils.transforms import NormalizerJoints2d
from src.data.synthetic_pose_dataset import SyntheticPoseDataset
from src.data.threedpw_dataset import ThreeDPWDataset, MixedPoseDataset
from torch.utils.data import DataLoader


class PredictionDebugger:
    """Comprehensive debugging for 3D pose predictions"""
    
    def __init__(self, checkpoint_path, data_root, device=None):
        self.checkpoint_path = checkpoint_path
        self.data_root = data_root
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load model
        self.model = self._load_model()
        
        # Joint names for SMPL
        self.joint_names = [
            'pelvis', 'left_hip', 'right_hip', 'spine1', 'left_knee', 
            'right_knee', 'spine2', 'left_ankle', 'right_ankle', 'spine3', 
            'left_foot', 'right_foot', 'neck', 'left_collar', 'right_collar',
            'head', 'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
            'left_wrist', 'right_wrist', 'left_hand', 'right_hand'
        ]
        
        # Action categories for 3DPW
        self.action_categories = {
            'walking': ['walking', 'running', 'jogging'],
            'sitting': ['sitting', 'chair'],
            'stairs': ['stairs', 'upstairs', 'downstairs'],
            'standing': ['standing', 'waiting'],
            'dynamic': ['dancing', 'sports', 'exercise'],
            'other': []
        }
    
    def _load_model(self):
        """Load the trained model"""
        print(f"Loading model from {self.checkpoint_path}")
        
        model = MLPLifterRotationHead(num_joints=24, dropout=0.1).to(self.device)
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        model.load_state_dict(checkpoint['model_state'])
        model.eval()
        
        return model
    
    def _categorize_sequence(self, seq_name):
        """Categorize sequence by action type"""
        seq_lower = seq_name.lower()
        
        for category, keywords in self.action_categories.items():
            for keyword in keywords:
                if keyword in seq_lower:
                    return category
        return 'other'
    
    def analyze_mixed_validation(self, max_samples=2000):
        """Analyze performance on mixed validation set"""
        print("\n" + "="*80)
        print("ANALYZING MIXED VALIDATION SET")
        print("="*80)
        
        normalizer = NormalizerJoints2d(img_size=512)
        val_split_txt = os.path.join(self.data_root, "splits", "val.txt")
        
        # Create mixed validation dataset
        synthetic_val = SyntheticPoseDataset(
            data_root=self.data_root,
            split_txt=val_split_txt,
            transform=normalizer,
            augment_2d=False
        )
        
        threedpw_val = ThreeDPWDataset(
            data_root=self.data_root,
            split="validation",
            transform=normalizer,
            min_confidence=0.3
        )
        
        mixed_val = MixedPoseDataset(
            synthetic_dataset=synthetic_val,
            threedpw_dataset=threedpw_val,
            real_data_ratio=0.586,
            seed=42
        )
        
        # Limit samples for faster analysis
        if max_samples and len(mixed_val) > max_samples:
            indices = np.random.choice(len(mixed_val), max_samples, replace=False)
            mixed_val = torch.utils.data.Subset(mixed_val, indices)
            print(f"  Using {max_samples} random samples for faster analysis")
        
        val_loader = DataLoader(mixed_val, batch_size=64, shuffle=False, num_workers=4)
        
        # Analyze by data type
        results = self._analyze_dataset(val_loader, "Mixed Validation")
        
        # Separate synthetic vs real
        synthetic_errors = []
        real_errors = []
        
        print("\nAnalyzing by data type...")
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Processing batches"):
                inputs = batch["joints_2d"].to(self.device)
                target_pos = batch["joints_3d_centered"].to(self.device)
                dataset_types = batch["dataset_type"]
                
                pos3d, _ = self.model(inputs)
                pos3d = pos3d.reshape(inputs.shape[0], 24, 3)
                
                errors = torch.norm(pos3d - target_pos, dim=2).mean(dim=1) * 1000  # in mm
                
                for i, dt in enumerate(dataset_types):
                    if dt == 'synthetic':
                        synthetic_errors.append(errors[i].item())
                    else:
                        real_errors.append(errors[i].item())
        
        print(f"\nMixed Validation Breakdown:")
        print(f"  Synthetic samples: {len(synthetic_errors)}, MPJPE: {np.mean(synthetic_errors):.1f}mm")
        print(f"  Real samples: {len(real_errors)}, MPJPE: {np.mean(real_errors):.1f}mm")
        print(f"  Gap: {np.mean(real_errors) - np.mean(synthetic_errors):.1f}mm")
        
        return results
    
    def analyze_pure_3dpw(self, max_samples=2000):
        """Analyze performance on pure 3DPW validation"""
        print("\n" + "="*80)
        print("ANALYZING PURE 3DPW VALIDATION SET")
        print("="*80)
        
        normalizer = NormalizerJoints2d(img_size=512)
        
        threedpw_val = ThreeDPWDataset(
            data_root=self.data_root,
            split="validation",
            transform=normalizer,
            min_confidence=0.3
        )
        
        print(f"  Total 3DPW validation samples: {len(threedpw_val)}")
        
        # Limit samples for faster analysis
        if max_samples and len(threedpw_val) > max_samples:
            indices = np.random.choice(len(threedpw_val), max_samples, replace=False)
            threedpw_val = torch.utils.data.Subset(threedpw_val, indices)
            print(f"  Using {max_samples} random samples for faster analysis")
        
        val_loader = DataLoader(threedpw_val, batch_size=64, shuffle=False, num_workers=4)
        
        results = self._analyze_dataset(val_loader, "Pure 3DPW")
        
        # Analyze by sequence
        sequence_errors = defaultdict(list)
        action_errors = defaultdict(list)
        
        print("\nAnalyzing by sequence and action...")
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Processing batches"):
                inputs = batch["joints_2d"].to(self.device)
                target_pos = batch["joints_3d_centered"].to(self.device)
                seq_names = batch["sequence_name"]
                
                pos3d, _ = self.model(inputs)
                pos3d = pos3d.reshape(inputs.shape[0], 24, 3)
                
                errors = torch.norm(pos3d - target_pos, dim=2).mean(dim=1) * 1000
                
                for i, seq in enumerate(seq_names):
                    sequence_errors[seq].append(errors[i].item())
                    action = self._categorize_sequence(seq)
                    action_errors[action].append(errors[i].item())
        
        print("\nError by Action Type:")
        for action in sorted(action_errors.keys()):
            errors = action_errors[action]
            if errors:  # Check if not empty
                print(f"  {action:12s}: {np.mean(errors):6.1f}mm (n={len(errors)})")
        
        print("\nWorst 5 Sequences:")
        seq_mean_errors = [(seq, np.mean(errors)) for seq, errors in sequence_errors.items() if errors]
        seq_mean_errors.sort(key=lambda x: x[1], reverse=True)
        for seq, error in seq_mean_errors[:5]:
            print(f"  {seq}: {error:.1f}mm")
        
        print("\nBest 5 Sequences:")
        for seq, error in seq_mean_errors[-5:]:
            print(f"  {seq}: {error:.1f}mm")
        
        return results, action_errors, sequence_errors
    
    def _analyze_dataset(self, dataloader, dataset_name):
        """Common analysis for any dataset"""
        
        # Initialize tracking
        joint_errors = np.zeros(24)
        joint_counts = np.zeros(24)
        
        total_errors = []
        
        # For tracking extremes
        worst_samples = []
        best_samples = []
        
        print(f"\nAnalyzing {dataset_name}...")
        start_time = time.time()
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(dataloader, desc="Computing errors")):
                inputs = batch["joints_2d"].to(self.device)
                target_pos = batch["joints_3d_centered"].to(self.device)
                sample_ids = batch.get("sample_id", [f"batch{batch_idx}_sample{i}" for i in range(len(inputs))])
                
                # Get predictions
                pos3d, rot6d = self.model(inputs)
                
                # Reshape
                batch_size = inputs.shape[0]
                pos3d = pos3d.reshape(batch_size, 24, 3)
                
                # Compute errors
                joint_diffs = pos3d - target_pos  # (B, 24, 3)
                joint_distances = torch.norm(joint_diffs, dim=2)  # (B, 24)
                
                # Update joint statistics
                joint_errors += joint_distances.sum(dim=0).cpu().numpy()
                joint_counts += batch_size
                
                # Sample-wise errors
                sample_errors = joint_distances.mean(dim=1) * 1000  # Convert to mm
                
                for i, error in enumerate(sample_errors):
                    error_item = error.item()
                    total_errors.append(error_item)
                    
                    # Track extremes
                    sample_info = {
                        'error': error_item,
                        'sample_id': sample_ids[i] if i < len(sample_ids) else f"sample_{i}",
                        'joint_errors': joint_distances[i].cpu().numpy() * 1000
                    }
                    
                    if len(worst_samples) < 10:
                        worst_samples.append(sample_info)
                        worst_samples.sort(key=lambda x: x['error'], reverse=True)
                    elif error_item > worst_samples[-1]['error']:
                        worst_samples[-1] = sample_info
                        worst_samples.sort(key=lambda x: x['error'], reverse=True)
                    
                    if len(best_samples) < 10:
                        best_samples.append(sample_info)
                        best_samples.sort(key=lambda x: x['error'])
                    elif error_item < best_samples[-1]['error']:
                        best_samples[-1] = sample_info
                        best_samples.sort(key=lambda x: x['error'])
        
        elapsed = time.time() - start_time
        print(f"  Analysis took {elapsed:.1f} seconds")
        
        # Compute final statistics
        mean_joint_errors = (joint_errors / joint_counts) * 1000  # in mm
        overall_mpjpe = np.mean(total_errors)
        
        # Print results
        print(f"\n{dataset_name} Results:")
        print(f"  Overall MPJPE: {overall_mpjpe:.1f}mm")
        print(f"  Std deviation: {np.std(total_errors):.1f}mm")
        print(f"  Min error: {np.min(total_errors):.1f}mm")
        print(f"  Max error: {np.max(total_errors):.1f}mm")
        
        print("\nPer-Joint MPJPE (sorted by error):")
        joint_ranking = sorted(enumerate(mean_joint_errors), key=lambda x: x[1], reverse=True)
        
        for idx, error in joint_ranking[:10]:  # Show top 10 worst joints
            print(f"  {self.joint_names[idx]:15s}: {error:6.1f}mm")
        print("  ...")
        for idx, error in joint_ranking[-3:]:  # Show best 3 joints
            print(f"  {self.joint_names[idx]:15s}: {error:6.1f}mm")
        
        print("\nWorst 5 predictions:")
        for sample in worst_samples[:5]:
            print(f"  {sample['sample_id']}: {sample['error']:.1f}mm")
        
        return {
            'overall_mpjpe': overall_mpjpe,
            'joint_errors': mean_joint_errors,
            'joint_ranking': joint_ranking,
            'worst_samples': worst_samples,
            'best_samples': best_samples,
            'all_errors': total_errors
        }
    
    def analyze_prediction_consistency(self):
        """Test if model behaves consistently on same inputs"""
        print("\n" + "="*80)
        print("ANALYZING PREDICTION CONSISTENCY")
        print("="*80)
        
        normalizer = NormalizerJoints2d(img_size=512)
        
        # Load a few 3DPW samples
        threedpw_val = ThreeDPWDataset(
            data_root=self.data_root,
            split="validation",
            transform=normalizer,
            min_confidence=0.3
        )
        
        test_loader = DataLoader(threedpw_val, batch_size=8, shuffle=False)
        test_batch = next(iter(test_loader))
        
        inputs = test_batch["joints_2d"].to(self.device)
        
        # Run multiple forward passes
        print("Running 10 forward passes on same batch...")
        predictions = []
        
        with torch.no_grad():
            for i in range(10):
                pos3d, _ = self.model(inputs)
                predictions.append(pos3d)
        
        predictions = torch.stack(predictions)  # (10, B, 72)
        
        # Compute variance
        pred_std = predictions.std(dim=0).mean()
        pred_mean = predictions.mean(dim=0)
        
        print(f"Prediction std across runs: {pred_std:.6f}")
        print(f"Prediction consistency: {'GOOD' if pred_std < 0.001 else 'BAD - Model is non-deterministic!'}")
        
        # Test with different batch sizes
        print("\nTesting with different batch configurations...")
        single_input = inputs[0:1]
        
        single_preds = []
        with torch.no_grad():
            for i in range(5):
                pos3d, _ = self.model(single_input)
                single_preds.append(pos3d)
        
        single_preds = torch.stack(single_preds)
        single_std = single_preds.std(dim=0).mean()
        
        print(f"Single sample std: {single_std:.6f}")
        
        return pred_std.item()
    
    def analyze_2d_quality(self, max_samples=1000):
        """Analyze 2D keypoint quality"""
        print("\n" + "="*80)
        print("ANALYZING 2D KEYPOINT QUALITY")
        print("="*80)
        
        normalizer = NormalizerJoints2d(img_size=512)
        
        # Check both synthetic and real
        datasets = {
            'synthetic': SyntheticPoseDataset(
                data_root=self.data_root,
                split_txt=os.path.join(self.data_root, "splits", "val.txt"),
                transform=normalizer,
                augment_2d=False
            ),
            '3dpw': ThreeDPWDataset(
                data_root=self.data_root,
                split="validation",
                transform=normalizer,
                min_confidence=0.3
            )
        }
        
        for name, dataset in datasets.items():
            print(f"\n{name.upper()} 2D Statistics:")
            print(f"  Dataset size: {len(dataset)}")
            
            # Limit samples for speed
            if max_samples and len(dataset) > max_samples:
                indices = np.random.choice(len(dataset), max_samples, replace=False)
                dataset = torch.utils.data.Subset(dataset, indices)
                print(f"  Using {max_samples} random samples for analysis")
            
            loader = DataLoader(dataset, batch_size=256, shuffle=False, num_workers=4)
            
            all_joints_2d = []
            all_confidences = []
            
            print(f"  Loading 2D keypoints...")
            for batch in tqdm(loader, desc=f"Processing {name}"):
                joints_2d = batch["joints_2d"]  # (B, 24, 2)
                
                # Handle both tensor and list confidences
                if "avg_confidence" in batch:
                    confidences = batch["avg_confidence"]
                    if isinstance(confidences, torch.Tensor):
                        all_confidences.extend(confidences.numpy())
                    else:
                        all_confidences.extend(confidences)
                
                all_joints_2d.append(joints_2d)
            
            all_joints_2d = torch.cat(all_joints_2d, dim=0).numpy()
            
            # Compute statistics
            print(f"  Shape: {all_joints_2d.shape}")
            print(f"  Mean position: ({all_joints_2d.mean(axis=(0,1))[0]:.3f}, {all_joints_2d.mean(axis=(0,1))[1]:.3f})")
            print(f"  Std position: ({all_joints_2d.std(axis=(0,1))[0]:.3f}, {all_joints_2d.std(axis=(0,1))[1]:.3f})")
            print(f"  Min: ({all_joints_2d.min():.3f}), Max: ({all_joints_2d.max():.3f})")
            
            # Check for outliers
            outliers = np.sum((all_joints_2d < -0.95) | (all_joints_2d > 0.95))
            print(f"  Outliers (near boundaries): {outliers} ({outliers/all_joints_2d.size*100:.2f}%)")
            
            # Joint-wise statistics
            joint_means = all_joints_2d.mean(axis=0)  # (24, 2)
            joint_stds = all_joints_2d.std(axis=0)    # (24, 2)
            
            print(f"\n  Most variable joints (by std):")
            joint_variability = joint_stds.mean(axis=1)
            top_variable = np.argsort(joint_variability)[-5:][::-1]
            for idx in top_variable:
                print(f"    {self.joint_names[idx]:15s}: std={joint_variability[idx]:.3f}")
            
            if name == '3dpw' and all_confidences:
                print(f"\n  Detection confidence statistics:")
                print(f"    Average confidence: {np.mean(all_confidences):.3f}")
                print(f"    Confidence range: [{np.min(all_confidences):.3f}, {np.max(all_confidences):.3f}]")
                print(f"    Low confidence (<0.5): {np.sum(np.array(all_confidences) < 0.5)} samples")
        
        return all_joints_2d
    
    def quick_analysis(self):
        """Quick analysis with limited samples for fast results"""
        print("\n" + "="*80)
        print("QUICK ANALYSIS MODE (Limited samples)")
        print("="*80)
        
        # Just analyze a subset
        self.analyze_2d_quality(max_samples=500)
        self.analyze_prediction_consistency()
        results_mixed = self.analyze_mixed_validation(max_samples=1000)
        results_3dpw, _, _ = self.analyze_pure_3dpw(max_samples=1000)
        
        # Quick summary
        print("\n" + "="*80)
        print("QUICK SUMMARY")
        print("="*80)
        print(f"Mixed validation MPJPE: {results_mixed['overall_mpjpe']:.1f}mm")
        print(f"Pure 3DPW MPJPE: {results_3dpw['overall_mpjpe']:.1f}mm")
        print(f"Domain gap: {results_3dpw['overall_mpjpe'] - results_mixed['overall_mpjpe']:.1f}mm")
        
        # Extremity vs core
        extremity_joints = [7, 8, 10, 11, 20, 21, 22, 23]
        core_joints = [0, 3, 6, 9, 12, 15]
        
        extremity_gap = np.mean([results_3dpw['joint_errors'][j] - results_mixed['joint_errors'][j] 
                                for j in extremity_joints])
        core_gap = np.mean([results_3dpw['joint_errors'][j] - results_mixed['joint_errors'][j] 
                           for j in core_joints])
        
        print(f"\nDomain gap by joint type:")
        print(f"  Core joints gap: {core_gap:.1f}mm")
        print(f"  Extremity joints gap: {extremity_gap:.1f}mm")
        
        return results_mixed, results_3dpw


def main():
    """Run complete debugging analysis"""
    
    # Configuration - MODIFY THESE PATHS
    CHECKPOINT_PATH = r"checkpoints\transformer_v2_fixed_20250613_222808\checkpoint_epoch_40.pth"  # Update this!
    DATA_ROOT = "data/"
    
    # Choose analysis mode
    print("Debug Analysis Options:")
    print("1. Quick analysis (fast, ~2-3 minutes)")
    print("2. Full analysis (complete, ~10-15 minutes)")
    
    mode = input("\nSelect mode (1 or 2): ").strip()
    
    # Create debugger
    debugger = PredictionDebugger(CHECKPOINT_PATH, DATA_ROOT)
    
    if mode == "1":
        # Quick analysis
        debugger.quick_analysis()
    else:
        # Full analysis
        print("Starting comprehensive debug analysis...")
        
        # 1. Analyze 2D quality
        joints_2d_stats = debugger.analyze_2d_quality(max_samples=2000)
        
        # 2. Check prediction consistency
        consistency = debugger.analyze_prediction_consistency()
        
        # 3. Analyze mixed validation
        results_mixed = debugger.analyze_mixed_validation(max_samples=3000)
        
        # 4. Analyze pure 3DPW
        results_3dpw, action_errors, sequence_errors = debugger.analyze_pure_3dpw(max_samples=3000)
        
        # 5. Create visualizations
        debugger.visualize_errors(results_mixed, results_3dpw)
        
        # 6. Save detailed report
        all_results = {
            'consistency': consistency,
            'results_mixed': results_mixed,
            'results_3dpw': results_3dpw,
            'action_errors': action_errors,
        }
        debugger.save_detailed_report(all_results)
    
    print("\n" + "="*80)
    print("DEBUGGING COMPLETE!")
    print("="*80)


if __name__ == "__main__":
    main()