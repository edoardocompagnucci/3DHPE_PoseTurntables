#!/usr/bin/env python3
"""
Test script for the mixed pose dataset
"""

import os
import sys
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
sys.path.insert(0, PROJECT_ROOT)

from src.data.mixed_pose_dataset import create_mixed_dataset
from src.utils.transforms import NormalizerJoints2d


def test_dataset_creation():
    """Test creating different types of datasets"""
    print("=" * 60)
    print("TESTING DATASET CREATION")
    print("=" * 60)
    
    data_root = os.path.join(PROJECT_ROOT, "data")
    
    # Test synthetic only
    print("\n1. Testing synthetic-only dataset...")
    try:
        synthetic_dataset = create_mixed_dataset(
            data_root=data_root,
            mode='synthetic',
            split='train',
            transform=NormalizerJoints2d(img_size=512),
            augment_2d=False
        )
        print(f"✅ Synthetic dataset created: {len(synthetic_dataset)} samples")
        print(f"   Data type: {synthetic_dataset.get_data_type()}")
    except Exception as e:
        print(f"❌ Synthetic dataset failed: {e}")
    
    # Test real only
    print("\n2. Testing real-only dataset...")
    try:
        real_dataset = create_mixed_dataset(
            data_root=data_root,
            mode='real',
            split='train',
            transform=NormalizerJoints2d(img_size=512),
            augment_2d=False
        )
        print(f"✅ Real dataset created: {len(real_dataset)} samples")
        print(f"   Data type: {real_dataset.get_data_type()}")
    except Exception as e:
        print(f"❌ Real dataset failed: {e}")
    
    # Test mixed
    print("\n3. Testing mixed dataset...")
    try:
        mixed_dataset = create_mixed_dataset(
            data_root=data_root,
            mode='mixed',
            split='train',
            synthetic_ratio=0.7,
            transform=NormalizerJoints2d(img_size=512),
            augment_2d=False
        )
        print(f"✅ Mixed dataset created: {len(mixed_dataset)} samples")
        distribution = mixed_dataset.get_dataset_distribution()
        print(f"   Distribution: {distribution}")
        
        total = sum(distribution.values())
        for dtype, count in distribution.items():
            print(f"   {dtype}: {count}/{total} ({count/total*100:.1f}%)")
            
    except Exception as e:
        print(f"❌ Mixed dataset failed: {e}")


def test_sample_loading():
    """Test loading individual samples"""
    print("\n" + "=" * 60)
    print("TESTING SAMPLE LOADING")
    print("=" * 60)
    
    data_root = os.path.join(PROJECT_ROOT, "data")
    
    try:
        # Create mixed dataset
        mixed_dataset = create_mixed_dataset(
            data_root=data_root,
            mode='mixed',
            split='train',
            synthetic_ratio=0.5,
            transform=NormalizerJoints2d(img_size=512),
            augment_2d=False
        )
        
        print(f"Testing samples from mixed dataset ({len(mixed_dataset)} total samples)")
        
        # Test first few samples
        for i in range(min(5, len(mixed_dataset))):
            print(f"\nSample {i}:")
            try:
                sample = mixed_dataset[i]
                
                print(f"  Data type: {sample.get('data_type', 'unknown')}")
                print(f"  Source dataset: {sample.get('source_dataset', 'unknown')}")
                print(f"  Source index: {sample.get('source_idx', 'unknown')}")
                
                # Check required fields
                required_fields = ['joints_2d', 'joints_3d_centered', 'rot_6d', 'K', 'R', 't']
                for field in required_fields:
                    if field in sample:
                        shape = sample[field].shape if hasattr(sample[field], 'shape') else 'scalar'
                        print(f"  {field}: {shape}")
                    else:
                        print(f"  {field}: MISSING ❌")
                
                # Verify data ranges
                joints_2d = sample['joints_2d']
                if hasattr(joints_2d, 'min') and hasattr(joints_2d, 'max'):
                    print(f"  joints_2d range: [{joints_2d.min():.3f}, {joints_2d.max():.3f}]")
                
            except Exception as e:
                print(f"  ❌ Error loading sample {i}: {e}")
                
    except Exception as e:
        print(f"❌ Failed to create dataset for sample testing: {e}")


def test_dataloader():
    """Test DataLoader with mixed dataset"""
    print("\n" + "=" * 60)
    print("TESTING DATALOADER")
    print("=" * 60)
    
    data_root = os.path.join(PROJECT_ROOT, "data")
    
    try:
        mixed_dataset = create_mixed_dataset(
            data_root=data_root,
            mode='mixed',
            split='train',
            synthetic_ratio=0.6,
            transform=NormalizerJoints2d(img_size=512),
            augment_2d=True  # Test with augmentation
        )
        
        dataloader = DataLoader(
            mixed_dataset,
            batch_size=8,
            shuffle=True,
            num_workers=2,
            drop_last=True
        )
        
        print(f"DataLoader created with batch_size=8")
        print(f"Total batches: {len(dataloader)}")
        
        # Test first batch
        print("\nTesting first batch...")
        batch = next(iter(dataloader))
        
        print(f"Batch keys: {list(batch.keys())}")
        
        # Check batch shapes
        for key, value in batch.items():
            if hasattr(value, 'shape'):
                print(f"  {key}: {value.shape}")
            elif isinstance(value, (list, tuple)):
                print(f"  {key}: list/tuple of length {len(value)}")
            else:
                print(f"  {key}: {type(value)}")
        
        # Check data type distribution in batch
        if 'data_type' in batch:
            data_types = batch['data_type']
            unique, counts = np.unique(data_types, return_counts=True)
            print(f"\nBatch data type distribution:")
            for dtype, count in zip(unique, counts):
                print(f"  {dtype}: {count}/8 ({count/8*100:.1f}%)")
        
        print("✅ DataLoader test completed successfully")
        
    except Exception as e:
        print(f"❌ DataLoader test failed: {e}")


def test_augmentation():
    """Test data augmentation"""
    print("\n" + "=" * 60)
    print("TESTING AUGMENTATION")
    print("=" * 60)
    
    data_root = os.path.join(PROJECT_ROOT, "data")
    
    try:
        # Create datasets with and without augmentation
        dataset_no_aug = create_mixed_dataset(
            data_root=data_root,
            mode='mixed',
            split='train',
            synthetic_ratio=0.5,
            transform=None,  # No normalization to see raw coords
            augment_2d=False
        )
        
        dataset_with_aug = create_mixed_dataset(
            data_root=data_root,
            mode='mixed',
            split='train',
            synthetic_ratio=0.5,
            transform=None,  # No normalization to see raw coords
            augment_2d=True,
            noise_std=0.02,
            confidence_noise=0.005,
            max_shift=0.005
        )
        
        print("Comparing augmented vs non-augmented data...")
        
        # Compare first few samples
        for i in range(min(3, len(dataset_no_aug))):
            sample_orig = dataset_no_aug[i]
            sample_aug = dataset_with_aug[i]
            
            joints_2d_orig = sample_orig['joints_2d']
            joints_2d_aug = sample_aug['joints_2d']
            
            if hasattr(joints_2d_orig, 'numpy'):
                joints_2d_orig = joints_2d_orig.numpy()
                joints_2d_aug = joints_2d_aug.numpy()
            
            diff = np.mean(np.abs(joints_2d_orig - joints_2d_aug))
            print(f"  Sample {i} ({sample_orig.get('data_type', 'unknown')}): "
                  f"Mean abs difference = {diff:.4f} pixels")
        
        print("✅ Augmentation test completed")
        
    except Exception as e:
        print(f"❌ Augmentation test failed: {e}")


def visualize_mixed_sample():
    """Create a visualization of mixed dataset samples"""
    print("\n" + "=" * 60)
    print("CREATING VISUALIZATION")
    print("=" * 60)
    
    data_root = os.path.join(PROJECT_ROOT, "data")
    
    try:
        mixed_dataset = create_mixed_dataset(
            data_root=data_root,
            mode='mixed',
            split='train',
            synthetic_ratio=0.5,
            transform=None,  # No normalization for visualization
            augment_2d=False
        )
        
        # Find one synthetic and one real sample
        synthetic_sample = None
        real_sample = None
        
        for i in range(min(20, len(mixed_dataset))):
            sample = mixed_dataset[i]
            if sample.get('data_type') == 'synthetic' and synthetic_sample is None:
                synthetic_sample = sample
            elif sample.get('data_type') == 'real' and real_sample is None:
                real_sample = sample
            
            if synthetic_sample is not None and real_sample is not None:
                break
        
        if synthetic_sample is None or real_sample is None:
            print("❌ Could not find both synthetic and real samples")
            return
        
        # Create visualization
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Plot 2D keypoints
        def plot_2d_keypoints(ax, joints_2d, title):
            if hasattr(joints_2d, 'numpy'):
                joints_2d = joints_2d.numpy()
            ax.scatter(joints_2d[:, 0], joints_2d[:, 1], s=30, alpha=0.7)
            ax.set_xlim(0, 512)
            ax.set_ylim(512, 0)  # Flip Y axis for image coordinates
            ax.set_title(title)
            ax.set_xlabel('X (pixels)')
            ax.set_ylabel('Y (pixels)')
            ax.grid(True, alpha=0.3)
        
        # Plot 3D pose
        def plot_3d_pose(ax, joints_3d, title):
            if hasattr(joints_3d, 'numpy'):
                joints_3d = joints_3d.numpy()
            ax.scatter(joints_3d[:, 0], joints_3d[:, 1], joints_3d[:, 2], s=30, alpha=0.7)
            ax.set_title(title)
            ax.set_xlabel('X (m)')
            ax.set_ylabel('Y (m)')
            ax.set_zlabel('Z (m)')
        
        # Synthetic sample
        plot_2d_keypoints(axes[0, 0], synthetic_sample['joints_2d'], 'Synthetic - 2D Keypoints')
        axes[0, 1] = fig.add_subplot(2, 2, 2, projection='3d')
        plot_3d_pose(axes[0, 1], synthetic_sample['joints_3d_centered'], 'Synthetic - 3D Pose')
        
        # Real sample
        plot_2d_keypoints(axes[1, 0], real_sample['joints_2d'], 'Real - 2D Keypoints')
        axes[1, 1] = fig.add_subplot(2, 2, 4, projection='3d')
        plot_3d_pose(axes[1, 1], real_sample['joints_3d_centered'], 'Real - 3D Pose')
        
        plt.tight_layout()
        
        # Save visualization
        output_dir = os.path.join(PROJECT_ROOT, "outputs")
        os.makedirs(output_dir, exist_ok=True)
        viz_path = os.path.join(output_dir, "mixed_dataset_samples.png")
        plt.savefig(viz_path, dpi=150, bbox_inches='tight')
        plt.show()
        
        print(f"✅ Visualization saved to: {viz_path}")
        
    except Exception as e:
        print(f"❌ Visualization failed: {e}")


def main():
    """Run all tests"""
    print("🧪 MIXED POSE DATASET TEST SUITE")
    print("=" * 60)
    
    # Run tests
    test_dataset_creation()
    test_sample_loading()
    test_dataloader()
    test_augmentation()
    visualize_mixed_sample()
    
    print("\n" + "=" * 60)
    print("🎉 TEST SUITE COMPLETED")
    print("=" * 60)


if __name__ == "__main__":
    main()