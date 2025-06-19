import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from tqdm import tqdm

# Add project root to path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
sys.path.insert(0, PROJECT_ROOT)

from data.mixed_pose_dataset import create_mixed_dataset
from utils.transforms import NormalizerJoints2d


def collect_keypoint_statistics(dataset, num_samples=1000):
    """Collect keypoint statistics per joint"""
    joints_2d_list = []
    joints_3d_list = []
    
    indices = np.random.choice(len(dataset), min(num_samples, len(dataset)), replace=False)
    
    for idx in tqdm(indices, desc="Processing samples"):
        sample = dataset[idx]
        joints_2d_list.append(sample['joints_2d'].numpy())
        joints_3d_list.append(sample['joints_3d_centered'].numpy())
    
    joints_2d = np.stack(joints_2d_list)  # (N, 24, 2)
    joints_3d = np.stack(joints_3d_list)  # (N, 24, 3)
    
    # Calculate per-joint statistics
    stats = {
        '2d_mean_x': joints_2d.mean(axis=0)[:, 0],  # (24,)
        '2d_mean_y': joints_2d.mean(axis=0)[:, 1],
        '2d_std_x': joints_2d.std(axis=0)[:, 0],
        '2d_std_y': joints_2d.std(axis=0)[:, 1],
        '3d_mean_x': joints_3d.mean(axis=0)[:, 0],
        '3d_mean_y': joints_3d.mean(axis=0)[:, 1],
        '3d_mean_z': joints_3d.mean(axis=0)[:, 2],
        '3d_std_x': joints_3d.std(axis=0)[:, 0],
        '3d_std_y': joints_3d.std(axis=0)[:, 1],
        '3d_std_z': joints_3d.std(axis=0)[:, 2],
        '2d_all': joints_2d,
        '3d_all': joints_3d
    }
    
    return stats


def plot_advanced_distributions():
    # Configuration
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "data"))
    NUM_SAMPLES = 2000
    
    normalizer = NormalizerJoints2d(img_size=512)
    
    print("Loading datasets...")
    
    # Create datasets
    datasets = {
        'Synthetic': create_mixed_dataset(
            data_root=DATA_ROOT, split='train',
            synthetic_ratio=1.0, real_ratio=0.0,
            transform=normalizer
        ),
        'Real Train': create_mixed_dataset(
            data_root=DATA_ROOT, split='train',
            synthetic_ratio=0.0, real_ratio=1.0,
            transform=normalizer
        ),
        'Real Val': create_mixed_dataset(
            data_root=DATA_ROOT, split='val',
            synthetic_ratio=0.0, real_ratio=1.0,
            transform=normalizer
        )
    }
    
    # Collect statistics
    all_stats = {}
    for name, dataset in datasets.items():
        print(f"\nCollecting {name} statistics...")
        all_stats[name] = collect_keypoint_statistics(dataset, NUM_SAMPLES)
    
    # Create figure with subplots
    fig = plt.figure(figsize=(20, 12))
    
    # Define joint groups for better visualization
    joint_groups = {
        'Torso': [0, 1, 2, 3, 6, 9, 12, 15],  # Root, hips, spine, neck, head
        'Arms': [13, 14, 16, 17, 18, 19, 20, 21, 22, 23],  # Shoulders to fingers
        'Legs': [4, 5, 7, 8, 10, 11]  # Knees to feet
    }
    
    # 1. Box plots for 2D distribution ranges
    ax1 = plt.subplot(3, 2, 1)
    data_2d_range = []
    for name, stats in all_stats.items():
        # Calculate range for each joint (max - min across samples)
        ranges = []
        for joint_idx in range(24):
            joint_data = stats['2d_all'][:, joint_idx, :]  # (N, 2)
            range_x = joint_data[:, 0].max() - joint_data[:, 0].min()
            range_y = joint_data[:, 1].max() - joint_data[:, 1].min()
            ranges.extend([range_x, range_y])
        data_2d_range.extend([(name, r) for r in ranges])
    
    df_2d_range = pd.DataFrame(data_2d_range, columns=['Dataset', 'Range'])
    sns.boxplot(data=df_2d_range, x='Dataset', y='Range', ax=ax1)
    ax1.set_title('2D Keypoint Range Distribution')
    ax1.set_ylabel('Range (normalized units)')
    ax1.grid(True, alpha=0.3)
    
    # 2. Box plots for 3D distribution ranges
    ax2 = plt.subplot(3, 2, 2)
    data_3d_range = []
    for name, stats in all_stats.items():
        ranges = []
        for joint_idx in range(24):
            joint_data = stats['3d_all'][:, joint_idx, :]  # (N, 3)
            range_x = joint_data[:, 0].max() - joint_data[:, 0].min()
            range_y = joint_data[:, 1].max() - joint_data[:, 1].min()
            range_z = joint_data[:, 2].max() - joint_data[:, 2].min()
            ranges.extend([range_x, range_y, range_z])
        data_3d_range.extend([(name, r) for r in ranges])
    
    df_3d_range = pd.DataFrame(data_3d_range, columns=['Dataset', 'Range'])
    sns.boxplot(data=df_3d_range, x='Dataset', y='Range', ax=ax2)
    ax2.set_title('3D Keypoint Range Distribution')
    ax2.set_ylabel('Range (meters)')
    ax2.grid(True, alpha=0.3)
    
    # 3. 2D centroid comparison
    ax3 = plt.subplot(3, 2, 3)
    colors = {'Synthetic': 'blue', 'Real Train': 'green', 'Real Val': 'orange'}
    markers = {'Torso': 'o', 'Arms': '^', 'Legs': 's'}
    
    for name, stats in all_stats.items():
        for group_name, joint_indices in joint_groups.items():
            x_means = stats['2d_mean_x'][joint_indices]
            y_means = stats['2d_mean_y'][joint_indices]
            ax3.scatter(x_means, y_means, color=colors[name], marker=markers[group_name],
                       s=100, alpha=0.7, label=f'{name} - {group_name}')
    
    ax3.set_xlabel('2D X (normalized)')
    ax3.set_ylabel('2D Y (normalized)')
    ax3.set_title('Mean 2D Joint Positions by Body Part')
    ax3.grid(True, alpha=0.3)
    ax3.set_aspect('equal')
    # Create custom legend
    handles, labels = ax3.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax3.legend(by_label.values(), by_label.keys(), loc='best', fontsize=8)
    
    # 4. 3D spread comparison (Y-Z plane)
    ax4 = plt.subplot(3, 2, 4)
    for name, stats in all_stats.items():
        # Plot mean positions with error bars showing std
        ax4.errorbar(stats['3d_mean_y'], stats['3d_mean_z'], 
                    xerr=stats['3d_std_y'], yerr=stats['3d_std_z'],
                    fmt='o', color=colors[name], alpha=0.5, label=name)
    
    ax4.set_xlabel('3D Y (meters)')
    ax4.set_ylabel('3D Z (meters)')
    ax4.set_title('3D Joint Spread (Y-Z plane with std)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # 5. Per-axis variance comparison
    ax5 = plt.subplot(3, 2, 5)
    x_pos = np.arange(3)
    width = 0.25
    
    for i, (name, stats) in enumerate(all_stats.items()):
        variances = [
            stats['3d_all'][:, :, 0].var(),  # X variance
            stats['3d_all'][:, :, 1].var(),  # Y variance
            stats['3d_all'][:, :, 2].var()   # Z variance
        ]
        ax5.bar(x_pos + i*width, variances, width, label=name, color=colors[name], alpha=0.7)
    
    ax5.set_xlabel('Axis')
    ax5.set_ylabel('Variance')
    ax5.set_title('3D Coordinate Variance by Axis')
    ax5.set_xticks(x_pos + width)
    ax5.set_xticklabels(['X', 'Y', 'Z'])
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # 6. Distribution overlap heatmap
    ax6 = plt.subplot(3, 2, 6)
    
    # Calculate KL divergence or histogram intersection for each coordinate
    dataset_names = list(all_stats.keys())
    n_datasets = len(dataset_names)
    overlap_matrix = np.zeros((n_datasets, n_datasets))
    
    for i in range(n_datasets):
        for j in range(n_datasets):
            if i == j:
                overlap_matrix[i, j] = 1.0
            else:
                # Calculate overlap using histogram intersection
                data_i = all_stats[dataset_names[i]]['3d_all'].reshape(-1, 3)
                data_j = all_stats[dataset_names[j]]['3d_all'].reshape(-1, 3)
                
                overlaps = []
                for axis in range(3):
                    hist_i, edges = np.histogram(data_i[:, axis], bins=50, density=True)
                    hist_j, _ = np.histogram(data_j[:, axis], bins=edges, density=True)
                    intersection = np.minimum(hist_i, hist_j).sum() * (edges[1] - edges[0])
                    overlaps.append(intersection)
                
                overlap_matrix[i, j] = np.mean(overlaps)
    
    sns.heatmap(overlap_matrix, annot=True, fmt='.3f', cmap='YlOrRd',
                xticklabels=dataset_names, yticklabels=dataset_names, ax=ax6)
    ax6.set_title('3D Distribution Overlap (Histogram Intersection)')
    
    plt.suptitle('Advanced Keypoint Distribution Analysis', fontsize=16)
    plt.tight_layout()
    
    # Print detailed statistics
    print("\n" + "="*80)
    print("DETAILED DISTRIBUTION ANALYSIS")
    print("="*80)
    
    # Print percentile comparisons
    print("\n2D Keypoint Percentiles (normalized):")
    percentiles = [5, 25, 50, 75, 95]
    for coord_idx, coord_name in enumerate(['X', 'Y']):
        print(f"\n{coord_name}-coordinate percentiles:")
        print(f"{'Dataset':<15}", end='')
        for p in percentiles:
            print(f"{f'P{p}':>10}", end='')
        print()
        print("-"*75)
        
        for name, stats in all_stats.items():
            data = stats['2d_all'][:, :, coord_idx].flatten()
            print(f"{name:<15}", end='')
            for p in percentiles:
                print(f"{np.percentile(data, p):>10.4f}", end='')
            print()
    
    print("\n3D Keypoint Percentiles (meters):")
    for coord_idx, coord_name in enumerate(['X', 'Y', 'Z']):
        print(f"\n{coord_name}-coordinate percentiles:")
        print(f"{'Dataset':<15}", end='')
        for p in percentiles:
            print(f"{f'P{p}':>10}", end='')
        print()
        print("-"*75)
        
        for name, stats in all_stats.items():
            data = stats['3d_all'][:, :, coord_idx].flatten()
            print(f"{name:<15}", end='')
            for p in percentiles:
                print(f"{np.percentile(data, p):>10.4f}", end='')
            print()
    
    plt.show()


if __name__ == "__main__":
    plot_advanced_distributions()