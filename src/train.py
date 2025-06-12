import os
import sys
from datetime import datetime

import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import numpy as np

from data.synthetic_pose_dataset import SyntheticPoseDataset
from models.rot_head import MLPLifterRotationHead
from utils.losses import mpjpe_loss, combined_pose_loss, combined_pose_bone_loss
from utils.transforms import NormalizerJoints2d
from src.data.threedpw_dataset import create_domain_mixing_datasets
from scripts.evaluate import evaluate_model

def main():
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "data"))
    CHECKPOINT_ROOT = "checkpoints"

    # Hyperparameters
    BATCH_SIZE = 64  
    DROPOUT_RATE = 0.1
    LEARNING_RATE = 8e-5  # Lower since you're already at good performance
    WEIGHT_DECAY = 1e-4
    NUM_EPOCHS = 350  # Increased for better convergence
    NUM_JOINTS = 24
    IMG_SIZE = 512

    LOSS_POS_WEIGHT = 1.0
    LOSS_ROT_WEIGHT = 0.08  # Slightly reduced

    # Early stopping with separate counters
    early_stopping_patience_mixed = 40
    early_stopping_patience_3dpw = 25  # Stricter for 3DPW
    early_stopping_counter_mixed = 0
    early_stopping_counter_3dpw = 0

    # Domain schedule - more aggressive since you're at 125mm
    def get_domain_schedule(epoch):
        """Get real data ratio and bone weight for current epoch"""
        if epoch <= 30:
            return 0.25, 0.08  # Start at 25%
        elif epoch <= 60:
            return 0.30, 0.06
        elif epoch <= 90:
            return 0.35, 0.04
        elif epoch <= 120:
            return 0.40, 0.03
        elif epoch <= 150:
            return 0.45, 0.02
        else:
            return 0.50, 0.01  # Max out at 50%

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Training configuration:")
    print(f"  Epochs: {NUM_EPOCHS}")
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  Learning rate: {LEARNING_RATE}")
    print(f"  Starting from 125mm baseline")

    normalizer = NormalizerJoints2d(img_size=IMG_SIZE)

    # Initialize with first domain settings
    current_real_ratio, current_bone_weight = get_domain_schedule(1)
    
    train_dataset, val_dataset = create_domain_mixing_datasets(
        data_root=DATA_ROOT,
        real_data_ratio=current_real_ratio,
        transform=normalizer,
        min_confidence=0.3
    )

    train_loader = DataLoader(train_dataset,
                              batch_size=BATCH_SIZE,
                              shuffle=True,
                              num_workers=12,
                              pin_memory=True,
                              persistent_workers=True,
                              prefetch_factor=6)
    
    val_loader = DataLoader(val_dataset,
                            batch_size=BATCH_SIZE,
                            shuffle=False,
                            num_workers=12,
                            pin_memory=True,
                            persistent_workers=True,
                            prefetch_factor=6)

    model = MLPLifterRotationHead(num_joints=NUM_JOINTS, dropout=DROPOUT_RATE).to(device)

    model_optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    # Scheduler with shorter cycles for domain shifts
    model_scheduler = CosineAnnealingWarmRestarts(
        model_optimizer,
        T_0=30,  # Matches domain shift frequency
        T_mult=1,
        eta_min=1e-6
    )

    os.makedirs(CHECKPOINT_ROOT, exist_ok=True)
    exp_name = f"mlp_lifter_improved_domain_{datetime.now():%Y%m%d_%H%M%S}"
    experiment_dir = os.path.join(CHECKPOINT_ROOT, exp_name)
    os.makedirs(experiment_dir, exist_ok=True)
    print(f"Saving checkpoints to: {experiment_dir}")

    # Tracking variables
    best_val_mpjpe = float("inf")
    best_state = None
    best_3dpw_mpjpe = float("inf")
    best_3dpw_state = None
    best_3dpw_epoch = -1
    
    best_val_mpjpe_per_ratio = {}
    best_state_per_ratio = {}
    train_losses, val_losses = [], []
    train_rot_losses, val_rot_losses = [], []
    train_bone_losses, val_bone_losses = [], []
    syn_losses, real_losses = [], []
    epoch_3dpw_scores = []
    domain_gaps = []  # Track 3DPW - Mixed gap

    for epoch in range(1, NUM_EPOCHS + 1):
        # Get current domain settings
        target_real_ratio, target_bone_weight = get_domain_schedule(epoch)
        
        # Check if we need to update domain
        if target_real_ratio != current_real_ratio:
            # Save checkpoint before domain shift
            checkpoint_path = os.path.join(experiment_dir, 
                                         f"checkpoint_ratio_{current_real_ratio:.0%}_epoch_{epoch-1}.pth")
            torch.save({
                "model_state": model.state_dict(),
                "optimizer_state": model_optimizer.state_dict(),
                "epoch": epoch-1,
                "real_ratio": current_real_ratio,
                "val_mpjpe": best_val_mpjpe_per_ratio.get(current_real_ratio, float("inf")) * 1000
            }, checkpoint_path)
            print(f"💾 Saved checkpoint for {current_real_ratio:.0%} ratio")
            
            current_real_ratio = target_real_ratio
            
            # Reduce learning rate on domain shift
            for param_group in model_optimizer.param_groups:
                param_group['lr'] *= 0.9
            
            print(f"📈 Domain shift to {current_real_ratio:.1%} real data")
            print(f"🔧 Learning rate reduced to {model_optimizer.param_groups[0]['lr']:.6f}")
            print(f"🦴 Bone weight: {target_bone_weight:.3f}")
            
            train_dataset, val_dataset = create_domain_mixing_datasets(
                data_root=DATA_ROOT,
                real_data_ratio=current_real_ratio,
                transform=normalizer,
                min_confidence=0.3
            )

            train_loader = DataLoader(train_dataset,
                                    batch_size=BATCH_SIZE,
                                    shuffle=True,
                                    num_workers=12,
                                    pin_memory=True,
                                    persistent_workers=True,
                                    prefetch_factor=6)
            
            val_loader = DataLoader(val_dataset,
                                    batch_size=BATCH_SIZE,
                                    shuffle=False,
                                    num_workers=12,
                                    pin_memory=True,
                                    persistent_workers=True,
                                    prefetch_factor=6)
            
        # Training phase
        model.train()
        running_loss = 0.0
        running_pos_loss = 0.0
        running_rot_loss = 0.0
        running_bone_loss = 0.0
        syn_loss_sum, real_loss_sum = 0.0, 0.0
        syn_count, real_count = 0, 0
        
        for batch in train_loader:
            inputs = batch["joints_2d"].to(device)
            target_pos = batch["joints_3d_centered"].to(device)
            target_rot = batch["rot_6d"].to(device)
            dataset_types = batch["dataset_type"]

            pos3d, rot6d = model(inputs)

            pred_dict = {
                'positions': pos3d,
                'rotations': rot6d
            }
            target_dict = {
                'positions': target_pos.flatten(1),
                'rotations': target_rot
            }
            
            loss_dict = combined_pose_bone_loss(pred_dict, target_dict, 
                                pos_weight=LOSS_POS_WEIGHT, 
                                rot_weight=LOSS_ROT_WEIGHT,
                                bone_weight=target_bone_weight)
            
            loss = loss_dict['total']

            # Track per-dataset losses
            for i, dataset_type in enumerate(dataset_types):
                sample_loss = loss_dict['total'].item() / len(dataset_types)
                if dataset_type == 'synthetic':
                    syn_loss_sum += sample_loss
                    syn_count += 1
                else:
                    real_loss_sum += sample_loss
                    real_count += 1

            model_optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            model_optimizer.step()
            
            running_loss += loss.item()
            running_pos_loss += loss_dict['position'].item()
            running_rot_loss += loss_dict['rotation'].item()
            running_bone_loss += loss_dict['bone'].item()

        avg_train = running_loss / len(train_loader)
        avg_train_pos = running_pos_loss / len(train_loader)
        avg_train_rot = running_rot_loss / len(train_loader)
        avg_train_bone = running_bone_loss / len(train_loader)
        avg_syn_loss = syn_loss_sum / syn_count if syn_count > 0 else 0
        avg_real_loss = real_loss_sum / real_count if real_count > 0 else 0
        
        train_losses.append(avg_train)
        train_rot_losses.append(avg_train_rot)
        train_bone_losses.append(avg_train_bone)
        syn_losses.append(avg_syn_loss)
        real_losses.append(avg_real_loss)

        # Validation phase
        model.eval()
        running_val = 0.0
        running_val_pos = 0.0
        running_val_rot = 0.0
        running_val_bone = 0.0
        
        with torch.no_grad():
            for batch in val_loader:
                inputs = batch["joints_2d"].to(device)
                target_pos = batch["joints_3d_centered"].to(device)
                target_rot = batch["rot_6d"].to(device)
                
                pos3d, rot6d = model(inputs)
                
                pred_dict = {
                    'positions': pos3d,
                    'rotations': rot6d
                }
                target_dict = {
                    'positions': target_pos.flatten(1),
                    'rotations': target_rot
                }
                
                loss_dict = combined_pose_bone_loss(pred_dict, target_dict, 
                                  pos_weight=LOSS_POS_WEIGHT, 
                                  rot_weight=LOSS_ROT_WEIGHT,
                                  bone_weight=target_bone_weight)
                
                running_val += loss_dict['total'].item()
                running_val_pos += loss_dict['position'].item()
                running_val_rot += loss_dict['rotation'].item()
                running_val_bone += loss_dict['bone'].item()

        avg_val = running_val / len(val_loader)
        avg_val_pos = running_val_pos / len(val_loader)
        avg_val_rot = running_val_rot / len(val_loader)
        avg_val_bone = running_val_bone / len(val_loader)
        
        val_losses.append(avg_val)
        val_rot_losses.append(avg_val_rot)
        val_bone_losses.append(avg_val_bone)

        # Adaptive 3DPW evaluation frequency
        should_eval_3dpw = False
        if epoch < 50 and epoch % 10 == 0:
            should_eval_3dpw = True
        elif epoch < 100 and epoch % 5 == 0:
            should_eval_3dpw = True
        elif epoch >= 100 and epoch % 3 == 0:
            should_eval_3dpw = True
        
        # Always evaluate if we just had a domain shift
        if epoch > 1 and get_domain_schedule(epoch-1)[0] != current_real_ratio:
            should_eval_3dpw = True
        
        pure_3dpw_mpjpe = None
        if should_eval_3dpw and evaluate_model is not None:
            print(f"🔍 Evaluating on pure 3DPW...")
            try:
                # Save temporary checkpoint
                temp_checkpoint = {
                    "model_state": model.state_dict(),
                    "epoch": epoch
                }
                temp_path = os.path.join(experiment_dir, f"temp_epoch_{epoch}.pth")
                torch.save(temp_checkpoint, temp_path)
                
                # Evaluate on pure 3DPW
                results_3dpw = evaluate_model(temp_path, DATA_ROOT, dataset_type='3dpw')
                pure_3dpw_mpjpe = results_3dpw['overall_mpjpe_mm']
                epoch_3dpw_scores.append((epoch, pure_3dpw_mpjpe))
                
                # Clean up
                os.remove(temp_path)
                
                # Calculate domain gap
                val_mm = avg_val_pos * 1000.0
                domain_gap = pure_3dpw_mpjpe - val_mm
                domain_gaps.append((epoch, domain_gap))
                
            except Exception as e:
                print(f"   3DPW evaluation failed: {e}")

        # Convert to mm for display
        train_mm = avg_train_pos * 1000.0
        val_mm = avg_val_pos * 1000.0
        rot_train = avg_train_rot
        rot_val = avg_val_rot
        bone_train = avg_train_bone
        bone_val = avg_val_bone

        current_model_lr = model_optimizer.param_groups[0]['lr']

        # Print comprehensive status
        print(f"\nEpoch {epoch}/{NUM_EPOCHS} (Real: {current_real_ratio:.1%})")
        print(f"  Train - Pos: {train_mm:.1f}mm, Rot: {rot_train:.4f}, Bone: {bone_train:.4f}")
        print(f"  Val   - Pos: {val_mm:.1f}mm, Rot: {rot_val:.4f}, Bone: {bone_val:.4f}")
        print(f"  Syn/Real - Syn: {avg_syn_loss:.4f}, Real: {avg_real_loss:.4f}")
        if pure_3dpw_mpjpe is not None:
            domain_gap = pure_3dpw_mpjpe - val_mm
            print(f"  Pure 3DPW: {pure_3dpw_mpjpe:.1f}mm (gap: {domain_gap:+.1f}mm)")
        print(f"  Bone Weight: {target_bone_weight:.3f}, LR: {current_model_lr:.6f}")
        print(f"  Early Stop: Mixed={early_stopping_counter_mixed}/{early_stopping_patience_mixed}, " + 
              f"3DPW={early_stopping_counter_3dpw}/{early_stopping_patience_3dpw}")

        model_scheduler.step()

        # Track improvements
        improved_mixed = False
        improved_3dpw = False
        
        if avg_val_pos < best_val_mpjpe:
            best_val_mpjpe = avg_val_pos
            best_state = model.state_dict().copy()
            improved_mixed = True
            print(f"✓ New best mixed validation: {val_mm:.1f} mm")

        # Update 3DPW best
        if pure_3dpw_mpjpe is not None and pure_3dpw_mpjpe < best_3dpw_mpjpe:
            best_3dpw_mpjpe = pure_3dpw_mpjpe
            best_3dpw_state = model.state_dict().copy()
            best_3dpw_epoch = epoch
            improved_3dpw = True
            print(f"✓ NEW BEST 3DPW: {pure_3dpw_mpjpe:.1f} mm 🎯")

        # Update early stopping counters
        if improved_mixed:
            early_stopping_counter_mixed = 0
        else:
            early_stopping_counter_mixed += 1
            
        if pure_3dpw_mpjpe is not None:
            if improved_3dpw:
                early_stopping_counter_3dpw = 0
            else:
                early_stopping_counter_3dpw += 1
        
        # Check early stopping (both must plateau)
        if (early_stopping_counter_mixed >= early_stopping_patience_mixed and 
            early_stopping_counter_3dpw >= early_stopping_patience_3dpw):
            print(f"Early stopping triggered after {epoch} epochs")
            print(f"Mixed validation plateaued for {early_stopping_counter_mixed} epochs")
            print(f"3DPW performance plateaued for {early_stopping_counter_3dpw} epochs")
            break
                
        # Track best per ratio
        if current_real_ratio not in best_val_mpjpe_per_ratio or avg_val_pos < best_val_mpjpe_per_ratio[current_real_ratio]:
            best_val_mpjpe_per_ratio[current_real_ratio] = avg_val_pos
            best_state_per_ratio[current_real_ratio] = model.state_dict().copy()
            print(f"✓ New best for {current_real_ratio:.1%} real data: {val_mm:.1f} mm")

    # Model selection with relative comparison
    print(f"\n🎯 Model Selection:")
    print(f"   Best Mixed Validation: {best_val_mpjpe*1000:.1f} mm")
    if best_3dpw_state is not None:
        print(f"   Best Pure 3DPW: {best_3dpw_mpjpe:.1f} mm (epoch {best_3dpw_epoch})")

    # Smart model selection
    mixed_mpjpe_mm = best_val_mpjpe * 1000
    
    if best_3dpw_state is not None:
        # Calculate relative performance
        improvement_ratio = best_3dpw_mpjpe / mixed_mpjpe_mm
        
        # If 3DPW is within 15% of mixed performance, prefer it
        # This accounts for domain gap while preferring real-world performance
        if improvement_ratio < 1.15:
            model.load_state_dict(best_3dpw_state)
            final_score = best_3dpw_mpjpe
            selection_reason = "best_3dpw_performance"
            print(f"✅ Selected 3DPW-optimized model: {final_score:.1f} mm")
            print(f"   (Ratio: {improvement_ratio:.2f}, within 15% threshold)")
        else:
            model.load_state_dict(best_state)
            final_score = mixed_mpjpe_mm
            selection_reason = "best_mixed_validation"
            print(f"✅ Selected mixed validation model: {final_score:.1f} mm")
            print(f"   (3DPW ratio: {improvement_ratio:.2f}, exceeds 15% threshold)")
    else:
        model.load_state_dict(best_state)
        final_score = mixed_mpjpe_mm
        selection_reason = "best_mixed_validation"
        print(f"✅ Selected mixed validation model: {final_score:.1f} mm")

    print(f"\nTraining complete!")
    print(f"Final selection: {selection_reason} with {final_score:.1f} mm")

    # Create comprehensive visualization
    real_ratios = sorted(best_val_mpjpe_per_ratio.keys())
    ratio_epochs = []
    ratio_perfs = []

    for ratio in real_ratios:
        ratio_epochs.append(f"{ratio:.0%}")
        ratio_perfs.append(best_val_mpjpe_per_ratio[ratio] * 1000.0)

    fig = plt.figure(figsize=(20, 24))
    
    # Create a 4x2 grid
    ax1 = fig.add_subplot(4, 2, 1)
    ax2 = fig.add_subplot(4, 2, 2)
    ax3 = fig.add_subplot(4, 2, 3)
    ax4 = fig.add_subplot(4, 2, 4)
    ax5 = fig.add_subplot(4, 2, 5)
    ax6 = fig.add_subplot(4, 2, 6)
    ax7 = fig.add_subplot(4, 2, 7)

    # Position loss
    ax1.plot([l*1000.0 for l in [loss['position'] if isinstance(loss, dict) else loss for loss in train_losses]], 
             label="Train MPJPE", alpha=0.8)
    ax1.plot([l*1000.0 for l in [loss['position'] if isinstance(loss, dict) else loss for loss in val_losses]], 
             label="Val MPJPE", alpha=0.8)
    ax1.axhline(y=125, color='gray', linestyle='--', alpha=0.5, label='Previous best')
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("MPJPE (mm)")
    ax1.set_title("Position Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Rotation loss
    ax2.plot(train_rot_losses, label="Train Rot Loss", alpha=0.8)
    ax2.plot(val_rot_losses, label="Val Rot Loss", alpha=0.8)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Geodesic Loss")
    ax2.set_title("Rotation Loss")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Bone loss
    ax3.plot(train_bone_losses, label="Train Bone Loss", alpha=0.8)
    ax3.plot(val_bone_losses, label="Val Bone Loss", alpha=0.8)
    ax3.set_xlabel("Epoch")
    ax3.set_ylabel("Bone Length Loss")
    ax3.set_title("Bone Consistency Loss")
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Synthetic vs Real loss
    ax4.plot(syn_losses, label="Synthetic Loss", alpha=0.8)
    ax4.plot(real_losses, label="Real Loss", alpha=0.8)
    ax4.set_xlabel("Epoch")
    ax4.set_ylabel("Loss")
    ax4.set_title("Synthetic vs Real Loss")
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # Per-ratio performance
    ax5.bar(ratio_epochs, ratio_perfs, alpha=0.7, color='green')
    ax5.set_xlabel("Real Data Ratio")
    ax5.set_ylabel("Best MPJPE (mm)")
    ax5.set_title("Best Performance Per Domain Ratio")
    ax5.axhline(y=125, color='gray', linestyle='--', alpha=0.5)
    for i, v in enumerate(ratio_perfs):
        ax5.text(i, v + 1, f"{v:.1f}", ha='center')
    ax5.grid(True, alpha=0.3)
    
    # 3DPW performance over time
    if epoch_3dpw_scores:
        epochs_3dpw, scores_3dpw = zip(*epoch_3dpw_scores)
        ax6.plot(epochs_3dpw, scores_3dpw, 'o-', label="3DPW MPJPE", color='red')
        ax6.axhline(y=125, color='gray', linestyle='--', alpha=0.5, label='Previous best')
        ax6.set_xlabel("Epoch")
        ax6.set_ylabel("MPJPE (mm)")
        ax6.set_title("Pure 3DPW Performance Evolution")
        ax6.legend()
        ax6.grid(True, alpha=0.3)
    
    # Domain gap evolution
    if domain_gaps:
        gap_epochs, gaps = zip(*domain_gaps)
        ax7.plot(gap_epochs, gaps, 'o-', label="Domain Gap", color='purple')
        ax7.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax7.set_xlabel("Epoch")
        ax7.set_ylabel("Gap (mm)")
        ax7.set_title("Domain Gap (3DPW - Mixed)")
        ax7.legend()
        ax7.grid(True, alpha=0.3)
    
    plt.tight_layout()
    curve_path = os.path.join(experiment_dir, "learning_curves.png")
    plt.savefig(curve_path, dpi=150)
    print(f"Learning curves → {curve_path}")

    # Save final model with comprehensive metadata
    final_path = os.path.join(experiment_dir, "final_model.pth")
    torch.save({
        "epoch": epoch,
        "model_state": model.state_dict(),
        "model_optimizer_state": model_optimizer.state_dict(),
        "best_val_mpjpe": best_val_mpjpe,
        "best_3dpw_mpjpe": best_3dpw_mpjpe,
        "best_3dpw_epoch": best_3dpw_epoch,
        "final_score": final_score,
        "selection_reason": selection_reason,
        "epoch_3dpw_scores": epoch_3dpw_scores,
        "domain_gaps": domain_gaps,
        "best_val_rot_loss": avg_val_rot,
        "best_val_bone_loss": avg_val_bone,
        "best_val_mpjpe_per_ratio": best_val_mpjpe_per_ratio,
        "final_selected_ratio": current_real_ratio,
        "hyperparameters": {
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "dropout_rate": DROPOUT_RATE,
            "pos_weight": LOSS_POS_WEIGHT,
            "rot_weight": LOSS_ROT_WEIGHT,
            "bone_weight": target_bone_weight,
            "final_real_ratio": current_real_ratio,
            "scheduler": "CosineAnnealingWarmRestarts_improved",
            "selection_threshold": 0.15,
            "baseline_mpjpe": 125.0
        }
    }, final_path)
    print(f"Final model → {final_path}")
    
    # Print summary statistics
    print(f"\n📊 Training Summary:")
    print(f"   Total epochs: {epoch}")
    print(f"   Best mixed validation: {best_val_mpjpe*1000:.1f} mm")
    if best_3dpw_mpjpe < float('inf'):
        print(f"   Best pure 3DPW: {best_3dpw_mpjpe:.1f} mm")
        print(f"   Final domain gap: {best_3dpw_mpjpe - best_val_mpjpe*1000:.1f} mm")
    print(f"   Real data ratios tested: {sorted(best_val_mpjpe_per_ratio.keys())}")
    
    if epoch_3dpw_scores:
        print(f"\n   3DPW evaluation history:")
        for e, score in epoch_3dpw_scores[-5:]:  # Last 5 evaluations
            print(f"     Epoch {e}: {score:.1f} mm")

if __name__ == "__main__":
    main()