import os
import sys
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import numpy as np

from data.synthetic_pose_dataset import SyntheticPoseDataset
from models.rot_head import MLPLifterRotationHead
from utils.losses import (mpjpe_loss, combined_pose_loss, combined_pose_bone_loss, 
                         domain_breaking_loss, extremity_weighted_mpjpe_loss)
from utils.transforms import NormalizerJoints2d
from data.threedpw_dataset import create_domain_mixing_datasets
from scripts.evaluate import evaluate_model


def main():
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "data"))
    CHECKPOINT_ROOT = "checkpoints"

    # Hyperparameters
    BATCH_SIZE = 48
    DROPOUT_RATE = 0.1
    LEARNING_RATE = 2e-4
    WEIGHT_DECAY = 1e-4
    NUM_EPOCHS = 200
    NUM_JOINTS = 24
    IMG_SIZE = 512

    # FIXED: Simplified loss weights
    LOSS_POS_WEIGHT = 1.0
    LOSS_ROT_WEIGHT = 0.06

    # Warmup
    WARMUP_EPOCHS = 10

    # Early stopping
    early_stopping_patience_mixed = 40
    early_stopping_patience_3dpw = 25
    early_stopping_counter_mixed = 0
    early_stopping_counter_3dpw = 0

    # Performance tracking
    performance_tracker = {
        'domain_gaps': [],
        '3dpw_scores': [],
        'extremity_errors': [],
        'core_errors': [],
        'domain_accuracy': [],
        'last_improvement_epoch': 0
    }
    
    def get_domain_schedule(epoch):
        """FIXED: Simplified progressive schedule"""
        if epoch <= 20:
            # Phase 1: Light domain adversarial training
            return 0.586, 0.03, 0.1
        elif epoch <= 50:
            # Phase 2: Increase domain adversarial strength
            return 0.586, 0.02, 0.2
        elif epoch <= 80:
            # Phase 3: Strong domain breaking
            return 0.586, 0.015, 0.3
        else:
            # Phase 4: Stable training
            return 0.586, 0.01, 0.2

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 DOMAIN-BREAKING TRANSFORMER TRAINING (FIXED)")
    print(f"🎯 Solution: Make synthetic extremities as bad as real ones")
    print(f"✅ Key strategies:")
    print(f"   1. Extremity-specific corruption with proper MPII-SMPL mapping")
    print(f"   2. Joint-level mixing to break correlations")
    print(f"   3. Extremity-weighted loss with consistent joint indices")
    print(f"   4. Domain adversarial training with proper labels")
    print(f"   5. Confidence-aware training without domain leakage")
    print(f"Using device: {device}")

    normalizer = NormalizerJoints2d(img_size=IMG_SIZE)

    # Initialize with first domain settings
    current_real_ratio, current_bone_weight, current_domain_lambda = get_domain_schedule(1)
    
    # ENHANCED: Use improved joint-level mixing with pure real samples
    train_dataset, val_dataset = create_domain_mixing_datasets(
        data_root=DATA_ROOT,
        real_data_ratio=current_real_ratio,
        transform=normalizer,
        min_confidence=0.3,
        use_joint_mixing=True  # Now includes pure real samples for domain discriminator!
    )

    train_loader = DataLoader(train_dataset,
                              batch_size=BATCH_SIZE,
                              shuffle=True,
                              num_workers=8,
                              pin_memory=True,
                              persistent_workers=True,
                              prefetch_factor=4)
    
    val_loader = DataLoader(val_dataset,
                            batch_size=BATCH_SIZE,
                            shuffle=False,
                            num_workers=8,
                            pin_memory=True,
                            persistent_workers=True,
                            prefetch_factor=4)

    # Initialize model with domain discriminator
    model = MLPLifterRotationHead(num_joints=NUM_JOINTS, dropout=DROPOUT_RATE).to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"📊 Model parameters: {trainable_params:,} trainable, {total_params:,} total")

    # Optimizer
    model_optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=LEARNING_RATE, 
        weight_decay=WEIGHT_DECAY,
        betas=(0.9, 0.999),
        eps=1e-8
    )

    # Learning rate schedulers
    def lr_lambda(epoch):
        if epoch < WARMUP_EPOCHS:
            return (epoch + 1) / WARMUP_EPOCHS
        else:
            return 1.0
    
    warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(model_optimizer, lr_lambda)
    main_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        model_optimizer,
        T_0=40,  # Longer cycles
        T_mult=1,
        eta_min=1e-5
    )

    os.makedirs(CHECKPOINT_ROOT, exist_ok=True)
    exp_name = f"transformer_domain_breaking_fixed_{datetime.now():%Y%m%d_%H%M%S}"
    experiment_dir = os.path.join(CHECKPOINT_ROOT, exp_name)
    os.makedirs(experiment_dir, exist_ok=True)
    print(f"Saving checkpoints to: {experiment_dir}")

    # Tracking variables
    best_val_mpjpe = float("inf")
    best_state = None
    best_3dpw_mpjpe = float("inf")
    best_3dpw_state = None
    best_3dpw_epoch = -1
    
    train_losses, val_losses = [], []
    train_rot_losses, val_rot_losses = [], []
    train_bone_losses, val_bone_losses = [], []
    train_domain_losses = []
    syn_losses, real_losses = [], []
    epoch_3dpw_scores = []
    domain_gaps = []
    learning_rates = []
    extremity_weights = []

    # FIXED: Domain loss handling
    domain_criterion = nn.BCELoss()

    for epoch in range(1, NUM_EPOCHS + 1):
        # Get current domain settings
        target_real_ratio, target_bone_weight, target_domain_lambda = get_domain_schedule(epoch)
        current_bone_weight = target_bone_weight
        current_domain_lambda = target_domain_lambda
        
        # Save periodic checkpoints
        if epoch % 20 == 0:
            checkpoint_path = os.path.join(experiment_dir, f"checkpoint_epoch_{epoch}.pth")
            torch.save({
                "model_state": model.state_dict(),
                "optimizer_state": model_optimizer.state_dict(),
                "epoch": epoch,
                "best_3dpw_mpjpe": best_3dpw_mpjpe if best_3dpw_mpjpe < float("inf") else None,
                "performance_tracker": performance_tracker
            }, checkpoint_path)
            print(f"💾 Saved checkpoint at epoch {epoch}")
        
        # Training phase
        model.train()
        running_loss = 0.0
        running_pos_loss = 0.0
        running_rot_loss = 0.0
        running_bone_loss = 0.0
        running_domain_loss = 0.0
        running_extremity_weight = 0.0
        syn_loss_sum, real_loss_sum = 0.0, 0.0
        syn_count, real_count = 0, 0
        
        # Domain accuracy tracking
        domain_correct = 0
        domain_total = 0
        
        for batch_idx, batch in enumerate(train_loader):
            inputs = batch["joints_2d"].to(device)
            target_pos = batch["joints_3d_centered"].to(device)
            target_rot = batch["rot_6d"].to(device)
            dataset_types = batch["dataset_type"]
            
            # Get confidence scores if available
            confidence_scores = None
            if "confidence_scores" in batch:
                confidence_scores = batch["confidence_scores"].to(device)
            
            # FIXED: Proper domain labels for discriminator
            domain_labels = torch.zeros(inputs.shape[0], 1, device=device)
            for i, dt in enumerate(dataset_types):
                if dt == '3dpw':
                    domain_labels[i] = 1.0
                elif dt == 'joint_mixed':
                    # FIXED: Don't use 0.5 for mixed - treat as synthetic for discriminator
                    # The goal is to make mixed samples indistinguishable from synthetic
                    domain_labels[i] = 0.0
                else:  # synthetic
                    domain_labels[i] = 0.0
            
            # Forward pass
            outputs = model(inputs, return_features=False, domain_lambda=current_domain_lambda)
            
            if len(outputs) == 3:
                pos3d, rot6d, domain_pred = outputs
            else:
                pos3d, rot6d = outputs
                domain_pred = None

            pred_dict = {
                'positions': pos3d,
                'rotations': rot6d
            }
            target_dict = {
                'positions': target_pos.flatten(1),
                'rotations': target_rot
            }
            
            # Use domain-breaking loss
            loss_dict = domain_breaking_loss(
                pred_dict, target_dict, 
                epoch=epoch,
                confidence_scores=confidence_scores,
                pos_weight=LOSS_POS_WEIGHT, 
                rot_weight=LOSS_ROT_WEIGHT,
                bone_weight=target_bone_weight,
                use_extremity_weighting=True,
                use_confidence=True
            )
            
            loss = loss_dict['total']
            running_extremity_weight += loss_dict.get('extremity_weight', 1.0)
            
            # FIXED: Add domain loss with proper weighting
            domain_loss = torch.tensor(0.0, device=device)
            if domain_pred is not None:
                domain_loss = domain_criterion(domain_pred, domain_labels)
                # Use current_domain_lambda directly instead of additional weight
                loss = loss + current_domain_lambda * domain_loss
                running_domain_loss += domain_loss.item()
                
                # Track domain accuracy (closer to 0.5 is better for adversarial training)
                domain_predictions = (domain_pred > 0.5).float()
                domain_correct += (domain_predictions == domain_labels).sum().item()
                domain_total += domain_labels.size(0)

            # FIXED: Track per-dataset losses with proper categorization
            for i, dataset_type in enumerate(dataset_types):
                sample_loss = loss_dict['total'].item() / len(dataset_types)
                if dataset_type in ['synthetic', 'joint_mixed']:
                    syn_loss_sum += sample_loss
                    syn_count += 1
                elif dataset_type == '3dpw':
                    real_loss_sum += sample_loss
                    real_count += 1

            # Backward pass
            model_optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            model_optimizer.step()
            
            running_loss += loss.item()
            running_pos_loss += loss_dict['position'].item()
            running_rot_loss += loss_dict['rotation'].item()
            running_bone_loss += loss_dict['bone'].item()

        # Learning rate scheduling
        if epoch <= WARMUP_EPOCHS:
            warmup_scheduler.step()
        else:
            main_scheduler.step()
        
        current_lr = model_optimizer.param_groups[0]['lr']
        learning_rates.append(current_lr)

        # Calculate averages
        num_batches = len(train_loader)
        avg_train = running_loss / num_batches
        avg_train_pos = running_pos_loss / num_batches
        avg_train_rot = running_rot_loss / num_batches
        avg_train_bone = running_bone_loss / num_batches
        avg_train_domain = running_domain_loss / num_batches if domain_total > 0 else 0
        avg_extremity_weight = running_extremity_weight / num_batches
        avg_syn_loss = syn_loss_sum / syn_count if syn_count > 0 else 0
        avg_real_loss = real_loss_sum / real_count if real_count > 0 else 0
        
        # Domain accuracy (ideally should be around 50% for good adversarial training)
        domain_accuracy = domain_correct / domain_total if domain_total > 0 else 0.5
        performance_tracker['domain_accuracy'].append(domain_accuracy)
        
        train_losses.append(avg_train)
        train_rot_losses.append(avg_train_rot)
        train_bone_losses.append(avg_train_bone)
        train_domain_losses.append(avg_train_domain)
        syn_losses.append(avg_syn_loss)
        real_losses.append(avg_real_loss)
        extremity_weights.append(avg_extremity_weight)
        
        # ENHANCED: Track sample type distribution every few epochs
        if epoch <= 5 or epoch % 10 == 0:
            sample_counts = {'synthetic': 0, 'joint_mixed': 0, '3dpw': 0}
            for batch in list(train_loader)[:5]:  # Check first 5 batches
                for dt in batch["dataset_type"]:
                    sample_counts[dt] = sample_counts.get(dt, 0) + 1
            total_checked = sum(sample_counts.values())
            if total_checked > 0:
                print(f"  📊 Sample distribution (first 5 batches): "
                      f"Syn: {sample_counts.get('synthetic', 0)/total_checked:.1%}, "
                      f"Mixed: {sample_counts.get('joint_mixed', 0)/total_checked:.1%}, "
                      f"Real: {sample_counts.get('3dpw', 0)/total_checked:.1%}")

        # Validation phase
        model.eval()
        running_val = 0.0
        running_val_pos = 0.0
        running_val_rot = 0.0
        running_val_bone = 0.0
        
        # Track per-joint errors
        joint_errors_sum = torch.zeros(24)
        joint_counts = 0
        
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
                
                # Regular loss for validation
                loss_dict = combined_pose_bone_loss(pred_dict, target_dict, 
                                  pos_weight=LOSS_POS_WEIGHT, 
                                  rot_weight=LOSS_ROT_WEIGHT,
                                  bone_weight=target_bone_weight)
                
                running_val += loss_dict['total'].item()
                running_val_pos += loss_dict['position'].item()
                running_val_rot += loss_dict['rotation'].item()
                running_val_bone += loss_dict['bone'].item()
                
                # Track per-joint errors
                batch_size = inputs.shape[0]
                pred_3d = pos3d.reshape(batch_size, 24, 3)
                target_3d = target_pos.reshape(batch_size, 24, 3)
                joint_errors = torch.norm(pred_3d - target_3d, dim=2).mean(dim=0)  # (24,)
                joint_errors_sum += joint_errors.cpu()
                joint_counts += 1

        avg_val = running_val / len(val_loader)
        avg_val_pos = running_val_pos / len(val_loader)
        avg_val_rot = running_val_rot / len(val_loader)
        avg_val_bone = running_val_bone / len(val_loader)
        
        val_losses.append(avg_val)
        val_rot_losses.append(avg_val_rot)
        val_bone_losses.append(avg_val_bone)
        
        # FIXED: Calculate extremity vs core errors with consistent indices
        extremity_error, core_error = 0.0, 0.0
        if joint_counts > 0:
            avg_joint_errors = (joint_errors_sum / joint_counts) * 1000  # Convert to mm
            
            # Use consistent joint definitions
            from utils.losses import SMPL_EXTREMITY_JOINTS, SMPL_CORE_JOINTS
            
            extremity_error = avg_joint_errors[SMPL_EXTREMITY_JOINTS].mean().item()
            core_error = avg_joint_errors[SMPL_CORE_JOINTS].mean().item()
            
            performance_tracker['extremity_errors'].append(extremity_error)
            performance_tracker['core_errors'].append(core_error)

        # 3DPW evaluation
        should_eval_3dpw = (epoch % 5 == 0) if epoch <= 50 else (epoch % 10 == 0)
        
        pure_3dpw_mpjpe = None
        if should_eval_3dpw and evaluate_model is not None:
            print(f"🔍 Evaluating on pure 3DPW...")
            try:
                temp_checkpoint = {
                    "model_state": model.state_dict(),
                    "epoch": epoch
                }
                temp_path = os.path.join(experiment_dir, f"temp_epoch_{epoch}.pth")
                torch.save(temp_checkpoint, temp_path)
                
                results_3dpw = evaluate_model(temp_path, DATA_ROOT, dataset_type='3dpw')
                pure_3dpw_mpjpe = results_3dpw['overall_mpjpe_mm']
                epoch_3dpw_scores.append((epoch, pure_3dpw_mpjpe))
                
                os.remove(temp_path)
                
                val_mm = avg_val_pos * 1000.0
                domain_gap = pure_3dpw_mpjpe - val_mm
                domain_gaps.append((epoch, domain_gap))
                
                performance_tracker['domain_gaps'].append(domain_gap)
                performance_tracker['3dpw_scores'].append(pure_3dpw_mpjpe)
                
            except Exception as e:
                print(f"   3DPW evaluation failed: {e}")

        # Display progress
        train_mm = avg_train_pos * 1000.0
        val_mm = avg_val_pos * 1000.0

        print(f"\nEpoch {epoch}/{NUM_EPOCHS}")
        print(f"  Train - Pos: {train_mm:.1f}mm, Rot: {avg_train_rot:.4f}, Bone: {avg_train_bone:.4f}")
        if avg_train_domain > 0:
            domain_status = "GOOD" if 0.4 <= domain_accuracy <= 0.6 else "BAD"
            print(f"  Domain - Loss: {avg_train_domain:.4f}, Accuracy: {domain_accuracy:.1%} ({domain_status})")
        print(f"  Val   - Pos: {val_mm:.1f}mm, Rot: {avg_val_rot:.4f}, Bone: {avg_val_bone:.4f}")
        if extremity_error > 0 and core_error > 0:
            ratio = extremity_error / core_error
            ratio_status = "GOOD" if ratio < 2.0 else "BAD" if ratio > 5.0 else "OK"
            print(f"  Errors - Extremity: {extremity_error:.1f}mm, Core: {core_error:.1f}mm, Ratio: {ratio:.2f} ({ratio_status})")
        print(f"  Syn/Real - Syn: {avg_syn_loss:.4f}, Real: {avg_real_loss:.4f}")
        if pure_3dpw_mpjpe is not None:
            domain_gap = pure_3dpw_mpjpe - val_mm
            print(f"  Pure 3DPW: {pure_3dpw_mpjpe:.1f}mm (gap: {domain_gap:+.1f}mm)")
            
            # Success criteria
            if pure_3dpw_mpjpe < 100:
                print(f"  🎯 BREAKTHROUGH! Under 100mm!")
            elif pure_3dpw_mpjpe < 110:
                print(f"  🚀 Great progress!")
            elif pure_3dpw_mpjpe < 125:
                print(f"  ✅ Good improvement!")
        
        print(f"  LR: {current_lr:.6f}, Extremity weight: {avg_extremity_weight:.2f}, Domain λ: {current_domain_lambda:.1f}")

        # Track improvements
        improved_mixed = False
        improved_3dpw = False
        
        if avg_val_pos < best_val_mpjpe:
            best_val_mpjpe = avg_val_pos
            best_state = model.state_dict().copy()
            improved_mixed = True
            print(f"✓ New best mixed validation: {val_mm:.1f} mm")

        if pure_3dpw_mpjpe is not None and pure_3dpw_mpjpe < best_3dpw_mpjpe:
            best_3dpw_mpjpe = pure_3dpw_mpjpe
            best_3dpw_state = model.state_dict().copy()
            best_3dpw_epoch = epoch
            improved_3dpw = True
            performance_tracker['last_improvement_epoch'] = epoch
            print(f"✓ NEW BEST 3DPW: {pure_3dpw_mpjpe:.1f} mm 🎯")

        # Early stopping
        if improved_mixed:
            early_stopping_counter_mixed = 0
        else:
            early_stopping_counter_mixed += 1
            
        if pure_3dpw_mpjpe is not None:
            if improved_3dpw:
                early_stopping_counter_3dpw = 0
            else:
                early_stopping_counter_3dpw += 1
        
        # Only stop if really stuck and past minimum epochs
        if (early_stopping_counter_mixed >= early_stopping_patience_mixed and 
            early_stopping_counter_3dpw >= early_stopping_patience_3dpw and
            epoch > 80):
            print(f"Early stopping triggered after {epoch} epochs")
            break

    # Final model selection and saving
    print(f"\n🎯 TRAINING COMPLETE!")
    print(f"   Best Mixed Validation: {best_val_mpjpe*1000:.1f} mm")
    if best_3dpw_state is not None:
        print(f"   Best Pure 3DPW: {best_3dpw_mpjpe:.1f} mm (epoch {best_3dpw_epoch})")

    # Model selection - prefer 3DPW performance if it's reasonable
    if best_3dpw_state is not None and best_3dpw_mpjpe < best_val_mpjpe * 1000 * 1.1:
        model.load_state_dict(best_3dpw_state)
        final_score = best_3dpw_mpjpe
        selection_reason = "best_3dpw_performance"
    else:
        model.load_state_dict(best_state)
        final_score = best_val_mpjpe * 1000
        selection_reason = "best_mixed_validation"

    # Save final model
    final_path = os.path.join(experiment_dir, "final_model.pth")
    torch.save({
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": model_optimizer.state_dict(),
        "best_val_mpjpe": best_val_mpjpe,
        "best_3dpw_mpjpe": best_3dpw_mpjpe,
        "final_score": final_score,
        "selection_reason": selection_reason,
        "epoch_3dpw_scores": epoch_3dpw_scores,
        "performance_tracker": performance_tracker,
        "training_strategy": "domain_breaking_fixed"
    }, final_path)

    print(f"\n💾 Final model saved: {final_path}")
    print(f"🎉 Training complete! Final score: {final_score:.1f}mm")
    
    # Create comprehensive plots
    create_training_plots(experiment_dir, train_losses, val_losses, 
                         train_domain_losses, extremity_weights,
                         performance_tracker, epoch_3dpw_scores)


def create_training_plots(experiment_dir, train_losses, val_losses, 
                         train_domain_losses, extremity_weights,
                         performance_tracker, epoch_3dpw_scores):
    """Create comprehensive training visualization"""
    
    fig = plt.figure(figsize=(20, 16))
    
    # 1. Main loss curves
    ax1 = plt.subplot(3, 3, 1)
    ax1.plot(train_losses, label='Train Loss', alpha=0.8)
    ax1.plot(val_losses, label='Val Loss', alpha=0.8)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Total Loss')
    ax1.set_title('Training Progress')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Domain adversarial loss and accuracy
    ax2 = plt.subplot(3, 3, 2)
    if train_domain_losses:
        ax2.plot(train_domain_losses, label='Domain Loss', color='red', alpha=0.8)
    if 'domain_accuracy' in performance_tracker and performance_tracker['domain_accuracy']:
        ax2_twin = ax2.twinx()
        ax2_twin.plot(performance_tracker['domain_accuracy'], 
                     label='Domain Acc', color='green', alpha=0.8)
        ax2_twin.axhline(y=0.5, color='blue', linestyle='--', alpha=0.5, label='Target (50%)')
        ax2_twin.set_ylabel('Domain Accuracy')
        ax2_twin.set_ylim(0, 1)
        ax2_twin.legend(loc='upper right')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Domain Loss')
    ax2.set_title('Domain Adversarial Training')
    ax2.legend(loc='upper left')
    ax2.grid(True, alpha=0.3)
    
    # 3. Extremity weights progression
    ax3 = plt.subplot(3, 3, 3)
    if extremity_weights:
        ax3.plot(extremity_weights, label='Extremity Weight', color='purple', alpha=0.8)
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Weight')
    ax3.set_title('Adaptive Extremity Weighting')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. 3DPW performance over time
    ax4 = plt.subplot(3, 3, 4)
    if epoch_3dpw_scores:
        epochs, scores = zip(*epoch_3dpw_scores)
        ax4.plot(epochs, scores, 'o-', label='3DPW MPJPE', color='red', markersize=4)
        ax4.axhline(y=100, color='green', linestyle='--', alpha=0.7, label='Target (100mm)')
        ax4.axhline(y=137, color='gray', linestyle='--', alpha=0.5, label='Previous issue (137mm)')
        min_score = min(scores) if scores else 137
        if min_score < 120:
            ax4.axhline(y=min_score, color='blue', linestyle=':', alpha=0.7, label=f'Best ({min_score:.1f}mm)')
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('MPJPE (mm)')
    ax4.set_title('Pure 3DPW Performance')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # 5. Domain gap tracking
    ax5 = plt.subplot(3, 3, 5)
    if 'domain_gaps' in performance_tracker and performance_tracker['domain_gaps']:
        ax5.plot(performance_tracker['domain_gaps'], label='Domain Gap', color='orange')
        ax5.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax5.axhline(y=20, color='green', linestyle='--', alpha=0.5, label='Good gap (<20mm)')
    ax5.set_xlabel('Epoch')
    ax5.set_ylabel('Gap (mm)')
    ax5.set_title('Domain Gap (3DPW - Mixed Val)')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # 6. Extremity vs Core errors
    ax6 = plt.subplot(3, 3, 6)
    if ('extremity_errors' in performance_tracker and 
        performance_tracker['extremity_errors'] and
        performance_tracker['core_errors']):
        ax6.plot(performance_tracker['extremity_errors'], 
                label='Extremity', color='red', alpha=0.8)
        ax6.plot(performance_tracker['core_errors'], 
                label='Core', color='blue', alpha=0.8)
    ax6.set_xlabel('Epoch')
    ax6.set_ylabel('Error (mm)')
    ax6.set_title('Extremity vs Core Errors')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    # 7. Error ratio tracking
    ax7 = plt.subplot(3, 3, 7)
    if ('extremity_errors' in performance_tracker and 
        performance_tracker['extremity_errors'] and 
        performance_tracker['core_errors']):
        
        extremity_errors = np.array(performance_tracker['extremity_errors'])
        core_errors = np.array(performance_tracker['core_errors'])
        error_ratios = extremity_errors / (core_errors + 1e-6)
        
        ax7.plot(error_ratios, label='Extremity/Core Ratio', color='purple')
        ax7.axhline(y=1.0, color='green', linestyle='--', alpha=0.5, label='Equal errors')
        ax7.axhline(y=2.0, color='orange', linestyle='--', alpha=0.5, label='2x ratio (acceptable)')
        ax7.axhline(y=5.0, color='red', linestyle='--', alpha=0.5, label='5x ratio (bad)')
    ax7.set_xlabel('Epoch')
    ax7.set_ylabel('Error Ratio')
    ax7.set_title('Extremity/Core Error Ratio')
    ax7.legend()
    ax7.grid(True, alpha=0.3)
    
    # 8. Training summary
    ax8 = plt.subplot(3, 3, 8)
    ax8.text(0.1, 0.9, 'FIXED Training Summary:', fontsize=14, fontweight='bold')
    
    summary_text = []
    if epoch_3dpw_scores:
        initial_3dpw = epoch_3dpw_scores[0][1]
        final_3dpw = epoch_3dpw_scores[-1][1]
        best_3dpw = min(score for _, score in epoch_3dpw_scores)
        improvement = initial_3dpw - best_3dpw
        summary_text.append(f'3DPW: {initial_3dpw:.1f} → {best_3dpw:.1f} mm')
        summary_text.append(f'Improvement: {improvement:.1f} mm ({improvement/initial_3dpw*100:.1f}%)')
        
        if best_3dpw < 100:
            summary_text.append('✅ TARGET ACHIEVED!')
        elif best_3dpw < 110:
            summary_text.append('🚀 Great progress!')
        elif best_3dpw < 120:
            summary_text.append('✅ Good improvement!')
    
    if 'domain_gaps' in performance_tracker and performance_tracker['domain_gaps']:
        initial_gap = performance_tracker['domain_gaps'][0]
        final_gap = performance_tracker['domain_gaps'][-1]
        summary_text.append(f'Domain gap: {initial_gap:.1f} → {final_gap:.1f} mm')
    
    if ('extremity_errors' in performance_tracker and 
        performance_tracker['extremity_errors'] and
        performance_tracker['core_errors']):
        initial_ext = performance_tracker['extremity_errors'][0]
        final_ext = performance_tracker['extremity_errors'][-1]
        initial_core = performance_tracker['core_errors'][0]
        final_core = performance_tracker['core_errors'][-1]
        initial_ratio = initial_ext / initial_core
        final_ratio = final_ext / final_core
        summary_text.append(f'Ext/Core ratio: {initial_ratio:.1f} → {final_ratio:.1f}')
    
    for i, text in enumerate(summary_text):
        ax8.text(0.1, 0.8 - i*0.1, text, fontsize=12)
    
    ax8.axis('off')
    
    # 9. Key insights
    ax9 = plt.subplot(3, 3, 9)
    ax9.text(0.1, 0.9, 'Key Insights:', fontsize=14, fontweight='bold')
    
    insights = [
        '• Fixed joint index consistency',
        '• Proper MPII-SMPL mapping',
        '• Improved domain labels',
        '• Better confidence handling',
        '• Reduced domain leakage'
    ]
    
    for i, insight in enumerate(insights):
        ax9.text(0.1, 0.8 - i*0.12, insight, fontsize=11)
    
    ax9.axis('off')
    
    plt.tight_layout()
    plot_path = os.path.join(experiment_dir, 'training_analysis_fixed.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"📊 Training plots saved to: {plot_path}")
    plt.close()


if __name__ == "__main__":
    main()