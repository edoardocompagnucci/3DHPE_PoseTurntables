import torch
from torch.utils.data import DataLoader
import os
from datetime import datetime
import matplotlib.pyplot as plt
from data.synthetic_pose_dataset import SyntheticPoseDataset
from models.lifter import MLPLifter 
from utils.losses import mpjpe_loss

def main():
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "data"))

    BATCH_SIZE = 64
    LEARNING_RATE = 0.001
    NUM_EPOCHS = 5
    NUM_JOINTS = 16

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_dataset = SyntheticPoseDataset(
        data_root=DATA_ROOT,
        split_txt=os.path.join(DATA_ROOT, "splits", "train.txt")
    )

    val_dataset = SyntheticPoseDataset(
        data_root=DATA_ROOT,
        split_txt=os.path.join(DATA_ROOT, "splits", "val.txt")
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    model = MLPLifter(num_joints=NUM_JOINTS).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )

    os.makedirs("checkpoints", exist_ok=True)
    experiment_name = f"mlp_lifter_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    experiment_dir = os.path.join("checkpoints", experiment_name)
    os.makedirs(experiment_dir, exist_ok=True)
    print(f"Saving checkpoints to: {experiment_dir}")

    train_losses = []
    val_losses = []

    best_val_mpjpe = float('inf')
    best_model_state = None

    for epoch in range(NUM_EPOCHS):

        model.train()
        train_loss = 0.0
        num_train_batches = 0
        
        for batch in train_loader:
            
            inputs_2d = batch["joints_2d"].to(device)
            targets_3d = batch["joints_3d"].to(device)
            
            predictions_3d = model(inputs_2d)
            
            loss = mpjpe_loss(predictions_3d, targets_3d)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            num_train_batches += 1
        
        model.eval()
        val_loss = 0.0
        num_val_batches = 0
        
        with torch.no_grad():
            for batch in val_loader:

                inputs_2d = batch["joints_2d"].to(device)
                targets_3d = batch["joints_3d"].to(device)
                
                predictions_3d = model(inputs_2d)

                loss = mpjpe_loss(predictions_3d, targets_3d)

                val_loss += loss.item()
                num_val_batches += 1
        
        avg_train_loss = train_loss / num_train_batches
        avg_val_loss = val_loss / num_val_batches

        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        
        print(f"Epoch {epoch+1}/{NUM_EPOCHS}, "
            f"Train MPJPE: {avg_train_loss:.2f} mm, "
            f"Val MPJPE: {avg_val_loss:.2f} mm")

        scheduler.step(avg_val_loss)
        
        if avg_val_loss < best_val_mpjpe:
            best_val_mpjpe = avg_val_loss
            best_model_state = model.state_dict().copy()
            
            checkpoint_path = os.path.join(experiment_dir, f"best_model_epoch{epoch+1}_mpjpe{avg_val_loss:.2f}.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
            }, checkpoint_path)
            
            print(f"New best model with Val MPJPE: {best_val_mpjpe:.2f} mm saved to {checkpoint_path}")

    model.load_state_dict(best_model_state)
    print(f"Training complete! Best validation MPJPE: {best_val_mpjpe:.2f} mm")

    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training MPJPE')
    plt.plot(val_losses, label='Validation MPJPE')
    plt.xlabel('Epochs')
    plt.ylabel('MPJPE (mm)')
    plt.title('Training and Validation MPJPE Over Time')
    plt.legend()
    plt.grid(True)

    plot_path = os.path.join(experiment_dir, 'learning_curve.png')
    plt.savefig(plot_path)
    plt.close()
    print(f"Learning curve saved to {plot_path}")

    final_path = os.path.join(experiment_dir, "final_model.pth")

    torch.save({
        'epoch': NUM_EPOCHS,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'final_train_loss': avg_train_loss,
        'final_val_loss': avg_val_loss,
        'best_val_mpjpe': best_val_mpjpe,
    }, final_path)

    print(f"Final model saved to {final_path}")

if __name__ == "__main__":
    main()