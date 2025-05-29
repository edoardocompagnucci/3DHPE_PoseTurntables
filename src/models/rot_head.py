import torch
import torch.nn as nn

class MLPLifterRotationHead(nn.Module):
    def __init__(self, num_joints=24, dropout_rate=0.25):
        super().__init__()
        
        input_size = num_joints * 2
        pos_output_size = num_joints * 3
        rot_output_size = num_joints * 6

        self.backbone = nn.Sequential(
            nn.Linear(input_size, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
        )

        self.position_head = nn.Linear(256, pos_output_size)

        self.rotation_head = nn.Linear(256, rot_output_size)
    
    def forward(self, x):
        if x.dim() > 2:
            batch_size = x.shape[0]
            x = x.reshape(batch_size, -1)
        
        features = self.backbone(x)
        
        positions = self.position_head(features)
        rotations = self.rotation_head(features)
        
        return {
            'positions': positions,
            'rotations': rotations
        }