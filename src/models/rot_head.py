import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    """Residual block with optional dimension change"""
    def __init__(self, in_dim, out_dim, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, out_dim)
        self.bn1 = nn.BatchNorm1d(out_dim)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        
        self.fc2 = nn.Linear(out_dim, out_dim)
        self.bn2 = nn.BatchNorm1d(out_dim)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        
        # Skip connection with dimension adjustment if needed
        self.skip = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()
        
    def forward(self, x):
        identity = self.skip(x)
        
        out = self.fc1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.dropout1(out)
        
        out = self.fc2(out)
        out = self.bn2(out)
        
        # Residual connection
        out = out + identity
        out = self.relu2(out)
        out = self.dropout2(out)
        
        return out


class MLPLifterRotationHead(nn.Module):
    """Original MLP architecture for baseline comparison"""
    def __init__(self, num_joints=24, dropout=0.15):
        super(MLPLifterRotationHead, self).__init__()
        input_dim = num_joints * 2

        self.backbone = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout * 0.75),

            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),

            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
        )

        self.position_head = nn.Linear(256, num_joints * 3)
        self.rotation_head = nn.Linear(256, num_joints * 6)

    def forward(self, joints_2d):
        B, J, _ = joints_2d.shape
        x = joints_2d.view(B, J * 2)
        feat = self.backbone(x)

        pos3d = self.position_head(feat)
        rot6d = self.rotation_head(feat)
        rot6d = rot6d.view(B, J, 6)

        return pos3d, rot6d


class MLPLifterResidualHead(nn.Module):
    """MLP with residual connections for 2D to 3D pose lifting"""
    def __init__(self, num_joints=24, dropout=0.25):
        super().__init__()
        input_dim = num_joints * 2
        
        # Initial projection
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Residual blocks with decreasing dimensions
        self.res_block1 = ResidualBlock(1024, 1024, dropout)
        self.res_block2 = ResidualBlock(1024, 512, dropout * 0.8)
        self.res_block3 = ResidualBlock(512, 384, dropout * 0.6)
        self.res_block4 = ResidualBlock(384, 256, dropout * 0.5)
        
        # Output heads with skip connection from input
        self.position_head = nn.Linear(256, num_joints * 3)
        self.rotation_head = nn.Linear(256, num_joints * 6)
        
        # Direct skip from 2D input to position prediction (preserves spatial info)
        self.input_to_pos_skip = nn.Linear(input_dim, num_joints * 3)
        
        # Initialize skip connection with small weights to start conservatively
        with torch.no_grad():
            self.input_to_pos_skip.weight.data *= 0.01
        
    def forward(self, joints_2d):
        B, J, _ = joints_2d.shape
        x = joints_2d.view(B, J * 2)
        
        # Store input for skip connection
        input_features = x
        
        # Main pathway
        x = self.input_proj(x)
        x = self.res_block1(x)
        x = self.res_block2(x)
        x = self.res_block3(x)
        x = self.res_block4(x)
        
        # Position prediction with input skip
        pos3d = self.position_head(x)
        pos3d_skip = self.input_to_pos_skip(input_features)
        pos3d = pos3d + 0.1 * pos3d_skip  # Weighted skip to preserve 2D structure
        
        # Rotation prediction (no skip - rotations are less directly related to 2D)
        rot6d = self.rotation_head(x)
        rot6d = rot6d.view(B, J, 6)
        
        return pos3d, rot6d  # pos3d is (B, 72), rot6d is (B, 24, 6)