import torch
import torch.nn as nn

class MLPLifterRotationHead(nn.Module):
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