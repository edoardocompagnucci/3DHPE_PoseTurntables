import torch
import torch.nn as nn

class MLPLifter(nn.Module):
    def __init__(self, num_joints=24, dropout_rate=0.3):
        super().__init__()

        input_size = num_joints * 2
        output_size = num_joints * 3

        self.model = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),  # Add dropout here
            
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),  # Add dropout here
            
            nn.Linear(512, 256),       # Add another layer for better capacity
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),  # Add dropout here
            
            nn.Linear(256, output_size)
        )
    
    def forward(self, x):

        if x.dim() > 2:
            batch_size = x.shape[0]
            x = x.reshape(batch_size, -1)

        output = self.model(x)

        return output
    
class MLPLifter_v2(nn.Module):
    """Step 1: Increase model capacity"""
    def __init__(self, num_joints=24, dropout_rate=0.3):
        super().__init__()

        input_size = num_joints * 2
        output_size = num_joints * 3

        self.model = nn.Sequential(
            # Layer 1: Bigger
            nn.Linear(input_size, 1024),  # 512 → 1024
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            # Layer 2: Bigger  
            nn.Linear(1024, 1024),        # 512 → 1024
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            # Layer 3: Same
            nn.Linear(1024, 512),         # 256 → 512 
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            # Layer 4: NEW layer
            nn.Linear(512, 256),          # New layer
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            # Output
            nn.Linear(256, output_size)
        )
    
    def forward(self, x):
        if x.dim() > 2:
            batch_size = x.shape[0]
            x = x.reshape(batch_size, -1)
        return self.model(x)