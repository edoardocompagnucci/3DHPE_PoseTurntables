import torch
import torch.nn as nn

class MLPLifter(nn.Module):
    def __init__(self, num_joints=16):
        super().__init__()

        input_size = num_joints * 2
        output_size = num_joints * 3

        self.model = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.ReLU(),

            nn.Linear(512, 512),
            nn.ReLu(),

            nn.Linear(512, output_size)
        )
    
    def forward(self, x):

        if x.dim() > 2:
            batch_size = x.shape[0]
            x = x.reshape(batch_size, -1) #(batch size, num joints, 2) --> (batch size, num joints * 2)

        output = self.model(x)

        return output