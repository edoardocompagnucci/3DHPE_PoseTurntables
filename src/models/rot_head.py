import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class PositionalEncoding(nn.Module):
    """Positional encoding for joint sequences"""
    
    def __init__(self, d_model, max_len=24):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]


class SpatialTransformerBlock(nn.Module):
    """Spatial transformer block for modeling joint relationships"""
    
    def __init__(self, d_model, nhead, dim_feedforward, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=False)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        # Self-attention
        src2, attention_weights = self.self_attn(src, src, src, attn_mask=src_mask,
                                               key_padding_mask=src_key_padding_mask)
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        
        # Feed forward
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        
        return src, attention_weights


class GradientReversalLayer(torch.autograd.Function):
    """FIXED: Gradient reversal for domain adversarial training"""
    
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.view_as(x)
    
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.lambda_, None


class DomainDiscriminator(nn.Module):
    """
    FIXED: Domain discriminator with improved architecture
    """
    def __init__(self, input_dim, hidden_dim=256, dropout=0.3):
        super().__init__()
        # Deeper network for better domain classification
        self.discriminator = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 4, 1),  # Binary classification
            nn.Sigmoid()
        )
        
        # Initialize to output ~0.5 initially
        with torch.no_grad():
            for module in self.discriminator:
                if isinstance(module, nn.Linear):
                    nn.init.xavier_normal_(module.weight, gain=0.1)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)
    
    def forward(self, x, lambda_=1.0):
        # Apply gradient reversal
        x_reversed = GradientReversalLayer.apply(x, lambda_)
        return self.discriminator(x_reversed)


class TransformerPoseLifter(nn.Module):
    """
    FIXED: Enhanced Transformer-based pose lifter with improved domain adversarial training
    """
    
    def __init__(self, num_joints=24, d_model=256, nhead=8, num_layers=6, 
                 dim_feedforward=1024, dropout=0.1, use_domain_discriminator=True):
        super().__init__()
        self.num_joints = num_joints
        self.d_model = d_model
        self.use_domain_discriminator = use_domain_discriminator
        
        # FIXED: Improved input projection with better regularization
        self.input_projection = nn.Sequential(
            nn.Linear(2, d_model // 2),
            nn.LayerNorm(d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(d_model // 2, d_model),
            nn.LayerNorm(d_model)
        )
        
        # Joint embedding for positional information
        self.joint_embedding = nn.Embedding(num_joints, d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, max_len=num_joints + 1)
        
        # Spatial transformer layers
        self.transformer_layers = nn.ModuleList([
            SpatialTransformerBlock(d_model, nhead, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])
        
        # Global feature aggregation token
        self.global_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        
        # FIXED: Improved output heads with better initialization
        self.position_head = nn.Sequential(
            nn.Linear(d_model, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 3)
        )
        
        self.rotation_head = nn.Sequential(
            nn.Linear(d_model, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 6)
        )
        
        # FIXED: Domain discriminator with better initialization
        if use_domain_discriminator:
            self.domain_discriminator = DomainDiscriminator(
                input_dim=d_model,
                hidden_dim=256,
                dropout=dropout * 1.5  # Higher dropout for discriminator
            )
        
        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """FIXED: Better weight initialization"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # Use Xavier initialization with small gain for stability
                nn.init.xavier_uniform_(module.weight, gain=0.8)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0, std=0.02)
        
        # FIXED: Initialize rotation head to output valid 6D rotations
        with torch.no_grad():
            # Last layer of rotation head should output near-identity rotations initially
            if hasattr(self.rotation_head[-1], 'weight'):
                self.rotation_head[-1].weight.data *= 0.1
                if self.rotation_head[-1].bias is not None:
                    # Initialize to identity rotation in 6D representation
                    bias = self.rotation_head[-1].bias.data.view(-1, 6)
                    bias[:, 0] = 1.0  # First column of rotation matrix
                    bias[:, 1] = 0.0
                    bias[:, 2] = 0.0
                    bias[:, 3] = 0.0  # Second column of rotation matrix  
                    bias[:, 4] = 1.0
                    bias[:, 5] = 0.0

    def forward(self, joints_2d, return_features=False, domain_lambda=1.0):
        """
        FIXED: Forward pass with improved stability
        
        Args:
            joints_2d: (batch_size, num_joints, 2)
            return_features: If True, also return features for analysis
            domain_lambda: Gradient reversal strength for domain adversarial training
            
        Returns:
            pos3d: (batch_size, num_joints * 3)
            rot6d: (batch_size, num_joints, 6) 
            domain_pred: (batch_size, 1) if domain discriminator is used
            features: global features if return_features is True
        """
        batch_size, num_joints, _ = joints_2d.shape
        
        # Input projection with residual-like connection
        x = self.input_projection(joints_2d)  # (B, J, d_model)
        
        # Add joint positional embeddings
        joint_indices = torch.arange(num_joints, device=joints_2d.device)
        joint_emb = self.joint_embedding(joint_indices).unsqueeze(0).expand(batch_size, -1, -1)
        x = x + joint_emb
        
        # Add global token for feature aggregation
        global_tokens = self.global_token.expand(batch_size, -1, -1)
        x = torch.cat([global_tokens, x], dim=1)  # (B, J+1, d_model)
        
        # Transpose for transformer (seq_len, batch, features)
        x = x.transpose(0, 1)
        
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Apply transformer layers with gradient checkpointing for memory efficiency
        attention_maps = []
        for layer in self.transformer_layers:
            if self.training and batch_size > 32:
                # Use gradient checkpointing for large batches
                x, attn_weights = torch.utils.checkpoint.checkpoint(layer, x, use_reentrant=False)
            else:
                x, attn_weights = layer(x)
            attention_maps.append(attn_weights)
        
        # Transpose back to (batch, seq_len, features)
        x = x.transpose(0, 1)
        
        # Split global and joint features
        global_features = x[:, 0]  # (B, d_model)
        joint_features = x[:, 1:]  # (B, J, d_model)
        
        # Predict 3D positions
        pos3d_per_joint = self.position_head(joint_features)  # (B, J, 3)
        pos3d = pos3d_per_joint.view(batch_size, -1)  # (B, J*3)
        
        # Predict 6D rotations
        rot6d = self.rotation_head(joint_features)  # (B, J, 6)
        
        outputs = [pos3d, rot6d]
        
        # FIXED: Domain discrimination with proper training logic
        if self.use_domain_discriminator:
            if self.training:
                # During training, always compute domain prediction
                domain_pred = self.domain_discriminator(global_features, domain_lambda)
                outputs.append(domain_pred)
            else:
                # During evaluation, don't compute domain prediction to save computation
                pass
        
        if return_features:
            outputs.append(global_features)
        
        return outputs if len(outputs) > 2 else (outputs[0], outputs[1])


class MLPLifterRotationHead(TransformerPoseLifter):
    """
    FIXED: Enhanced transformer model with optimal configuration for domain-breaking
    """
    def __init__(self, num_joints=24, dropout=0.1):
        # OPTIMIZED CONFIGURATION based on analysis
        super().__init__(
            num_joints=num_joints,
            d_model=384,           # Increased capacity
            nhead=12,              # More attention heads
            num_layers=8,          # Deeper network
            dim_feedforward=1536,  # Larger feedforward 
            dropout=dropout,
            use_domain_discriminator=True  # Enable domain adversarial training
        )
        print("🚀 Using FIXED Enhanced Transformer with Domain-Breaking")
        print(f"   Architecture: d_model=384, heads=12, layers=8")
        print(f"   Domain discriminator: ENABLED with improved initialization")
        print(f"   Key fixes: Joint indices, MPII-SMPL mapping, domain labels")
        print(f"   Expected performance: <100mm MPJPE on 3DPW")


# Alternative configurations for experimentation
class TransformerPoseLifterXL(TransformerPoseLifter):
    """Extra large model if the enhanced version still needs more capacity"""
    def __init__(self, num_joints=24, dropout=0.1):
        super().__init__(
            num_joints=num_joints,
            d_model=512,
            nhead=16,
            num_layers=10,
            dim_feedforward=2048,
            dropout=dropout,
            use_domain_discriminator=True
        )
        print("🚀 Using XL Transformer (if enhanced version needs more capacity)")


class TransformerPoseLifterSmall(TransformerPoseLifter):
    """Smaller model for comparison/ablation studies"""
    def __init__(self, num_joints=24, dropout=0.1):
        super().__init__(
            num_joints=num_joints,
            d_model=256,
            nhead=8,
            num_layers=6,
            dim_feedforward=1024,
            dropout=dropout,
            use_domain_discriminator=False
        )
        print("🔍 Using Small Transformer (for comparison)")


class SimpleMLP(nn.Module):
    """Simple MLP baseline for comparison"""
    def __init__(self, num_joints=24, hidden_dim=1024, dropout=0.1):
        super().__init__()
        self.num_joints = num_joints
        
        self.position_mlp = nn.Sequential(
            nn.Linear(num_joints * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(), 
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_joints * 3)
        )
        
        self.rotation_mlp = nn.Sequential(
            nn.Linear(num_joints * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout), 
            nn.Linear(hidden_dim, num_joints * 6)
        )
        
    def forward(self, joints_2d, **kwargs):
        batch_size = joints_2d.shape[0]
        joints_flat = joints_2d.view(batch_size, -1)
        
        pos3d = self.position_mlp(joints_flat)
        rot6d = self.rotation_mlp(joints_flat).view(batch_size, self.num_joints, 6)
        
        return pos3d, rot6d