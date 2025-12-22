import torch
import torch.nn as nn
import torch.nn.functional as F

# ==================== KAN Layer (Fixed Implementation) ====================
class KANConv2d(nn.Module):
    """
    Kolmogorov-Arnold Network Convolution Layer
    Uses learnable B-spline basis functions instead of fixed activations
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, 
                 spline_order=3, num_control_points=8):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.spline_order = spline_order
        self.num_control_points = num_control_points

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, 
                             padding=kernel_size//2, bias=False)
        
        # Learnable control points for B-spline basis functions, w/ each output channel having its own spline
        self.control_points = nn.Parameter(
            torch.randn(out_channels, num_control_points) * 0.1
        )
        
        # Grid for evaluating splines (fixed)
        self.register_buffer('grid', torch.linspace(-1, 1, 100))
        
    def b_spline_basis(self, x, control_points):
        """Compute B-spline basis function values"""
        # Normalize input to [-1, 1] range
        x_normalized = torch.tanh(x)
        
        # Simple cubic B-spline approximation using learnable control points
        n_points = control_points.shape[-1]
        basis_values = []
        
        for i in range(n_points - self.spline_order):
            # Distance to control point
            t = (x_normalized + 1) * (n_points - self.spline_order - 1) / 2
            basis = torch.clamp(1 - torch.abs(t - i), 0, 1) ** self.spline_order
            basis_values.append(basis)
        
        basis_tensor = torch.stack(basis_values, dim=-1)
        
        # Weighted sum of basis functions
        return torch.sum(basis_tensor * control_points[..., :len(basis_values)], dim=-1)
    
    def forward(self, x):
        # Apply base convolution
        conv_out = self.conv(x)
        
        b, c, h, w = conv_out.shape
        conv_flat = conv_out.view(b, c, -1)
        
        # Apply channel-wise spline transformation
        activated = torch.stack([
            self.b_spline_basis(conv_flat[:, i], self.control_points[i])
            for i in range(c)
        ], dim=1)
        
        return activated.view(b, c, h, w)


# ==================== Building Blocks ====================
class MultiPooling(nn.Module):
    """Combines Average and Max Pooling"""
    def __init__(self):
        super().__init__()
        self.avg_pool = nn.AvgPool2d(2, stride=2)
        self.max_pool = nn.MaxPool2d(2, stride=2)
    
    def forward(self, x):
        avg = self.avg_pool(x)
        max_p = self.max_pool(x)
        return torch.cat([avg, max_p], dim=1)


class UnifiedAttention(nn.Module):
    """Unified spatial and channel attention mechanism"""
    def __init__(self, channels):
        super().__init__()
        # Spatial attention
        self.spatial_conv = nn.Conv2d(1, 1, kernel_size=3, padding=1)
        
        # Channel attention
        self.channel_fc = nn.Linear(channels, channels)
        
    def forward(self, x):
        b, c, h, w = x.shape
        
        # Spatial attention
        spatial_map = torch.mean(x, dim=1, keepdim=True)
        spatial_att = torch.sigmoid(self.spatial_conv(spatial_map))
        
        # Channel attention
        channel_map = torch.mean(x, dim=[2, 3], keepdim=True)
        channel_att = torch.sigmoid(self.channel_fc(channel_map.squeeze(-1).squeeze(-1)))
        channel_att = channel_att.unsqueeze(-1).unsqueeze(-1)
        
        # Combine attentions
        return x * spatial_att * channel_att


class AttentionGate(nn.Module):
    """Attention Gate for skip connections"""
    def __init__(self, channels):
        super().__init__()
        self.conv_x = nn.Conv2d(channels, channels, kernel_size=1)
        self.conv_y = nn.Conv2d(channels, channels, kernel_size=1)
        self.conv_g = nn.Conv2d(channels, 1, kernel_size=1)
        self.bn = nn.BatchNorm2d(channels)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
    def forward(self, x, y, g):
        theta_x = self.conv_x(x)
        theta_y = self.conv_y(y)
        phi_g = self.conv_g(g)
        phi_g = self.upsample(phi_g)
        
        concat = theta_x + theta_y + phi_g
        concat = self.bn(concat)
        f = F.relu(concat)
        
        psi = x * y * f
        return psi


class ConvBlock(nn.Module):
    """Convolutional block with depthwise separable convolution"""
    def __init__(self, in_channels, out_channels, dropout=0.1, batch_norm=True):
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=3, 
                                   padding=1, groups=in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(in_channels) if batch_norm else nn.Identity()
        self.bn2 = nn.BatchNorm2d(out_channels) if batch_norm else nn.Identity()
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        
    def forward(self, x):
        x = self.depthwise(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pointwise(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.dropout(x)
        return x


class TransformerEncoder(nn.Module):
    """Transformer encoder with multi-scale features"""
    def __init__(self, channels, num_heads=8, dropout=0.1, scale_factor=1.0):
        super().__init__()
        self.scale_factor = scale_factor
        self.channels = channels
        
        self.mha = nn.MultiheadAttention(channels, num_heads, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(channels, channels * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(channels * 4, channels)
        )
        self.ln1 = nn.LayerNorm(channels)
        self.ln2 = nn.LayerNorm(channels)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        b, c, h, w = x.shape
        
        # Reshape for transformer
        x_flat = x.flatten(2).transpose(1, 2)  # (B, H*W, C)
        
        # Multi-head attention
        attn_out, _ = self.mha(x_flat, x_flat, x_flat)
        attn_out = self.dropout(attn_out)
        x_flat = self.ln1(x_flat + attn_out)
        
        # Feed-forward network
        ffn_out = self.ffn(x_flat)
        ffn_out = self.dropout(ffn_out)
        x_flat = self.ln2(x_flat + ffn_out)
        
        # Reshape back
        return x_flat.transpose(1, 2).reshape(b, c, h, w)


class WeightedSum(nn.Module):
    """Learnable weighted sum of multiple feature maps"""
    def __init__(self, num_features, channels):
        super().__init__()
        self.weights = nn.Parameter(torch.ones(num_features) / num_features)
        
    def forward(self, features):
        # Resize all features to the same size
        target_size = features[0].shape[2:]
        resized = []
        for f in features:
            if f.shape[2:] != target_size:
                f = F.interpolate(f, size=target_size, mode='bilinear', align_corners=True)
            resized.append(f)
        
        # Weighted sum
        weights = F.softmax(self.weights, dim=0)
        output = sum(w * f for w, f in zip(weights, resized))
        return output


# ==================== OMEGA-Net Model ====================
class OMEGANet(nn.Module):
    """
    OMEGA-Net: Dual-input U-Net with KAN convolutions and attention mechanisms
    """
    def __init__(self, in_channels=3, base_filters=64, num_heads=8, dropout=0.1):
        super().__init__()
        
        # ===== Encoder Stream 1 (Normalized images) =====
        self.enc1_1 = KANConv2d(in_channels, base_filters//4)
        self.pool1_1 = MultiPooling()
        
        self.enc2_1 = KANConv2d(base_filters//2, base_filters//2)  # *2 from MultiPooling
        self.pool2_1 = MultiPooling()
        
        self.enc3_1 = KANConv2d(base_filters, base_filters)
        self.pool3_1 = MultiPooling()
        
        self.enc4_1 = KANConv2d(base_filters*2, base_filters*2)
        self.pool4_1 = MultiPooling()
        
        # Transformer at bottleneck
        self.trans_fine_1 = TransformerEncoder(base_filters*4, num_heads, dropout, scale_factor=1.0)
        self.trans_coarse_1 = TransformerEncoder(base_filters*4, num_heads, dropout, scale_factor=0.5)
        self.weighted_sum_1 = WeightedSum(2, base_filters*4)
        
        self.bottleneck_1 = ConvBlock(base_filters*4, base_filters*4, dropout, batch_norm=True)
        
        # ===== Encoder Stream 2 (Hematoxylin images) =====
        self.enc1_2 = KANConv2d(in_channels, base_filters//4)
        self.pool1_2 = MultiPooling()
        
        self.enc2_2 = KANConv2d(base_filters//2, base_filters//2)
        self.pool2_2 = MultiPooling()
        
        self.enc3_2 = KANConv2d(base_filters, base_filters)
        self.pool3_2 = MultiPooling()
        
        self.enc4_2 = KANConv2d(base_filters*2, base_filters*2)
        self.pool4_2 = MultiPooling()
        
        # Transformer at bottleneck
        self.trans_fine_2 = TransformerEncoder(base_filters*4, num_heads, dropout, scale_factor=1.0)
        self.trans_coarse_2 = TransformerEncoder(base_filters*4, num_heads, dropout, scale_factor=0.5)
        self.weighted_sum_2 = WeightedSum(2, base_filters*4)
        
        self.bottleneck_2 = ConvBlock(base_filters*4, base_filters*4, dropout, batch_norm=True)
        
        # ===== Decoder with Attention Gates =====
        self.ag4 = AttentionGate(base_filters*2)
        self.up4 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.att4 = UnifiedAttention(base_filters*8)
        self.dec4 = ConvBlock(base_filters*8, base_filters*2, dropout, batch_norm=True)
        
        self.ag3 = AttentionGate(base_filters)
        self.up3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.att3 = UnifiedAttention(base_filters*4)
        self.dec3 = ConvBlock(base_filters*4, base_filters, dropout, batch_norm=True)
        
        self.ag2 = AttentionGate(base_filters//2)
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.att2 = UnifiedAttention(base_filters*2)
        self.dec2 = ConvBlock(base_filters*2, base_filters//2, dropout, batch_norm=True)
        
        self.ag1 = AttentionGate(base_filters//4)
        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.att1 = UnifiedAttention(base_filters)
        self.dec1 = ConvBlock(base_filters, base_filters//4, dropout, batch_norm=True)
        
        # Final output
        self.out_conv = nn.Conv2d(base_filters//4, 1, kernel_size=1)
        
    def forward(self, x1, x2):
        # ===== Encoder Stream 1 =====
        e1_1 = self.enc1_1(x1)
        p1_1 = self.pool1_1(e1_1)
        
        e2_1 = self.enc2_1(p1_1)
        p2_1 = self.pool2_1(e2_1)
        
        e3_1 = self.enc3_1(p2_1)
        p3_1 = self.pool3_1(e3_1)
        
        e4_1 = self.enc4_1(p3_1)
        p4_1 = self.pool4_1(e4_1)
        
        trans_f1 = self.trans_fine_1(p4_1)
        trans_c1 = self.trans_coarse_1(p4_1)
        trans_comb1 = self.weighted_sum_1([trans_f1, trans_c1])
        
        b1 = self.bottleneck_1(trans_comb1)
        
        # ===== Encoder Stream 2 =====
        e1_2 = self.enc1_2(x2)
        p1_2 = self.pool1_2(e1_2)
        
        e2_2 = self.enc2_2(p1_2)
        p2_2 = self.pool2_2(e2_2)
        
        e3_2 = self.enc3_2(p2_2)
        p3_2 = self.pool3_2(e3_2)
        
        e4_2 = self.enc4_2(p3_2)
        p4_2 = self.pool4_2(e4_2)
        
        trans_f2 = self.trans_fine_2(p4_2)
        trans_c2 = self.trans_coarse_2(p4_2)
        trans_comb2 = self.weighted_sum_2([trans_f2, trans_c2])
        
        b2 = self.bottleneck_2(trans_comb2)
        
        # ===== Bridge: Concatenate bottlenecks =====
        bridge = torch.cat([b1, b2], dim=1)
        
        # ===== Decoder with Attention =====
        ag4 = self.ag4(e4_1, e4_2, bridge)
        d4 = self.up4(bridge)
        d4 = self.att4(d4)
        d4 = torch.cat([ag4, d4], dim=1)
        d4 = self.dec4(d4)
        d4 = d4 * e4_1 * e4_2
        
        ag3 = self.ag3(e3_1, e3_2, d4)
        d3 = self.up3(d4)
        d3 = self.att3(d3)
        d3 = torch.cat([ag3, d3], dim=1)
        d3 = self.dec3(d3)
        d3 = d3 * e3_1 * e3_2
        
        ag2 = self.ag2(e2_1, e2_2, d3)
        d2 = self.up2(d3)
        d2 = self.att2(d2)
        d2 = torch.cat([ag2, d2], dim=1)
        d2 = self.dec2(d2)
        d2 = d2 * e2_1 * e2_2
        
        ag1 = self.ag1(e1_1, e1_2, d2)
        d1 = self.up1(d2)
        d1 = self.att1(d1)
        d1 = torch.cat([ag1, d1], dim=1)
        d1 = self.dec1(d1)
        d1 = d1 * e1_1 * e1_2
        
        # Final output
        out = self.out_conv(d1)
        return torch.sigmoid(out)
