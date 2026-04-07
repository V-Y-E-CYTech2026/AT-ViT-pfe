import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

# git clone https://github.com/Blealtan/efficient-kan.git
from efficient_kan_module.src.efficient_kan.kan import KAN


class ATViT(nn.Module):
    """Cross Attention Vision Transformer with dual input branches and patch weighting mechanism."""
    def __init__(self, base_model, num_classes):
        super(ATViT, self).__init__()
        self.base_model = base_model
        self.num_classes = num_classes

        # Extract components from the base model
        self.patch_embed_orig = self.base_model.patch_embed[0]  # Small branch (240x240, patch_size=12)
        self.patch_embed_seg = self.base_model.patch_embed[1]   # Large branch (224x224, patch_size=16)
        self.pos_drop = self.base_model.pos_drop
        self.blocks = self.base_model.blocks
        self.norm = self.base_model.norm

        # Au lieu de : self.head = nn.Linear(1152, num_classes)
        # On définit un KAN avec potentiellement une couche cachée pour capter 
        # les relations non linéaires entre les features globales.
        # Format KAN : [input_dim, hidden_dim, output_dim]
        self.head = KAN([1152, 128, num_classes])  # Combined feature size (384 + 768)

        # Patch grid sizes (excluding CLS token)
        self.num_patches_orig = (240 // 12) ** 2  # 400 patches (20x20 grid)
        self.num_patches_seg = (224 // 16) ** 2   # 196 patches (14x14 grid)
        self.patch_size_orig = 12
        self.patch_size_seg = 16

        # Grid dimensions
        self.grid_h_orig = 240 // 12  # 20
        self.grid_w_orig = 240 // 12  # 20
        self.grid_h_seg = 224 // 16   # 14
        self.grid_w_seg = 224 // 16   # 14

        # Positional embeddings
        self.pos_embed_orig = nn.Parameter(torch.zeros(1, self.num_patches_orig + 1, 384))
        self.pos_embed_seg = nn.Parameter(torch.zeros(1, self.num_patches_seg + 1, 768))
        nn.init.trunc_normal_(self.pos_embed_orig, std=0.02)
        nn.init.trunc_normal_(self.pos_embed_seg, std=0.02)

        # CLS tokens for both branches
        self.cls_token_orig = nn.Parameter(torch.zeros(1, 1, 384))
        self.cls_token_seg = nn.Parameter(torch.zeros(1, 1, 768))
        nn.init.trunc_normal_(self.cls_token_orig, std=0.02)
        nn.init.trunc_normal_(self.cls_token_seg, std=0.02)

    def compute_patch_weights_from_segmented(self, seg_x, device):
        """Compute weight scalars for each patch based on segmented image content."""
        B, C, H, W = seg_x.shape
        grid_h = H // self.patch_size_seg  # 14
        grid_w = W // self.patch_size_seg  # 14

        # Unfold the segmented image into patches
        seg_patches = seg_x.unfold(2, self.patch_size_seg, self.patch_size_seg).unfold(3, self.patch_size_seg, self.patch_size_seg)
        seg_patches = seg_patches.contiguous().view(B, C, grid_h, grid_w, self.patch_size_seg, self.patch_size_seg)
        seg_patches = seg_patches.permute(0, 2, 3, 1, 4, 5).reshape(B, grid_h * grid_w, C, self.patch_size_seg, self.patch_size_seg)

        # Compute the number of non-black pixels per patch
        non_black_count = (seg_patches > 0).float().sum(dim=(2, 3, 4))  # Sum over channels, height, width
        total_pixels = self.patch_size_seg * self.patch_size_seg * C
        weight_scalars = non_black_count / total_pixels  # Normalize to [0, 1]

        # Apply sigmoid to make weights more pronounced and ensure they're in (0, 1)
        weight_scalars = torch.sigmoid(weight_scalars * 6 - 3)  # Scale and shift before sigmoid

        # Add small epsilon to prevent zero weights
        weight_scalars = weight_scalars + 0.05
        weight_scalars = weight_scalars / weight_scalars.max(dim=1, keepdim=True)[0]  # Normalize per batch

        return weight_scalars  # Shape: (B, 196)

    def upsample_weight_scalars(self, weight_scalars, device):
        """Upsample weight scalars from 14x14 (segmented) to 20x20 (original) grid."""
        B = weight_scalars.shape[0]

        # Reshape to 2D grid: (B, 14, 14)
        weight_grid = weight_scalars.view(B, self.grid_h_seg, self.grid_w_seg)

        # Add channel dimension for interpolation: (B, 1, 14, 14)
        weight_grid = weight_grid.unsqueeze(1)

        # Upsample to 20x20 using bilinear interpolation
        upsampled_grid = F.interpolate(
            weight_grid, 
            size=(self.grid_h_orig, self.grid_w_orig), 
            mode='bilinear', 
            align_corners=False
        )

        # Reshape back to flat: (B, 400)
        upsampled_scalars = upsampled_grid.squeeze(1).view(B, self.num_patches_orig)

        return upsampled_scalars

    def apply_patch_weights_to_embeddings(self, embeddings, weight_scalars):
        """Apply weight scalars to patch embeddings by element-wise multiplication."""
        weights_expanded = weight_scalars.unsqueeze(-1)  # (B, num_patches, 1)
        weighted_embeddings = embeddings * weights_expanded
        return weighted_embeddings

    def forward(self, x):
        """Forward pass through the AT-ViT model."""
        orig_x = x['original']  # (batch_size, 3, 240, 240)
        seg_x = x['segmented']  # (batch_size, 3, 224, 224)

        batch_size = orig_x.shape[0]
        device = orig_x.device

        # Compute patch weight scalars from segmented image
        seg_weight_scalars = self.compute_patch_weights_from_segmented(seg_x, device)  # (B, 196)
        orig_weight_scalars = self.upsample_weight_scalars(seg_weight_scalars, device)  # (B, 400)

        # Process original image through small branch
        orig_tokens = self.patch_embed_orig(orig_x)  # (batch_size, 400, 384)
        orig_tokens = self.apply_patch_weights_to_embeddings(orig_tokens, orig_weight_scalars)
        cls_tokens_orig = self.cls_token_orig.expand(batch_size, -1, -1)  # (batch_size, 1, 384)
        orig_tokens = torch.cat([cls_tokens_orig, orig_tokens], dim=1)  # (batch_size, 401, 384)
        orig_tokens = orig_tokens + self.pos_embed_orig
        orig_tokens = self.pos_drop(orig_tokens)

        # Process segmented image through large branch
        seg_tokens = self.patch_embed_seg(seg_x)  # (batch_size, 196, 768)
        seg_tokens = self.apply_patch_weights_to_embeddings(seg_tokens, seg_weight_scalars)
        cls_tokens_seg = self.cls_token_seg.expand(batch_size, -1, -1)  # (batch_size, 1, 768)
        seg_tokens = torch.cat([cls_tokens_seg, seg_tokens], dim=1)  # (batch_size, 197, 768)
        seg_tokens = seg_tokens + self.pos_embed_seg
        seg_tokens = self.pos_drop(seg_tokens)

        # Pass through blocks with cross-attention
        for block in self.blocks:
            orig_tokens, seg_tokens = block([orig_tokens, seg_tokens])

        # Normalize and combine features
        orig_features = self.norm[0](orig_tokens[:, 0])  # CLS token
        seg_features = self.norm[1](seg_tokens[:, 0])    # CLS token
        combined_features = torch.cat((orig_features, seg_features), dim=1)

        # Final classification
        output = self.head(combined_features)
        return output