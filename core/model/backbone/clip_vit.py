import torch
import torch.nn as nn
import clip

class CLIPViT(nn.Module):
    """CLIP ViT backbone"""
    
    def __init__(self, model_name='ViT-B/32', is_feature=False, avg_pool=False, **kwargs):
        super().__init__()
        
        # Load CLIP model
        model, _ = clip.load(model_name, device='cpu')  # 先加载到CPU
        self.visual = model.visual
        
        # Convert model to float32
        self.visual = self.visual.float()
        
        # Delete unused components
        del model.transformer
        del model.token_embedding
        del model.positional_embedding
        del model.ln_final
        del model.text_projection
        del model.logit_scale
        
        # Configuration
        self.is_feature = is_feature
        self.avg_pool = avg_pool
        
    def forward(self, x):
        # Ensure input size is correct
        if x.shape[-1] != 224 or x.shape[-2] != 224:
            raise ValueError(
                f"Input image size must be 224x224, but got {x.shape[-2]}x{x.shape[-1]}"
            )
            
        # Convert input to float32 if needed
        if x.dtype != torch.float32:
            x = x.float()
            
        # Forward pass
        features = self.visual(x)
        
        # Return features based on configuration
        if self.is_feature:
            return features
        elif self.avg_pool:
            return nn.functional.adaptive_avg_pool2d(features, 1).squeeze(-1).squeeze(-1)
        else:
            return features 