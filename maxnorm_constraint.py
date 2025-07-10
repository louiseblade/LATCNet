import torch
import torch.nn as nn

def apply_max_norm_refined(
        max_norm_val, modules_to_apply, layers=(nn.Conv1d, nn.Conv2d, nn.Linear),
        skip_names=()
):
    with torch.no_grad():
        for parent_module in modules_to_apply:
            for name, module in parent_module.named_modules():
                if isinstance(module, layers) and not any(s in name for s in skip_names):
                    if hasattr(module, "weight") and module.weight is not None:
                        weight_flat = module.weight.view(module.weight.size(0), -1)
                        norms = weight_flat.norm(p=2, dim=1, keepdim=True)
                        desired = torch.clamp(norms, max=max_norm_val)
                        scale = desired / (norms + 1e-7)

                        # Same broadcast logic
                        scale_shape = [-1] + [1] * (module.weight.dim() - 1)
                        scale = scale.view(*scale_shape)
                        module.weight.mul_(scale)

def check_weight_norms(modules_to_apply, layers=(nn.Conv1d, nn.Conv2d, nn.Linear), max_norm_val=0.6):
    for parent_module in modules_to_apply:
        for name, module in parent_module.named_modules():
            if isinstance(module, layers):
                if hasattr(module, 'weight') and module.weight is not None:
                    weight_flat = module.weight.view(module.weight.size(0), -1)
                    norms = weight_flat.norm(p=2, dim=1)
                    if torch.any(norms > max_norm_val + 1e-4):
                        print(f"Layer {name} has weights exceeding max_norm: {norms.max().item():.4f}")
                    else:
                        print(f"Layer {name} norms are within the max_norm limit.")