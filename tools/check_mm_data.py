from safetensors.torch import load_file
import torch, torch.nn.functional as F
data = load_file('/apdcephfs_fsgm3/share_303700817/patchychen/ImageCaption/qwen36_dflash/gen/qwen36_imagecaption_eagle3_v5-1_051937_layerids/hidden_states/hs_17526.safetensors')
img = (data['token_ids'] == 248056)
slc = F.normalize(data['hidden_states'][img, 0].float(), dim=-1)
sim = slc @ slc.t(); n = sim.shape[0]
print(f"shape={tuple(data['hidden_states'].shape)} "
    f"image-pad pairwise cos={((sim-torch.eye(n)).sum()/(n*(n-1))).item():.4f} "
    f"(target ~0.58, broken=0.99)")