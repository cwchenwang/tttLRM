from skimage.metrics import structural_similarity
from lpips import LPIPS
import torch
import einops

@torch.no_grad()
def compute_psnr(predict, target):
    """
    predict, target: (B, C, H, W) in range [0, 1]
    """
    target = target.clip(min=0, max=1)
    predict = predict.clip(min=0, max=1)
    mse = einops.reduce((target - predict) ** 2, "b c h w -> b", "mean")
    return -10 * mse.log10()

@torch.no_grad()
def compute_ssim(predict, target):
    """
    predict, target: (B, C, H, W) in range [0, 1]
    """
    predict = predict.clamp(0, 1)
    target = target.clamp(0, 1)
    ssim = [
        structural_similarity(
            predict[i].cpu().numpy(),
            target[i].cpu().numpy(),
            multichannel=True,
            channel_axis=0,
            data_range=1.0,
        ) for i in range(predict.size(0))
    ]
    ssim = torch.tensor(ssim, device=predict.device, dtype=predict.dtype)
    return ssim

@torch.no_grad()
def compute_lpips(predict, target):
    """
    predict, target: (B, C, H, W) in range [0, 1]
    """
    predict = predict.clamp(0, 1)
    target = target.clamp(0, 1)
    lpips_fn = LPIPS(net="vgg").to(predict.device)
    batch_size = 10
    values = []
    for i in range(0, predict.size(0), batch_size):
        value = lpips_fn.forward(
            predict[i : i + batch_size],
            target[i : i + batch_size],
            normalize=True,
        )
        values.append(value)
    value = torch.cat(values, dim=0)
    value = value[:, 0, 0, 0] # (B,)
    return value
        