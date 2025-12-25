import copy
import os
import time
import math
import cv2

import lpips
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision
from easydict import EasyDict as edict
from einops.layers.torch import Rearrange
from einops import rearrange
from PIL import Image

from .loss import PerceptualLoss

from functools import partial

from utils.metrics import compute_psnr, compute_lpips, compute_ssim
from utils import camera_utils 
from .block import Block, _init_weights
from .lact_ttt import full_ttt_op, ar_ttt_op
from utils import sp_support

import matplotlib.pyplot as plt
from gsplat import rasterization
from .gaussian_renderer import GaussianRenderer

def compute_rays(fxfycxcy, c2w, h, w):
    """Transform target before computing loss
    Args:
        fxfycxcy (torch.tensor): [b, v, 4]
        c2w (torch.tensor): [b, v, 4, 4]
    Returns:
        ray_o: (b, v, 3, h, w)
        ray_d: (b, v, 3, h, w)
    """
    b, v = fxfycxcy.size(0), fxfycxcy.size(1)

    # Efficient meshgrid equivalent using broadcasting
    idx_x = torch.arange(w, device=c2w.device)[None, :].expand(h, -1)  # [h, w]
    idx_y = torch.arange(h, device=c2w.device)[:, None].expand(-1, w)  # [h, w]

    # Reshape for batched matrix multiplication
    idx_x = idx_x.flatten().expand(b * v, -1)           # [b*v, h*w]
    idx_y = idx_y.flatten().expand(b * v, -1)           # [b*v, h*w]

    fxfycxcy = fxfycxcy.reshape(b * v, 4)               # [b*v, 4]
    c2w = c2w.reshape(b * v, 4, 4)                      # [b*v, 4, 4]

    x = (idx_x + 0.5 - fxfycxcy[:, 2:3]) / fxfycxcy[:, 0:1]     # [b*v, h*w]
    y = (idx_y + 0.5 - fxfycxcy[:, 3:4]) / fxfycxcy[:, 1:2]     # [b*v, h*w]
    z = torch.ones_like(x)                                      # [b*v, h*w]

    ray_d = torch.stack([x, y, z], dim=1)                       # [b*v, 3, h*w]
    ray_d = torch.bmm(c2w[:, :3, :3], ray_d)                    # [b*v, 3, h*w]
    ray_d = ray_d / torch.norm(ray_d, dim=1, keepdim=True)      # [b*v, 3, h*w]

    ray_o = c2w[:, :3, 3:4].expand(b * v, -1, h*w)              # [b*v, 3, h*w]

    ray_o = ray_o.reshape(b, v, 3, h, w)                        # [b, v, 3, h, w]
    ray_d = ray_d.reshape(b, v, 3, h, w)                        # [b, v, 3, h, w]

    return ray_o, ray_d

gs_renderer = GaussianRenderer.apply

class Renderer(nn.Module):
    def __init__(self, config, sh_degree=None):
        super().__init__()
        self.config = config

    @torch.amp.custom_fwd(cast_inputs=torch.float32, device_type='cuda')
    def forward(self, xyz, features, scaling, rotation, opacity, C2W, fxfycxcy, W, H, sh_degree, near_plane, far_plane):
        renderings = gs_renderer(xyz, features, scaling, rotation, opacity, C2W, fxfycxcy, W, H, sh_degree, near_plane, far_plane).permute(0, 1, 4, 2, 3)
        depth = renderings[:, :, 3:4]
        return edict(render=renderings[:, :, :3], depth=depth)

def prepare_input_target(data, config):
    input, target, virtual = {}, {}, {}
    if config.training.get("sample_ar", False) or config.training.get("sample_mixed_length", False):
        num_input_views = data['num_input_views'][0].item()
        num_virtual_views = num_input_views
        num_target_views = num_input_views
    else:
        num_input_views = config.training.num_input_views
        num_target_views = config.training.num_target_views
        num_virtual_views = config.training.num_virtual_views
    
    target_idx = None
    for key, value in data.items():
        if key.startswith('virtual'):
            continue
        elif key in ['apos', 'scene_scale', 'num_input_views', 'scene_name']:
            input[key] = value
            continue
        # The first num_input_views views are always used as input
        input[key] = value[:, :num_input_views]
        if target_idx is None:
            bsz, num_views = value.shape[:2]
            if config.training.target_has_input:
                target_idx = np.array([random.sample(range(num_views), num_target_views) for _ in range(bsz)])
            else: # For inference, the target views are the remaining views
                target_idx = np.array([np.arange(num_input_views, data['image'][i].size(0)).tolist() for i in range(bsz)])
            target_idx = torch.from_numpy(target_idx).long().to(value.device)
        value_index = target_idx.reshape(target_idx.size(0), target_idx.size(1), *[1] * (value.dim() - 2)) if value.dim() > 2 else target_idx
        target[key] = torch.gather(value, dim=1, index=value_index.expand(-1, -1, *value.size()[2:]))
    virtual = {k[len("virtual_"):]: data[k] for k in data.keys() if k.startswith("virtual_")}

    return edict(input), edict(target), edict(virtual)

class tttLRM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        if self.config.training.depth_loss_weight > 0.0:
            if self.config.model.get('use_moge', False):
                from moge.model.v2 import MoGeModel
                self.depth_anything = MoGeModel.from_pretrained("Ruicheng/moge-2-vitl")
            else:
                from depth_anything_v2.dpt import DepthAnythingV2
                '''
                self.depth_anything = DepthAnything.from_pretrained('LiheYoung/depth_anything_{:}14'.format('vits')).eval()
                '''
                model_configs = {
                    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
                    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
                    'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]}
                }
                encoder = 'vits' # or 'vitb', 'vits'
                self.depth_anything = DepthAnythingV2(**model_configs[encoder])
                self.depth_anything.load_state_dict(torch.load(f'depth_anything_v2/depth_anything_{encoder}.pth'))

            for param in self.depth_anything.parameters():
                param.requires_grad = False
        
        if self.config.training.perceptual_loss_weight > 0.0:
            self.perceptual_loss_module = PerceptualLoss()
            self.perceptual_loss_module.eval()
            # freeze the perceptual loss module
            for param in self.perceptual_loss_module.parameters():
                param.requires_grad = False

        self.dim = config.model.transformer.d
        self.num_input_views = config.training.num_input_views 
        self.ttt_scan = config.model.ttt_scan
        self.num_layers = config.model.transformer.n_layer
        self.dim = config.model.transformer.d
        self.patch_size = config.model.patch_size

        self.blocks = nn.ModuleList()
        for _ in range(self.num_layers):
            self.blocks.append(Block(dim=self.dim, bias=False, block_config=config.model.block_config))
        self.blocks.apply(_init_weights)

        self.tokenizer = nn.Sequential(
            Rearrange('b v c (hh ph) (ww pw) -> (b v) (hh ww) (ph pw c)', ph=self.patch_size, pw=self.patch_size),
            nn.Linear(config.model.in_channels * self.patch_size**2, self.dim, bias=False)
        )
        self.tokenizer.apply(_init_weights)
        self.input_layernorm = nn.LayerNorm(self.dim, bias=False)
        self.decoder = nn.Sequential(
            nn.LayerNorm(self.dim, bias=False),
            nn.Linear(self.dim, (3 + (config.model.gaussians.sh_degree + 1) ** 2 * 3 + 3 + 4 + 1) * self.patch_size ** 2, bias=False)
        )
        self.decoder.apply(_init_weights)
        self.renderer = Renderer(config)

    def train(self, mode=True):
        # override the train method to keep the fronzon modules in eval mode
        super().train(mode)
        if self.config.training.perceptual_loss_weight > 0.0:
            self.perceptual_loss_module.eval()
        if self.config.training.depth_loss_weight > 0.0:
            self.depth_anything.eval()

    def get_overview(self):
        count_train_params = lambda model: sum(
            p.numel() for p in model.parameters() if p.requires_grad
        )

        overview = edict(
            tokenizer=count_train_params(self.tokenizer),
            blocks=count_train_params(self.blocks)
            + count_train_params(self.input_layernorm),
            image_token_decoder=count_train_params(self.decoder),
        )
        return overview

    def compute_loss(self, rendering, target, xyz, c2w, opacity, disp_rel=None):
        B, V, _, H, W = rendering.size()
        rendering = rendering.reshape(B * V, -1, H, W)
        target = target.reshape(B * V, -1, H, W)

        l2_loss = F.mse_loss(rendering, target)
        psnr = -10.0 * torch.log10(l2_loss)
        psnr_values = compute_psnr(target, rendering)
        psnr_real = psnr_values.reshape(B, V).mean(dim=0).mean()

        perceptual_loss = torch.tensor(0.0).to(l2_loss.device)
        if self.config.training.get('perceptual_loss_weight', 0.0) > 0.0:
            perceptual_loss = self.perceptual_loss_module(rendering, target)

        opacity_loss = torch.tensor(0.0).to(l2_loss.device)
        if self.config.training.get('opacity_loss_weight', 0.0) > 0.0:
            opacity_loss = opacity.sigmoid().mean()

        depth_loss = torch.tensor(0.0).to(l2_loss.device)
        if self.config.training.get('depth_loss_weight', 0.0) > 0.0:
            xyz = xyz.flatten(3, 4)
            w2c = c2w.inverse()

            xyz_cam = torch.matmul(w2c[:, :, :3, :3], xyz) + w2c[:, :, :3, 3:4]
            depth = xyz_cam[:, :, 2:3, :]
            disp_rel = disp_rel.flatten(3,4)
            disp = 1.0 / depth.clamp(min=1e-2)
            disp_median = torch.median(disp, dim=-1, keepdim=True)[0]
            disp_var = (disp - disp_median).abs().mean(dim=-1, keepdim=True)
            disp_rel_median = torch.median(disp_rel, dim=-1, keepdim=True)[0]
            disp_rel_var = (disp_rel - disp_rel_median).abs().mean(dim=-1, keepdim=True)
            disp = (disp - disp_median) / (disp_var + 1e-6)
            disp_rel = (disp_rel - disp_rel_median) / (disp_rel_var + 1e-6)
            depth_loss = F.smooth_l1_loss(disp, disp_rel)

        loss = (
            self.config.training.l2_loss_weight * l2_loss
            + self.config.training.perceptual_loss_weight * perceptual_loss
            + self.config.training.opacity_loss_weight * opacity_loss
            + self.config.training.depth_loss_weight * depth_loss
        )

        loss_dict = {
            'loss': loss,
            'l2_loss': l2_loss,
            'psnr': psnr,
            'psnr_real': psnr_real,
            'perceptual_loss': perceptual_loss,
            'opacity_loss': opacity_loss,
            'depth_loss': depth_loss,
        }
        return edict(loss_dict)

    def forward(self, data_batch, create_visual=False, iter=0):
        """
        image (torch.tensor): [b, v, c, h, w]
        fxfycxcy (torch.tensor): [b, v, 4]
        c2w (torch.tensor): [b, v, 4, 4]
        """
        input, target, virtual = prepare_input_target(data_batch, self.config)
        num_input_views = input.image.size(1)
        num_virtual_views = virtual.c2w.size(1) 
        sp_world_size = sp_support.get_sp_world_size()
        num_input_views = num_input_views // sp_world_size
        num_virtual_views = num_virtual_views // sp_world_size

        if self.config.training.depth_loss_weight > 0.0:
            input_image_local = sp_support.sp_input_broadcast_scatter(input.image)
            b, v, _, h, w = input_image_local.size()
            h_ = (h // 14) * 14
            w_ = (w // 14) * 14
            imgs_ = nn.functional.interpolate(input_image_local.reshape(b*v,_,h,w), (h_,w_), mode='bilinear') # bv x 3 x h x w
            with torch.no_grad():
                if self.config.model.get('use_moge', False):
                    disp_rel = self.depth_anything.infer(imgs_)['depth'].unsqueeze(1)
                    disp_rel = 1 / disp_rel
                else:
                    disp_rel = self.depth_anything(imgs_).unsqueeze(1) # bv x 1 x h x w
            disp_rel = nn.functional.interpolate(disp_rel, (h, w), mode='nearest').reshape(b,v,1,h,w).to(input.image.dtype) # bv x 1 x h x w
     
        B, V, C, H, W = input['image'].shape
        input['ray_o'], input['ray_d'] = compute_rays(input['fxfycxcy'], input['c2w'], H, W)
        target['ray_o'], target['ray_d'] = compute_rays(target['fxfycxcy'], target['c2w'], H, W)
        virtual['ray_o'], virtual['ray_d'] = compute_rays(virtual['fxfycxcy'], virtual['c2w'], H, W)
        if 'input_indices' in virtual:
            idx = virtual['input_indices'].long().view(B, -1, 1, 1, 1).expand(-1, -1, C, H, W)
            virtual['image'] = torch.gather(input['image'], dim=1, index=idx)
        
        input_c2w_local = sp_support.sp_input_broadcast_scatter(input['c2w'])
        posed_input = torch.cat([input['ray_o'], input['ray_d'], torch.cross(input['ray_o'], input['ray_d'], dim=2), input['image'] * 2.0 - 1.0], dim=2)
        posed_input = sp_support.sp_input_broadcast_scatter(posed_input, scatter_dim=1)
        for key in virtual:
            virtual[key] = sp_support.sp_input_broadcast_scatter(virtual[key].contiguous(), scatter_dim=1)
        for key in target:
            if not self.training:
                target[key], new_shape = sp_support.sp_input_broadcast_scatter(target[key].contiguous(), scatter_dim=1, different_size=True)
            else:
                target[key] = sp_support.sp_input_broadcast_scatter(target[key].contiguous(), scatter_dim=1)

        B, V, C, H, W = posed_input.size()
        posed_query = torch.cat([virtual['ray_o'], virtual['ray_d'], torch.cross(virtual['ray_o'], virtual['ray_d'], dim=2), virtual['image'] * 2.0 - 1.0], dim=2)
        model_input = torch.cat((posed_input, posed_query), dim=1)

        # Start Model Forward
        image_tokens = self.tokenizer(model_input)  # [b*v, n_patches, d]
        _, N_patches, D = image_tokens.shape
        image_tokens = image_tokens.reshape(-1, (num_input_views + num_virtual_views) * N_patches, D)
        image_tokens = self.input_layernorm(image_tokens)
          
        num_img_tokens = H * W // (self.patch_size**2)
        num_input_tokens = num_input_views * num_img_tokens
        num_target_tokens = num_virtual_views * num_img_tokens
        if self.ttt_scan == "ar":
            ttt_config = ar_ttt_op(
                update_minibatch=num_img_tokens * self.config.model.get("miniupdate_views", 4) // sp_world_size,
                length=num_input_tokens + num_target_tokens,
                update_length=num_input_tokens,
            )
        elif self.ttt_scan == "full":
            ttt_config = full_ttt_op(
                update_minibatch=num_input_tokens,
                apply_only_minibatch=0,
                length=num_input_tokens + num_target_tokens,
                update_length=num_input_tokens,
            )
        info = {
            "num_img_tokens": num_img_tokens,
            "num_input_tokens": num_input_tokens,
            "num_target_tokens": num_target_tokens,
            "ttt_config": ttt_config
        }

        for i in range(self.num_layers):
            image_tokens, _ = self.blocks[i](image_tokens, shape_info=info)
        
        input_tokens, query_tokens = image_tokens.split([num_input_views * N_patches, num_target_tokens], dim=1)     
        gaussians = self.decoder(query_tokens)
        gaussians = gaussians.reshape(B, -1, (3 + (self.config.model.gaussians.sh_degree + 1) ** 2 * 3 + 3 + 4 + 1))
        num_local_gaussians = gaussians.size(1)
     
        xyz, features, scaling, rotation, opacity = gaussians.split([3, (self.config.model.gaussians.sh_degree + 1) ** 2 * 3, 3, 4, 1], dim=2)
        features = features.reshape(features.size(0), features.size(1), (self.config.model.gaussians.sh_degree + 1) ** 2, 3).contiguous()
        scaling = (scaling - 2.3).clamp(max=self.config.model.gaussians.get('max_scaling', -1.2))
        opacity = opacity - 2.0

        # Align position to each ray
        xyz = rearrange(xyz, "b (v h w ph pw) c -> b v c (h ph) (w pw)", v=num_virtual_views, h=H // self.patch_size, w=W // self.patch_size, ph=self.patch_size, pw=self.patch_size)
        xyz = xyz.mean(dim=2, keepdim=True).sigmoid() * self.config.model.get('max_dist', 500)
        xyz = virtual['ray_o'] + xyz * virtual['ray_d']
        xyz_local = xyz.clone()
        xyz = rearrange(xyz, "b v c (h ph) (w pw) -> b (v h w ph pw) c", ph=self.patch_size, pw=self.patch_size)

        # Save local xyz and opacity for loss computation
        opacity_local = opacity.clone()  

        # Gather Gaussians for the same sequence for rendering
        xyz = sp_support.sp_all_gather(xyz, gather_dim=1, length=num_local_gaussians * sp_support.get_sp_world_size())
        features = sp_support.sp_all_gather(features, gather_dim=1, length=num_local_gaussians * sp_support.get_sp_world_size())
        scaling = sp_support.sp_all_gather(scaling, gather_dim=1, length=num_local_gaussians * sp_support.get_sp_world_size())
        rotation = sp_support.sp_all_gather(rotation, gather_dim=1, length=num_local_gaussians * sp_support.get_sp_world_size())
        opacity = sp_support.sp_all_gather(opacity, gather_dim=1, length=num_local_gaussians * sp_support.get_sp_world_size())
     
        # Gaussian Pruning
        threshold = None
        keep_idx = None
        gaussian_usage = (opacity.sigmoid() > self.config.get('usage_threshold', 0.001)).float().mean(dim=1).squeeze(-1) # (B,)
        prune_ratio = self.config.model.gaussians.get("prune_ratio", 0.0)
        if prune_ratio > 0.0:
            random_ratio = self.config.model.gaussians.get("random_ratio", 0.0)
            if data_batch['num_input_views'][0].item() >= 64: # assuming batch size is 1
                prune_ratio = self.config.model.gaussians.get("prune_ratio_64", 0.30)
                random_ratio = self.config.model.gaussians.get("random_ratio_64", 0.05)

            random_ratio = (1 - prune_ratio) * random_ratio
            keepnum = int(prune_ratio * xyz.size(1))
            sort_idx = opacity.argsort(dim=1, descending=True)
            keep_idx = sort_idx[:, :keepnum]
            rest_idx = sort_idx[:, keepnum:]
            if random_ratio > 0.0:
                randnum = int(random_ratio * xyz.size(1))
                rand_idx = torch.randperm(rest_idx.size(1))[:randnum]
                keep_idx = torch.cat((keep_idx, rest_idx[:, rand_idx]), dim=1)
            xyz = xyz.gather(1, keep_idx.expand(-1, -1, xyz.size(2)))
            features = features.gather(1, keep_idx.unsqueeze(-1).expand(-1, -1, features.size(2), features.size(3)))
            scaling = scaling.gather(1, keep_idx.expand(-1, -1, scaling.size(2)))
            rotation = rotation.gather(1, keep_idx.expand(-1, -1, rotation.size(2)))
            opacity = opacity.gather(1, keep_idx.expand(-1, -1, opacity.size(2)))

        # Render at target camera pose
        render = None
        if target is not None:
            render_start = time.time()
            render = self.renderer(xyz, features, scaling, rotation, opacity, target['c2w'], target['fxfycxcy'], W, H, self.config.model.gaussians.sh_degree, self.config.model.get('near_plane', 0.1), self.config.model.get('far_plane', 10000000000.0))
            if not self.training:
                render["render"] = sp_support.sp_all_gather(render["render"], gather_dim=1, length=new_shape[1])
                target = edict({key: sp_support.sp_all_gather(target[key], gather_dim=1, length=new_shape[1]) for key in target})
                virtual = edict({key: sp_support.sp_all_gather(virtual[key], gather_dim=1, length=num_virtual_views * sp_world_size) for key in virtual})

            # Compute Loss 
            loss_metrics = self.compute_loss(render["render"], target['image'], xyz_local, input_c2w_local, opacity_local, disp_rel=disp_rel if self.config.training.depth_loss_weight > 0.0 else None)      
     
        # for logging
        if loss_metrics is not None:
            loss_metrics.gaussian_usage = gaussian_usage

        result = edict(
            input=input,
            target=target,
            virtual=virtual,
            img_tokens=image_tokens,
            loss_metrics=loss_metrics,
            render=render,
            keep_idx=keep_idx,
            gaussians= {'xyz': xyz.float(), 'feature': features.float(), 'scale': scaling.float(), 'rotation': rotation.float(), 'opacity': opacity.float()}
        )
        if self.config.training.depth_loss_weight > 0.0:
            disp_rel = sp_support.sp_all_gather(disp_rel, gather_dim=1, length=num_input_views * sp_world_size)
            result.disp_rel = disp_rel
        return result
    
    def save_gaussian_ply(self, gaussian_dict, save_path, opacity_threshold=None):
        """
        Adapted from the original 3D GS implementation
        https://github.com/graphdeco-inria/gaussian-splatting/blob/main/scene/gaussian_model.py
        """
        from plyfile import PlyData, PlyElement
        xyz = gaussian_dict["xyz"].detach().cpu().float() # (N, 3)
        normal = torch.zeros_like(xyz) # (N, 3)
        N = xyz.shape[0]
        feature = gaussian_dict["feature"].detach().cpu().float() # (N, (sh_degree+1)**2, 3)
        f_dc = feature[:, 0].contiguous() # (N, 3)
        f_rest_full = torch.zeros(N, 3*(3+1)**2-3).float()
        if feature.shape[1] > 1:
            f_rest = feature[:, 1:].transpose(1, 2).reshape(N, -1) # (N, 3*(sh_degree+1)**2-3)
            f_rest_full[:, :f_rest.shape[1]] = f_rest
        f_rest_full = f_rest_full.contiguous()
        scale = gaussian_dict["scale"].detach().cpu().float() # (N, 3)
        opacity = gaussian_dict["opacity"].detach().cpu().float() # (N, 1)
        rotation = gaussian_dict["rotation"].detach().cpu().float() # (N, 4)
        attributes = np.concatenate([xyz.numpy(), 
                                     normal.numpy().astype(np.uint8),
                                     f_dc.numpy(),
                                     f_rest_full.numpy(),
                                     opacity.numpy(),
                                     scale.numpy(),
                                     rotation.numpy()
                                    ], axis=1)
        if opacity_threshold is not None:                             
            attributes = attributes[opacity.squeeze(-1).sigmoid().numpy() > opacity_threshold]
        attribute_list = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        attribute_list += ['f_dc_{}'.format(i) for i in range(f_dc.shape[1])]
        attribute_list += ['f_rest_{}'.format(i) for i in range(f_rest_full.shape[1])]
        attribute_list += ['opacity']
        attribute_list += ['scale_{}'.format(i) for i in range(scale.shape[1])]
        attribute_list += ['rot_{}'.format(i) for i in range(rotation.shape[1])]
        dtype_full = [(attribute, 'f4') for attribute in attribute_list]
        dtype_full[3:6] = [(attribute, 'u1') for attribute in attribute_list[3:6]]
        elements = np.empty(attributes.shape[0], dtype=dtype_full)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(save_path)
    
    def save_input_video(self, input_intr, input_c2ws, gaussian_dict, H, W, save_path, insert_frame_num = 8):
        """
        Interpolate input frames and save rendered video
        input_intr: (V, 4), (fx, fy, cx, cy)
        input_c2ws: (V, 4, 4)
        """
        import cv2
        import subprocess
        V = input_intr.shape[0]
        device = input_intr.device
        input_intr = input_intr.detach().cpu().float()
        input_c2ws = input_c2ws.detach().cpu().float()

        input_intr_mat = torch.zeros((V, 3, 3))
        input_intr_mat[:, 0, 0] = input_intr[:, 0]
        input_intr_mat[:, 1, 1] = input_intr[:, 1]
        input_intr_mat[:, 0, 2] = input_intr[:, 2]
        input_intr_mat[:, 1, 2] = input_intr[:, 3]
        input_c2ws = torch.cat([input_c2ws, input_c2ws[:1]], dim=0) # wrap around
        input_intr_mat = torch.cat([input_intr_mat, input_intr_mat[:1]], dim=0) # wrap around
        c2ws, intr_mat, _ = camera_utils.get_interpolated_poses_many(input_c2ws[:, :3, :4], input_intr_mat, steps_per_transition = insert_frame_num)
        V = c2ws.shape[0]
        c2ws_mat = torch.eye(4).unsqueeze(0).repeat(V, 1, 1)
        c2ws_mat[:, :3, :4] = c2ws
        intr_fxfycxcy = torch.zeros(V, 4)
        intr_fxfycxcy[:, 0] = intr_mat[:, 0, 0]
        intr_fxfycxcy[:, 1] = intr_mat[:, 1, 1]
        intr_fxfycxcy[:, 2] = intr_mat[:, 0, 2]
        intr_fxfycxcy[:, 3] = intr_mat[:, 1, 2]
        c2ws_mat = c2ws_mat.to(device)
        intr_fxfycxcy = intr_fxfycxcy.to(device)

        xyz = gaussian_dict["xyz"].detach().float().to(device) # (N, 3)
        feature = gaussian_dict["feature"].detach().float().to(device) # (N, (sh_degree+1)**2, 3)
        scale = gaussian_dict["scale"].detach().float().to(device) # (N, 3)
        rotation = gaussian_dict["rotation"].detach().float().to(device) # (N, 4)
        opacity = gaussian_dict["opacity"].detach().float().to(device) # (N, 1)

        renderings = []
        with torch.autocast(enabled=False, device_type="cuda"):
            for i in range(V):
                rendering = GaussianRenderer.render(xyz, feature, scale, rotation, opacity,
                                                    c2ws_mat[i], intr_fxfycxcy[i], W, H,
                                                    self.config.model.gaussians.sh_degree,
                                                    self.config.model.get("near_plane", 0.1),
                                                    self.config.model.get("far_plane", 10000000000.0))
                rendering = rendering.squeeze(0).clamp(0, 1).cpu().numpy() # (H, W, 3)
                rendering = (rendering * 255).astype(np.uint8)
                rendering = cv2.cvtColor(rendering, cv2.COLOR_RGB2BGR)
                renderings.append(rendering)
        tmp_save_path = save_path.replace(".mp4", "_tmp.mp4")
        video_writer = cv2.VideoWriter(tmp_save_path, cv2.VideoWriter_fourcc(*'mp4v'), 30, (W, H))
        for r in renderings:
            video_writer.write(r)
        video_writer.release()
        subprocess.run(f"ffmpeg -y -i {tmp_save_path} -vcodec libx264 -f mp4 {save_path} -loglevel quiet", shell=True) 
        os.remove(tmp_save_path)

    def save_evaluations(self, save_dir, output_dict, batch):
        import torchvision

        input_images = output_dict["input"]["image"] # (B, Vin, 3, H, W)
        target_images = output_dict["target"]["image"] # (B, V, 3, H, W)
        renderings = output_dict["render"]["render"] # (B, V, H, W, 3)
        gaussians = output_dict["gaussians"]
        gaussian_usage = output_dict["loss_metrics"]["gaussian_usage"] # (B,)
        B, V, _, H, W = target_images.shape

        for i in range(B):
            uid = output_dict['input']['index'][i, 0, -1].item()
            scene_name = batch["scene_name"][i]
            scene_dir = os.path.join(save_dir, "%06d_%s" % (uid, scene_name))
            os.makedirs(scene_dir, exist_ok=True)

            # evaluation metrics
            psnr = compute_psnr(renderings[i], target_images[i]) # (V,)
            ssim = compute_ssim(renderings[i], target_images[i]) # (V,)
            lpips = compute_lpips(renderings[i], target_images[i]) # (V,)
            with open(os.path.join(scene_dir, "metrics.csv"), "w") as f:
                f.write("index, psnr, ssim, lpips\n")
                for j in range(V):
                    f.write(f"{j}, {psnr[j].item()}, {ssim[j].item()}, {lpips[j].item()}\n")
                f.write(f"mean, {psnr.mean().item()}, {ssim.mean().item()}, {lpips.mean().item()}\n")
                f.write(f"gaussian_usage, {gaussian_usage[i].item()}\n")
                f.close()
            print(f'uid: {uid}, psnr: {psnr.mean().item()}, ssim: {ssim.mean().item()}, lpips: {lpips.mean().item()}')

            # save images
            input_images_path = os.path.join(scene_dir, "input_images.png")
            input_image = input_images[i].permute(1, 2, 0, 3).flatten(2, 3) # (3, H, Vin*W)
            torchvision.utils.save_image(input_image, input_images_path)
            os.makedirs(os.path.join(scene_dir, "target"), exist_ok=True)
            os.makedirs(os.path.join(scene_dir, "rendering"), exist_ok=True)
            for j in range(V):
                target_path = os.path.join(scene_dir, "target", f"{j}.png")
                rendering_path = os.path.join(scene_dir, "rendering", f"{j}.png")
                torchvision.utils.save_image(target_images[i, j], target_path)
                torchvision.utils.save_image(renderings[i, j], rendering_path)

            # save gaussian ply
            gaussian = {k: v[i] for k, v in gaussians.items()}
            opacity_threshold = self.config.model.gaussians.get("usage_threshold", 0.001)
            self.save_gaussian_ply(gaussian, os.path.join(scene_dir, f"gaussians_{str(opacity_threshold).split('.')[-1]}.ply"), opacity_threshold)

            # save input traj video
            input_intr = output_dict['input']["fxfycxcy"][i]
            input_c2ws = output_dict['input']["c2w"][i]
            self.save_input_video(input_intr, input_c2ws, gaussian, H, W, os.path.join(scene_dir, "input_traj.mp4"), insert_frame_num=8)