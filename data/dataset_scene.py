import random
import traceback
import json
import numpy as np
import pandas as pd
import torch
from easydict import EasyDict as edict
from torch.utils.data import Dataset
import os
from sklearn.cluster import KMeans
from PIL import Image
import torchvision.transforms as transforms

def normalize(x):
    return x / np.linalg.norm(x)

def mean_pose(c2ws: np.ndarray):
    center = c2ws[:, :3, 3].mean(0)
    vec2 = c2ws[:, :3, 2].sum(0)
    up = c2ws[:, :3, 1].sum(0)

    vec2 = normalize(vec2)
    vec0 = normalize(np.cross(up, vec2))
    vec1 = normalize(np.cross(vec2, vec0))
    m = np.stack([vec0, vec1, vec2, center], 1)

    avg_pos = np.zeros((4, 4))
    avg_pos[3, 3] = 1.0
    avg_pos[:3] = m
    return avg_pos

def normalize_with_mean_pose(
    in_c2ws,
    frame_method="mean_cam",
):
    if frame_method == "mean_cam":
        # bottom = np.reshape([0, 0, 0, 1.0], [1, 4])
        # apos = poses_avg(in_c2ws)
        # apos = np.concatenate([apos, bottom], -2)
        apos = mean_pose(in_c2ws)
        in_c2ws = (np.linalg.inv(apos) @ in_c2ws) # align coordinate system to the mean camera
    elif frame_method == "first_cam":
        in_c2ws = (np.linalg.inv(in_c2ws[0]) @ in_c2ws) # align coordinate system to the first camera
        apos = in_c2ws[0]
    else:
        raise NotImplementedError

    scene_scale = np.max(np.abs(in_c2ws[:, :3, 3]))
    in_c2ws[:, :3, 3] /= scene_scale
    return in_c2ws, np.linalg.inv(apos), 1.0 / scene_scale

def kmeans_input(c2ws: np.ndarray, num_input_views: int, input_indices: list = None):
    if input_indices is None:
        input_indices = list(range(len(c2ws)))
    cam_dirs = np.concatenate([c2ws[:, :3, 3], c2ws[:, :3, 2]], axis=1)
    cluster_centers = KMeans(n_clusters=num_input_views, random_state=0, n_init="auto").fit(cam_dirs).cluster_centers_ # (num_input, 6)
    input_frame_idx = []
    for center in cluster_centers:
        dists = np.linalg.norm(cam_dirs - center, axis=1)
        input_frame_idx.append(input_indices[np.argmin(dists)])
    return sorted(input_frame_idx)

def resize_and_crop(image, target_size, fxfycxcy):
    """
    Resize and center crop image to target_size, adjusting camera parameters accordingly.
    
    Args:
        image: PIL Image
        target_size: (height, width) tuple
        fxfycxcy: [fx, fy, cx, cy] list
    
    Returns:
        tuple: (resized_cropped_image, adjusted_fxfycxcy)
    """
    original_width, original_height = image.size  # PIL image is (width, height)
    target_height, target_width = target_size
    
    fx, fy, cx, cy = fxfycxcy
    
    # Calculate scale factor to fill target size (resize to cover)
    scale_x = target_width / original_width
    scale_y = target_height / original_height


    # Resize image
    new_width = int(round(original_width * scale_x))
    new_height = int(round(original_height * scale_y))
    resized_image = image.resize((new_width, new_height), Image.LANCZOS)

    # Calculate crop box for center crop
    left = (new_width - target_width) // 2
    top = (new_height - target_height) // 2
    right = left + target_width
    bottom = top + target_height

    # Crop image
    cropped_image = resized_image.crop((left, top, right, bottom))

    # Adjust camera parameters
    # Scale focal lengths and principal points
    new_fx = fx * scale_x
    new_fy = fy * scale_y
    new_cx = cx * scale_x - left
    new_cy = cy * scale_y - top

    return cropped_image, [new_fx, new_fy, new_cx, new_cy]

class Dataset(Dataset):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.ar_input_views_list = np.arange(self.config.get("sp_size", 4), self.config.training.num_input_views + 1, self.config.get("sp_size", 4)).tolist() if self.config.training.get("sample_ar", False) else None
        self.sample_ar = self.config.training.get("sample_ar", False)
        self.input_length_list = [16, 32, 64, 64] if self.config.training.get("sample_mixed_length", False) else None

        self.all_camera_paths = json.load(open(self.config.training.dataset_path, "r"))
        self.all_camera_paths = [x.strip() for x in self.all_camera_paths]
        self.all_camera_paths = [x for x in self.all_camera_paths if len(x) > 0]

    def __len__(self):
        return len(self.all_camera_paths)
    
    def select_views(self, data_json, cameras, num_views, num_input_views, view_selection_config):
        if view_selection_config.type == "random":
            return random.sample(range(len(cameras)), num_views)
        elif view_selection_config.type == "kmeans":
            json_key = f'fold_{view_selection_config.get("fold_size", 8)}_kmeans_{num_input_views}_input'
            fold_size = view_selection_config.get("fold_size", 8)
            all_frames = list(range(len(cameras)))
            target_frames = [i for i in all_frames if i % fold_size == 0]
            target_frames.sort()
            if json_key in data_json:
                input_frame_idx = data_json[json_key]
            else:
                rest_frames = [i for i in all_frames if i not in target_frames]
                rest_cams = [cameras[i] for i in rest_frames]
                rest_w2cs = np.stack([np.array(cam["w2c"]) for cam in rest_cams])
                rest_c2ws = np.linalg.inv(rest_w2cs)
                input_frame_idx = kmeans_input(rest_c2ws, num_input_views, input_indices=rest_frames)
            return input_frame_idx + target_frames
        elif view_selection_config.type == "two_frame":
            min_frame_dist = view_selection_config.min_frame_dist
            max_frame_dist = view_selection_config.max_frame_dist
            if len(cameras) <= min_frame_dist:
                return []
            frame_dist = random.randint(min_frame_dist, min(max_frame_dist, len(cameras) - 1))
            start_frame = random.randint(0, len(cameras) - frame_dist - 1)
            end_frame = start_frame + frame_dist
            rest_frames = random.sample(range(start_frame + 1, end_frame), num_views - 2)
            return [start_frame, end_frame] + rest_frames
        else:
            raise NotImplementedError


    def __getitem__(self, idx):
        if type(idx) == tuple:
            idx, feat_idx = idx
            num_input_views = self.ar_input_views_list[feat_idx] if self.config.training.get("sample_ar", False) else self.input_length_list[feat_idx]
            num_virtual_views = num_input_views
            num_views = self.config.training.num_views
        else:
            num_input_views = self.config.training.num_input_views
            num_virtual_views = self.config.training.num_virtual_views
            num_views = self.config.training.num_views

        try:
            camera_path = self.all_camera_paths[idx].strip()
            data_json = json.load(open(camera_path, "r"))
            scene_name = data_json.get("scene_name", "").replace("/", "_")
            frames = data_json['frames']
            image_base_dir = camera_path.rsplit('/', 1)[0]

            if self.config.get("evaluation", False) and self.config.get("kmeans_input", True) and f'fold_8_kmeans_{num_input_views}_input' in data_json.keys():
                image_choices = data_json[f'fold_8_kmeans_{num_input_views}_input']
                test_set = list(range(0, len(frames), 8))

                if self.config.model.get('ttt_scan', 'full') in ['ar']:
                    image_choices = np.array(sorted(image_choices)[:num_input_views]).reshape(-1, self.config.get('sp_size', 1)).flatten(order='F').tolist()
                image_choices += test_set 
            else:
                if len(frames) < num_views:
                    return self.__getitem__(random.randint(0, len(self) - 1))
                view_selector_config = self.config.training.get(
                    "view_selector", edict({"type": "random"})
                )
                image_choices = self.select_views(data_json, frames, num_views, num_input_views, view_selector_config)
                if len(image_choices) < num_views:
                    return self.__getitem__(random.randint(0, len(self) - 1))
                
                if self.config.training.get("sample_ar", False):
                    image_choices = sorted(image_choices)[:num_input_views * 2] # select the first a few ordered views
                    input_indices = np.array(sorted(random.sample(image_choices, num_input_views))).reshape(-1, self.config.get('sp_size', 1)).flatten(order='F').tolist() # consider the order due to sp
                    image_choices = input_indices + list(set(image_choices) - set(input_indices))
                elif self.config.model.get('ttt_scan', 'full') in ['ar']:
                    input_choices = image_choices[:num_input_views]
                    input_choices = np.array(sorted(input_choices)).reshape(-1, self.config.get('sp_size', 1)).flatten(order='F').tolist()
                    image_choices = input_choices + image_choices[num_input_views:]
                elif num_input_views < num_views // 4: # avoid sampling too much novel views
                    input_choices = sorted(image_choices)[::num_input_views]
                    image_choices = input_choices + random.sample(image_choices, num_input_views * 2)

            input_images, input_fxfycxcy, input_c2ws = [], [], []
            for index in image_choices:
                image_path = os.path.join(image_base_dir, frames[index]['file_path'])
                image = Image.open(image_path)
                fxfycxcy = [frames[index]['fx'], frames[index]['fy'], frames[index]['cx'], frames[index]['cy']]

                image, fxfycxcy = resize_and_crop(image, (self.config.model.image_size, self.config.model.image_size_x), fxfycxcy)

                # Convert RGBA to RGB if needed
                if image.mode == 'RGBA':
                    # Create a white background and paste the RGBA image on it
                    rgb_image = Image.new('RGB', image.size, (255, 255, 255))
                    rgb_image.paste(image, mask=image.split()[-1])  # Use alpha channel as mask
                    image = rgb_image
                elif image.mode != 'RGB':
                    # Convert any other mode to RGB
                    image = image.convert('RGB')

                input_images.append(transforms.ToTensor()(image))
                input_fxfycxcy.append(fxfycxcy)
                input_c2ws.append(np.linalg.inv(frames[index]['w2c']))
        except:
            traceback.print_exc()
            print(f"error loading data: {self.all_camera_paths[idx]}")
            return self.__getitem__(random.randint(0, len(self) - 1))
        
        input_c2ws, apos, scene_scale = normalize_with_mean_pose(np.array(input_c2ws), frame_method=self.config.training.frame_method)
        input_c2ws = torch.from_numpy(input_c2ws).float()
        input_fxfycxcy = torch.from_numpy(np.array(input_fxfycxcy)).float()
        image_choices = torch.from_numpy(np.array(image_choices)).long().unsqueeze(-1)
        scene_indices = torch.tensor(idx).long().unsqueeze(0).expand_as(image_choices)
        indices = torch.cat([image_choices, scene_indices], dim=-1)

        if num_virtual_views == num_input_views:
            virtual_c2ws = input_c2ws[:num_virtual_views]
            virtual_fxfycxcy = input_fxfycxcy[:num_virtual_views]
            virtual_input_indices = np.arange(num_input_views)
        else:
            virtual_input_indices = []
            cam_dirs = np.concatenate([input_c2ws[:num_input_views, :3, 3], input_c2ws[:num_input_views, :3, 2]], axis=1)
            cluster_centers = KMeans(n_clusters=num_virtual_views, random_state=0, n_init="auto").fit(cam_dirs).cluster_centers_ # (num_input, 6)
            virtual_c2ws, virtual_fxfycxcy = [], []
            for center in cluster_centers:
                dists = np.linalg.norm(cam_dirs - center, axis=1)
                virtual_c2ws.append(input_c2ws[np.argmin(dists)])
                virtual_fxfycxcy.append(input_fxfycxcy[np.argmin(dists)])
                virtual_input_indices.append(np.argmin(dists))
            virtual_c2ws = torch.stack(virtual_c2ws, dim=0)
            virtual_fxfycxcy = torch.stack(virtual_fxfycxcy, dim=0)
            virtual_input_indices = torch.from_numpy(np.array(virtual_input_indices)).long() if self.config.model.get('virtual_from_input', True) else None

        result = {
            "image": torch.stack(input_images),
            "c2w": input_c2ws,
            "fxfycxcy": input_fxfycxcy,
            "index": indices,
            "scene_name": scene_name,
            'virtual_c2w': virtual_c2ws,
            'virtual_fxfycxcy': virtual_fxfycxcy,
            'virtual_input_indices': virtual_input_indices,
            'num_input_views': num_input_views,
            'scene_scale': scene_scale,
            'apos': apos
        }
        return result

if __name__ == "__main__":
    # test dataset
    import omegaconf
    config = omegaconf.OmegaConf.load('/mnt/localssd/dl3dvconfig.yaml')
    config.training.frame_method = 'mean_cam'
    dataset = Dataset(config)
    for i in range(len(dataset)):
        print(dataset[i])
        import pdb; pdb.set_trace();