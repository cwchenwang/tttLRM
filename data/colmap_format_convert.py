#!/usr/bin/env python3
"""
COLMAP Format Converter

Converts COLMAP sparse reconstruction output to the opencv_cameras.json format
used by the tttLRM inference pipeline.

Expected COLMAP input structure:
    scene_dir/
    ├── sparse/0/  (or sparse/)
    │   ├── cameras.bin/txt
    │   ├── images.bin/txt
    │   └── points3D.bin/txt  (not needed)
    └── images/

Output: opencv_cameras.json in the same format as dl3dv_format_convert.py produces.

Usage:
    python data/colmap_format_convert.py \
        --source_dir /path/to/colmap_scene \
        --output_dir /path/to/output \
        --sparse_dir sparse/0 \
        --images_dir images
"""

import os
import json
import struct
import argparse
import collections
from pathlib import Path

import numpy as np
import cv2


# COLMAP camera model IDs
CAMERA_MODEL_IDS = {
    0: "SIMPLE_PINHOLE",
    1: "PINHOLE",
    2: "SIMPLE_RADIAL",
    3: "RADIAL",
    4: "OPENCV",
    5: "OPENCV_FISHEYE",
    6: "FULL_OPENCV",
    7: "FOV",
    8: "SIMPLE_RADIAL_FISHEYE",
    9: "RADIAL_FISHEYE",
    10: "THIN_PRISM_FISHEYE",
}

CameraModel = collections.namedtuple("CameraModel", ["model_id", "model_name", "num_params"])
CAMERA_MODELS = {
    0: CameraModel(0, "SIMPLE_PINHOLE", 3),
    1: CameraModel(1, "PINHOLE", 4),
    2: CameraModel(2, "SIMPLE_RADIAL", 4),
    3: CameraModel(3, "RADIAL", 5),
    4: CameraModel(4, "OPENCV", 8),
    5: CameraModel(5, "OPENCV_FISHEYE", 8),
    6: CameraModel(6, "FULL_OPENCV", 12),
    7: CameraModel(7, "FOV", 5),
    8: CameraModel(8, "SIMPLE_RADIAL_FISHEYE", 4),
    9: CameraModel(9, "RADIAL_FISHEYE", 5),
    10: CameraModel(10, "THIN_PRISM_FISHEYE", 12),
}

Camera = collections.namedtuple("Camera", ["id", "model", "width", "height", "params"])
Image = collections.namedtuple("Image", ["id", "qvec", "tvec", "camera_id", "name"])


# --- Binary readers ---

def read_cameras_binary(path):
    cameras = {}
    with open(path, "rb") as f:
        num_cameras = struct.unpack("<Q", f.read(8))[0]
        for _ in range(num_cameras):
            camera_id = struct.unpack("<i", f.read(4))[0]
            model_id = struct.unpack("<i", f.read(4))[0]
            width = struct.unpack("<Q", f.read(8))[0]
            height = struct.unpack("<Q", f.read(8))[0]
            num_params = CAMERA_MODELS[model_id].num_params
            params = struct.unpack(f"<{num_params}d", f.read(8 * num_params))
            cameras[camera_id] = Camera(
                id=camera_id, model=CAMERA_MODEL_IDS[model_id],
                width=width, height=height, params=np.array(params)
            )
    return cameras


def read_images_binary(path):
    images = {}
    with open(path, "rb") as f:
        num_images = struct.unpack("<Q", f.read(8))[0]
        for _ in range(num_images):
            image_id = struct.unpack("<i", f.read(4))[0]
            qvec = np.array(struct.unpack("<4d", f.read(32)))
            tvec = np.array(struct.unpack("<3d", f.read(24)))
            camera_id = struct.unpack("<i", f.read(4))[0]
            name = b""
            while True:
                ch = f.read(1)
                if ch == b"\x00":
                    break
                name += ch
            name = name.decode("utf-8")
            # Skip 2D points (num_points2D * (x, y, point3D_id))
            num_points2D = struct.unpack("<Q", f.read(8))[0]
            f.read(num_points2D * (8 + 8 + 8))  # x(double) + y(double) + id(int64)
            images[image_id] = Image(
                id=image_id, qvec=qvec, tvec=tvec,
                camera_id=camera_id, name=name
            )
    return images


# --- Text readers ---

def read_cameras_text(path):
    cameras = {}
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            camera_id = int(parts[0])
            model_name = parts[1]
            width = int(parts[2])
            height = int(parts[3])
            params = np.array([float(p) for p in parts[4:]])
            cameras[camera_id] = Camera(
                id=camera_id, model=model_name,
                width=width, height=height, params=params
            )
    return cameras


def read_images_text(path):
    images = {}
    with open(path, "r") as f:
        lines = [l.strip() for l in f if l.strip() and not l.strip().startswith("#")]
    # Images are stored as pairs of lines: metadata + 2D points
    for i in range(0, len(lines), 2):
        parts = lines[i].split()
        image_id = int(parts[0])
        qvec = np.array([float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])])
        tvec = np.array([float(parts[5]), float(parts[6]), float(parts[7])])
        camera_id = int(parts[8])
        name = parts[9]
        images[image_id] = Image(
            id=image_id, qvec=qvec, tvec=tvec,
            camera_id=camera_id, name=name
        )
    return images


# --- Helpers ---

def qvec2rotmat(qvec):
    """Convert COLMAP quaternion (w, x, y, z) to 3x3 rotation matrix."""
    w, x, y, z = qvec
    R = np.array([
        [1 - 2*y*y - 2*z*z, 2*x*y - 2*w*z, 2*x*z + 2*w*y],
        [2*x*y + 2*w*z, 1 - 2*x*x - 2*z*z, 2*y*z - 2*w*x],
        [2*x*z - 2*w*y, 2*y*z + 2*w*x, 1 - 2*x*x - 2*y*y],
    ])
    return R


def extract_intrinsics(camera):
    """
    Extract fx, fy, cx, cy and distortion coefficients from a COLMAP camera.
    Returns (fx, fy, cx, cy, distort) where distort is a numpy array or None.
    """
    model = camera.model
    params = camera.params

    if model == "SIMPLE_PINHOLE":
        f, cx, cy = params[0], params[1], params[2]
        return f, f, cx, cy, None
    elif model == "PINHOLE":
        fx, fy, cx, cy = params[0], params[1], params[2], params[3]
        return fx, fy, cx, cy, None
    elif model == "SIMPLE_RADIAL":
        f, cx, cy, k1 = params[0], params[1], params[2], params[3]
        return f, f, cx, cy, np.array([k1, 0.0, 0.0, 0.0])
    elif model == "RADIAL":
        f, cx, cy, k1, k2 = params[0], params[1], params[2], params[3], params[4]
        return f, f, cx, cy, np.array([k1, k2, 0.0, 0.0])
    elif model == "OPENCV":
        fx, fy, cx, cy = params[0], params[1], params[2], params[3]
        k1, k2, p1, p2 = params[4], params[5], params[6], params[7]
        return fx, fy, cx, cy, np.array([k1, k2, p1, p2])
    elif model == "FULL_OPENCV":
        fx, fy, cx, cy = params[0], params[1], params[2], params[3]
        k1, k2, p1, p2 = params[4], params[5], params[6], params[7]
        return fx, fy, cx, cy, np.array([k1, k2, p1, p2])
    else:
        raise ValueError(f"Unsupported COLMAP camera model: {model}")


def find_sparse_dir(source_dir):
    """Auto-detect the sparse reconstruction directory."""
    source = Path(source_dir)
    candidates = [
        source / "sparse" / "0",
        source / "sparse",
        source / "colmap" / "sparse" / "0",
        source / "colmap" / "sparse",
    ]
    for c in candidates:
        if (c / "cameras.bin").exists() or (c / "cameras.txt").exists():
            return c
    return None


def find_images_dir(source_dir):
    """Auto-detect the images directory."""
    source = Path(source_dir)
    candidates = [
        source / "images",
        source / "input",
    ]
    for c in candidates:
        if c.exists() and c.is_dir():
            return c
    return None


def read_colmap_model(sparse_dir):
    """Read cameras and images from COLMAP sparse dir (binary preferred, text fallback)."""
    sparse_dir = Path(sparse_dir)

    if (sparse_dir / "cameras.bin").exists():
        cameras = read_cameras_binary(sparse_dir / "cameras.bin")
        images = read_images_binary(sparse_dir / "images.bin")
    elif (sparse_dir / "cameras.txt").exists():
        cameras = read_cameras_text(sparse_dir / "cameras.txt")
        images = read_images_text(sparse_dir / "images.txt")
    else:
        raise FileNotFoundError(f"No cameras.bin or cameras.txt found in {sparse_dir}")

    return cameras, images


def process_one_scene(source_dir, output_dir, sparse_dir=None, images_dir=None):
    """
    Process one COLMAP scene into opencv_cameras.json format.

    Args:
        source_dir: Path to the COLMAP scene root
        output_dir: Where to write opencv_cameras.json and undistorted images
        sparse_dir: Relative path to sparse reconstruction (auto-detect if None)
        images_dir: Relative path to images directory (auto-detect if None)

    Returns:
        bool: True if successful
    """
    source_dir = Path(source_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Processing: {source_dir}")
    scene_name = source_dir.name

    # Resolve sparse dir
    if sparse_dir is not None:
        sparse_path = source_dir / sparse_dir
    else:
        sparse_path = find_sparse_dir(source_dir)
    if sparse_path is None or not sparse_path.exists():
        print(f"Error: Could not find sparse reconstruction in {source_dir}")
        return False

    # Resolve images dir
    if images_dir is not None:
        images_path = source_dir / images_dir
    else:
        images_path = find_images_dir(source_dir)
    if images_path is None or not images_path.exists():
        print(f"Error: Could not find images directory in {source_dir}")
        return False

    # Read COLMAP model
    cameras, images = read_colmap_model(sparse_path)
    print(f"Found {len(cameras)} camera(s), {len(images)} image(s)")

    # Sort images by name for deterministic output
    sorted_images = sorted(images.values(), key=lambda img: img.name)

    # Check for distortion in any camera
    has_distortion = False
    for cam in cameras.values():
        _, _, _, _, distort = extract_intrinsics(cam)
        if distort is not None and np.any(np.abs(distort) > 1e-8):
            has_distortion = True
            break

    # Create undistort dir if needed
    if has_distortion:
        undistort_dir = output_dir / "images_undistort"
        undistort_dir.mkdir(parents=True, exist_ok=True)

    new_data = []
    for img in sorted_images:
        cam = cameras[img.camera_id]
        fx, fy, cx, cy, distort = extract_intrinsics(cam)

        # Build w2c from COLMAP quaternion + translation
        R = qvec2rotmat(img.qvec)
        t = img.tvec
        w2c = np.eye(4)
        w2c[:3, :3] = R
        w2c[:3, 3] = t

        # Find the source image
        image_file = images_path / img.name
        if not image_file.exists():
            print(f"Warning: Image not found: {image_file}")
            continue

        image = cv2.imread(str(image_file), cv2.IMREAD_COLOR)
        if image is None:
            print(f"Warning: Failed to load image: {image_file}")
            continue

        h, w_img, _ = image.shape

        # Skip vertical images
        if h > w_img:
            print(f"Skipping vertical image: {img.name} ({h}x{w_img})")
            continue

        # Scale intrinsics if COLMAP resolution differs from actual image
        scale_x = w_img / cam.width
        scale_y = h / cam.height
        fx_scaled = fx * scale_x
        fy_scaled = fy * scale_y
        cx_scaled = cx * scale_x
        cy_scaled = cy * scale_y

        # Undistort if needed
        needs_undistort = distort is not None and np.any(np.abs(distort) > 1e-8)
        if needs_undistort:
            intr = np.array([[fx_scaled, 0, cx_scaled], [0, fy_scaled, cy_scaled], [0, 0, 1]])
            new_intr, roi = cv2.getOptimalNewCameraMatrix(intr, distort, (w_img, h), 0, (w_img, h))
            image = cv2.undistort(image, intr, distort, None, new_intr)
            fx_scaled = new_intr[0, 0]
            fy_scaled = new_intr[1, 1]
            cx_scaled = new_intr[0, 2]
            cy_scaled = new_intr[1, 2]

            output_image_path = undistort_dir / img.name
            output_image_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(output_image_path), image)
            file_path = "images_undistort/" + img.name
        else:
            # Copy or symlink original image
            dest_images_dir = output_dir / "images"
            dest_images_dir.mkdir(parents=True, exist_ok=True)
            dest_path = dest_images_dir / img.name
            if not dest_path.exists():
                dest_path.parent.mkdir(parents=True, exist_ok=True)
                os.symlink(os.path.abspath(image_file), str(dest_path))
            file_path = "images/" + img.name

        h_out, w_out = image.shape[:2]

        frame = {
            "w": w_out,
            "h": h_out,
            "fx": float(fx_scaled),
            "fy": float(fy_scaled),
            "cx": float(cx_scaled),
            "cy": float(cy_scaled),
            "w2c": w2c.tolist(),
            "file_path": file_path,
        }
        new_data.append(frame)

    if not new_data:
        print(f"Error: No valid images processed for {source_dir}")
        return False

    data = {
        "scene_name": scene_name,
        "frames": new_data,
    }

    output_json = output_dir / "opencv_cameras.json"
    with open(output_json, "w") as f:
        json.dump(data, f, indent=4)

    print(f"Successfully processed {source_dir}: {len(new_data)} frames -> {output_json}")
    return True


def create_data_path_json(output_dir, output_file):
    """
    Create colmap_data_path.json listing all opencv_cameras.json paths.

    Args:
        output_dir: Directory containing processed scene folders
        output_file: Path to the output JSON file
    """
    output_dir = Path(output_dir)
    data_paths = []

    if (output_dir / "opencv_cameras.json").exists():
        # Single scene case
        data_paths.append(str(output_dir / "opencv_cameras.json"))
    else:
        # Multiple scenes
        for item in sorted(output_dir.iterdir()):
            if item.is_dir() and not item.name.startswith("."):
                cameras_file = item / "opencv_cameras.json"
                if cameras_file.exists():
                    data_paths.append(str(cameras_file))

    with open(output_file, "w") as f:
        json.dump(data_paths, f, indent=2)

    print(f"Created {output_file} with {len(data_paths)} data path(s)")


def main():
    parser = argparse.ArgumentParser(description="Convert COLMAP sparse reconstruction to opencv_cameras.json format")
    parser.add_argument("--source_dir", type=str, required=True,
                        help="Path to a single COLMAP scene, or a directory of scenes")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for processed data")
    parser.add_argument("--sparse_dir", type=str, default=None,
                        help="Relative path to sparse reconstruction within source_dir (default: auto-detect)")
    parser.add_argument("--images_dir", type=str, default=None,
                        help="Relative path to images within source_dir (default: auto-detect)")
    parser.add_argument("--data_path_json", type=str, default=None,
                        help="Output path for the data path JSON (default: <output_dir>/../colmap_data_path.json)")
    args = parser.parse_args()

    source_dir = Path(args.source_dir)
    output_dir = Path(args.output_dir)

    if not source_dir.exists():
        print(f"Error: {source_dir} does not exist")
        return

    # Check if source_dir is a single scene or a directory of scenes
    sparse_path = find_sparse_dir(source_dir) if args.sparse_dir is None else source_dir / args.sparse_dir
    is_single_scene = sparse_path is not None and sparse_path.exists()

    if is_single_scene:
        print(f"Processing single COLMAP scene: {source_dir}")
        process_one_scene(source_dir, output_dir, args.sparse_dir, args.images_dir)
    else:
        print(f"Processing directory of scenes: {source_dir}")
        converted = 0
        for item in sorted(source_dir.iterdir()):
            if item.is_dir() and not item.name.startswith("."):
                scene_output = output_dir / item.name
                if process_one_scene(item, scene_output, args.sparse_dir, args.images_dir):
                    converted += 1
        print(f"Converted {converted} scene(s)")

    # Create data path JSON
    data_path_json = args.data_path_json
    if data_path_json is None:
        data_path_json = str(output_dir.parent / "colmap_data_path.json")
    create_data_path_json(output_dir, data_path_json)

    print("Done!")


if __name__ == "__main__":
    main()
