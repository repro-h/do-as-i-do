"""
Project meshes for all frames in a layout JSON, in-process (no subprocesses).

Scale and intrinsics are read from the layout JSON itself (no CLI args needed).
Uses new_quat + manual pose-to-camera frame conversion pipeline.

Usage:
python run_project_mesh_combined.py \
    --video /path/to/video.mp4 \
    --mesh /path/to/video_dir/video_segmentation/masks/frame_000010_masks/<object>/<object>.obj \
    --json /path/to/video_dir/obj_tracking_out/<object>/combined_visualization/layout.json \
    --output-base /path/to/video_dir/obj_tracking_out/<object>/combined_visualization/projected
"""

import argparse
import glob
import json
import os
import re
import subprocess
import sys

import cv2
import numpy as np
import torch
from scipy.spatial.transform import Rotation as R

from pytorch3d.renderer import (
    PerspectiveCameras,
    PointLights,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    SoftPhongShader,
    TexturesVertex,
)
from pytorch3d.io import load_objs_as_meshes
from pytorch3d.structures import Meshes


def parse_args():
    parser = argparse.ArgumentParser(
        description="Project meshes for all frames in a layout JSON (single process).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--video", required=True,
                        help="Path to input video (MP4 file) OR directory of "
                             "0-indexed PNG frames named NNNNNN.png.")
    parser.add_argument("--mesh", required=True, help="Path to object mesh (.obj)")
    parser.add_argument("--json", required=True, help="Path to layout JSON")
    parser.add_argument("--output-base", required=True, help="Output directory for frames")
    parser.add_argument("--hand-meshes", default=None, help="Path to hand meshes NPZ file")
    parser.add_argument("--hands", default="both", choices=["left", "right", "both"],
                        help="Which hand(s) to render")
    parser.add_argument("--layer-order", default="hand_front",
                        choices=["object_front", "hand_front"],
                        help="Which layer appears in front")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"],
                        help="Device to use for rendering")
    parser.add_argument("--object-color", default=None,
                        help="Solid RGB color (0-1) for the object as 'r,g,b' "
                             "(e.g. '0.6,0.9,0.6'). If set, overrides the mesh's "
                             "loaded textures.")
    parser.add_argument("--left-hand-color", default="0.8,0.5,0.5",
                        help="Solid RGB color (0-1) for the left hand as 'r,g,b'.")
    parser.add_argument("--right-hand-color", default="0.5,0.5,0.8",
                        help="Solid RGB color (0-1) for the right hand as 'r,g,b'.")
    parser.add_argument("--use-optimized-mesh-scale", action="store_true",
                        help="Use top-level translation_scale_optimization.mesh_scale "
                             "(optimized) instead of per-frame local_to_scene.scale "
                             "(== mesh_scale_original).")
    return parser.parse_args()


def _parse_rgb(s):
    parts = [float(x) for x in s.split(',')]
    if len(parts) != 3:
        raise ValueError(f"Expected 'r,g,b'; got {s!r}")
    return tuple(parts)


def cam_to_pytorch3d(verts_cam):
    """Convert vertices from camera frame (x-right, y-down, z-fwd) to PyTorch3D frame (x-left, y-up, z-fwd)."""
    verts_p3d = torch.zeros_like(verts_cam)
    verts_p3d[:, 0] = -verts_cam[:, 0]
    verts_p3d[:, 1] = -verts_cam[:, 1]
    verts_p3d[:, 2] = verts_cam[:, 2]
    return verts_p3d


def make_renderer(fx, fy, cx, cy, width, height, device,
                  max_faces_per_bin=200000):
    """Create a PyTorch3D renderer with given camera intrinsics."""
    focal_length = torch.tensor([[fx, fy]], dtype=torch.float32, device=device)
    principal_point = torch.tensor([[cx, cy]], dtype=torch.float32, device=device)

    cameras = PerspectiveCameras(
        focal_length=focal_length,
        principal_point=principal_point,
        image_size=((height, width),),
        in_ndc=False,
        device=device,
    )

    # bin_size=None lets pytorch3d auto-pick the coarse-to-fine binning, which
    # avoids the sub-pixel coverage gaps ("speckles") that bin_size=0 (naive
    # rasterization) produces on dense meshes. faces_per_pixel>1 + tiny blur
    # also fill gaps when small faces fall between pixel centers.
    # max_faces_per_bin is generous so dense reconstructed meshes (~36k faces
    # observed in the wild) don't silently truncate with "Bin size was too small
    # in the coarse rasterization phase" overflow warnings.
    raster_settings = RasterizationSettings(
        image_size=(height, width),
        blur_radius=1e-5,
        faces_per_pixel=8,
        bin_size=None,
        max_faces_per_bin=max_faces_per_bin,
    )

    lights = PointLights(device=device, location=[[0.0, 0.0, -3.0]])

    return MeshRenderer(
        rasterizer=MeshRasterizer(cameras=cameras, raster_settings=raster_settings),
        shader=SoftPhongShader(device=device, cameras=cameras, lights=lights),
    )


def transform_to_camera_frame(verts, rot_matrix, translation, scale, device,
                              camera_frame=False):
    """Apply scale + rotation + translation to vertices and return them in the
    camera frame (x-right, y-down, z-fwd). If camera_frame=True, the rotation
    and translation are already camera-frame and the axis swap is skipped."""
    verts_scaled = verts * scale
    R_torch = torch.tensor(rot_matrix, dtype=torch.float32, device=device)
    T_torch = torch.tensor(translation, dtype=torch.float32, device=device)
    verts_pose = verts_scaled @ R_torch.T + T_torch

    if camera_frame:
        return verts_pose
    # Pose frame (x-fwd, y-left, z-up) -> camera frame (x-right, y-down, z-fwd)
    verts_cam = torch.zeros_like(verts_pose)
    verts_cam[:, 0] = -verts_pose[:, 1]
    verts_cam[:, 1] = -verts_pose[:, 2]
    verts_cam[:, 2] = verts_pose[:, 0]
    return verts_cam


def render_mesh(mesh, rot_matrix, translation, scale, renderer, device,
                camera_frame=False):
    """Render mesh with given pose. Returns rgb (H,W,3) float [0,1] and alpha (H,W) float [0,1].

    If camera_frame=True, (rot_matrix, translation) are interpreted as already in
    the camera frame and the pose->camera axis swap is skipped.
    """
    verts_cam = transform_to_camera_frame(
        mesh.verts_packed(), rot_matrix, translation, scale, device, camera_frame,
    )
    verts_p3d = cam_to_pytorch3d(verts_cam)

    mesh_transformed = mesh.clone()
    mesh_transformed = mesh_transformed.update_padded(verts_p3d.unsqueeze(0))

    images = renderer(mesh_transformed)
    return images[0, ..., :3].cpu().numpy(), images[0, ..., 3].cpu().numpy()


def render_hand_mesh(vertices, faces, renderer, device, color=(0.8, 0.7, 0.6)):
    """Render a hand mesh. Vertices are already in camera frame."""
    verts_cam = torch.tensor(vertices, dtype=torch.float32, device=device)
    verts_p3d = cam_to_pytorch3d(verts_cam)

    faces_torch = torch.tensor(faces, dtype=torch.int64, device=device)
    textures = TexturesVertex(
        verts_features=torch.tensor([[color]], dtype=torch.float32, device=device).expand(-1, verts_p3d.shape[0], -1)
    )
    hand_mesh = Meshes(verts=[verts_p3d], faces=[faces_torch], textures=textures)

    images = renderer(hand_mesh)
    return images[0, ..., :3].cpu().numpy(), images[0, ..., 3].cpu().numpy()


def render_combined(meshes_data, renderer, device):
    """Render multiple meshes in one pass so the rasterizer's z-buffer handles
    inter-mesh occlusion correctly.

    meshes_data: list of (verts_cam, faces, color) tuples.
      verts_cam: (V, 3) torch.Tensor already in camera frame (x-right, y-down, z-fwd).
      faces:    (F, 3) torch.Tensor int64 (on any device; moved to `device`).
      color:    (r, g, b) float tuple in [0, 1].
    Returns (rgb, alpha) numpy arrays in [0, 1].
    """
    verts_p3d_list = []
    faces_list = []
    feats_list = []
    offset = 0
    for verts_cam, faces, color in meshes_data:
        verts_p3d = cam_to_pytorch3d(verts_cam)
        n = verts_p3d.shape[0]
        verts_p3d_list.append(verts_p3d)
        faces_list.append(faces.to(device=device, dtype=torch.int64) + offset)
        color_t = torch.tensor(color, dtype=torch.float32, device=device)
        feats_list.append(color_t.expand(n, 3))
        offset += n

    verts_cat = torch.cat(verts_p3d_list, dim=0)
    faces_cat = torch.cat(faces_list, dim=0)
    feats_cat = torch.cat(feats_list, dim=0).unsqueeze(0)  # (1, V, 3)
    combined = Meshes(
        verts=[verts_cat],
        faces=[faces_cat],
        textures=TexturesVertex(verts_features=feats_cat),
    )
    images = renderer(combined)
    return images[0, ..., :3].cpu().numpy(), images[0, ..., 3].cpu().numpy()


def read_video_frame(cap, frame_idx, last_read_idx):
    """Read a specific frame from the video capture, seeking only if necessary.
    Returns (frame_rgb, new_last_read_idx) or (None, last_read_idx) on failure."""
    if frame_idx != last_read_idx + 1:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame_bgr = cap.read()
    if not ret:
        return None, last_read_idx
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    return frame_rgb, frame_idx


class DirectoryVideoReader:
    """Drop-in replacement for cv2.VideoCapture when the 'video' is a directory
    of PNG/JPG frames named NNNNNN.png (0-indexed). Exposes read_frame(idx) and
    .width / .height, plus a no-op release() so calling code can stay uniform.
    """

    _EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp")

    def __init__(self, frames_dir):
        self.dir = frames_dir
        paths = {}
        for name in os.listdir(frames_dir):
            stem, ext = os.path.splitext(name)
            if ext.lower() not in self._EXTS:
                continue
            try:
                idx = int(stem)
            except ValueError:
                continue
            paths[idx] = os.path.join(frames_dir, name)
        if not paths:
            raise FileNotFoundError(
                f"No numerically-named frame images found in {frames_dir}")
        self.paths = paths
        first = cv2.imread(paths[min(paths)])
        if first is None:
            raise IOError(f"Could not read first frame in {frames_dir}")
        self.height, self.width = first.shape[:2]

    def read_frame(self, frame_idx):
        p = self.paths.get(frame_idx)
        if p is None:
            return None
        bgr = cv2.imread(p)
        if bgr is None:
            return None
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    def release(self):
        pass


def main():
    args = parse_args()

    # Device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Using device: {device}")

    # Load layout JSON
    with open(args.json) as f:
        data = json.load(f)

    objects = [obj for obj in data["objects"] if "frame_idx" in obj]
    if not objects:
        print("No objects with frame_idx found in layout JSON.")
        return
    print(f"Loaded layout.json with {len(objects)} objects to render")

    # Sort by frame_idx for sequential video reading
    objects.sort(key=lambda o: o["frame_idx"])

    # Pre-compute fallback intrinsics: first frame that has intrinsics_normalized
    fallback_intr = None
    for obj in objects:
        if "intrinsics_normalized" in obj:
            fallback_intr = obj["intrinsics_normalized"]
            print(f"Fallback intrinsics from frame {obj['frame_idx']}: "
                  f"fx_norm={fallback_intr['fx_norm']:.5f}, fy_norm={fallback_intr['fy_norm']:.5f}")
            break
    if fallback_intr is None:
        print("[error] No frame in layout JSON has intrinsics_normalized.")
        sys.exit(1)

    # Auto-detect camera-frame layout (output of camera-frame optimization).
    # In that mode the optimized pose lives in *_camera_frame fields and the
    # top-level translation_scale_optimization.mesh_scale supersedes per-frame scale.
    is_camera_frame_layout = data.get("frame") == "camera_frame"
    if is_camera_frame_layout:
        print("Detected camera-frame layout — using "
              "translation_camera_frame + quat_wxyz_camera_frame "
              "and skipping pose->camera axis swap.")

    # Optionally swap to the top-level optimized mesh_scale (uniform across frames)
    override_scale = None
    if args.use_optimized_mesh_scale or is_camera_frame_layout:
        try:
            override_scale = float(
                data["translation_scale_optimization"]["mesh_scale"]
            )
        except (KeyError, TypeError) as e:
            if args.use_optimized_mesh_scale:
                print(f"[error] --use-optimized-mesh-scale set but "
                      f"data['translation_scale_optimization']['mesh_scale'] not found: {e}")
                sys.exit(1)
            else:
                print(f"[warn] camera-frame layout but no mesh_scale found; "
                      f"falling back to per-frame local_to_scene.scale ({e})")
        if override_scale is not None:
            orig = data["translation_scale_optimization"].get("mesh_scale_original")
            print(f"Using optimized mesh_scale={override_scale:.6f} "
                  f"(was mesh_scale_original={orig})")

    # Open video source. Accepts either a directory of 0-indexed PNG frames
    # (e.g. $CLIP_DIR/all_frames) or an MP4 file.
    if not os.path.exists(args.video):
        print(f"[error] Video not found: {args.video}")
        sys.exit(1)
    if os.path.isdir(args.video):
        frame_source = DirectoryVideoReader(args.video)
        width, height = frame_source.width, frame_source.height
        cap = None
        print(f"Frame dir source: {args.video}  ({len(frame_source.paths)} frames, {width}x{height})")
    else:
        cap = cv2.VideoCapture(args.video)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_source = None
        print(f"MP4 source: {args.video}  ({width}x{height})")

    # Load mesh once
    if not os.path.exists(args.mesh):
        print(f"[error] Mesh not found: {args.mesh}")
        sys.exit(1)
    print("Loading mesh...")
    mesh = load_objs_as_meshes([args.mesh], device=device)

    if args.object_color is not None or mesh.textures is None:
        rgb = _parse_rgb(args.object_color) if args.object_color is not None else (0.75, 0.45, 0.25)
        verts_list = mesh.verts_list()
        faces_list = mesh.faces_list()
        n_verts = verts_list[0].shape[0]
        color_t = torch.tensor(rgb, dtype=torch.float32, device=device)
        verts_features = color_t.expand(n_verts, 3).unsqueeze(0)
        mesh = Meshes(
            verts=verts_list, faces=faces_list,
            textures=TexturesVertex(verts_features=verts_features),
        )
        reason = "requested" if args.object_color is not None else "mesh has no texture"
        print(f"Object mesh re-colored to solid RGB={rgb} ({reason})")

    # Dense SAM3D reconstructions can contain close to one million faces. The
    # coarse rasterizer silently drops faces when this cap is too small.
    renderer_max_faces_per_bin = max(200000, int(mesh.num_faces_per_mesh().max().item()))
    print(f"Renderer max_faces_per_bin={renderer_max_faces_per_bin}")

    # Load hand data once
    hand_data = None
    hands_to_render = []
    hand_colors = {
        "left": _parse_rgb(args.left_hand_color),
        "right": _parse_rgb(args.right_hand_color),
    }
    if args.hand_meshes:
        if not os.path.exists(args.hand_meshes):
            print(f"[error] Hand meshes not found: {args.hand_meshes}")
            sys.exit(1)
        hand_data = np.load(args.hand_meshes)
        hands_to_render = ["left", "right"] if args.hands == "both" else [args.hands]
        print(f"Loaded hand meshes, rendering: {hands_to_render}")

    os.makedirs(args.output_base, exist_ok=True)

    # Renderer cache: avoid recreating for identical intrinsics
    renderer_cache = {}

    def get_renderer(intr, W, H):
        fx = intr["fx_norm"] * W
        fy = intr["fy_norm"] * H
        cx = intr.get("cx_norm", 0.5) * W
        cy = intr.get("cy_norm", 0.5) * H
        key = (round(fx, 4), round(fy, 4), round(cx, 4), round(cy, 4), W, H)
        if key not in renderer_cache:
            renderer_cache[key] = make_renderer(
                fx, fy, cx, cy, W, H, device,
                max_faces_per_bin=renderer_max_faces_per_bin,
            )
        return renderer_cache[key]

    # Process all frames
    last_read_idx = -1
    for i, obj in enumerate(objects):
        frame_idx = obj["frame_idx"]
        lts = obj["local_to_scene"]

        scale = override_scale if override_scale is not None else lts["scale"][0]
        if is_camera_frame_layout:
            quat = lts["quat_wxyz_camera_frame"]
            translation = list(lts["translation_camera_frame"])  # already camera-frame
        else:
            quat = lts["new_quat"]
            tx, ty, tz = lts["translation"]
            translation = [tz, tx, ty]

        intr = obj.get("intrinsics_normalized", fallback_intr)
        renderer = get_renderer(intr, width, height)

        # Read video frame (dir or MP4)
        if frame_source is not None:
            frame_rgb = frame_source.read_frame(frame_idx)
        else:
            frame_rgb, last_read_idx = read_video_frame(cap, frame_idx, last_read_idx)
        if frame_rgb is None:
            print(f"  Skipping frame {frame_idx} (could not read from source)")
            continue

        # Convert quaternion to rotation matrix
        quat_xyzw = np.array([quat[1], quat[2], quat[3], quat[0]])  # wxyz -> xyzw
        rot_matrix = R.from_quat(quat_xyzw).as_matrix()

        # Joint render path: when the object has an explicit solid color and we
        # also have hand meshes for this frame, render everything in a single
        # rasterization pass so the rasterizer's z-buffer resolves occlusion
        # correctly. Falls back to the separate-render + 2D alpha composite below.
        use_joint = (
            args.object_color is not None
            and hand_data is not None
            and frame_idx < hand_data["right_vertices"].shape[0]
            and len(hands_to_render) > 0
        )

        if use_joint:
            obj_verts_cam = transform_to_camera_frame(
                mesh.verts_packed(), rot_matrix, translation, scale, device,
                camera_frame=is_camera_frame_layout,
            )
            meshes_data = [(
                obj_verts_cam,
                mesh.faces_packed(),
                _parse_rgb(args.object_color),
            )]
            for hand in hands_to_render:
                hand_verts = torch.tensor(
                    hand_data[f"{hand}_vertices"][frame_idx],
                    dtype=torch.float32, device=device,
                )
                hand_faces = torch.tensor(
                    hand_data[f"{hand}_faces"], dtype=torch.int64, device=device,
                )
                meshes_data.append((hand_verts, hand_faces, hand_colors[hand]))

            render_rgb, render_alpha = render_combined(meshes_data, renderer, device)
            render_rgb = (render_rgb * 255).clip(0, 255).astype(np.uint8)
            mask = render_alpha > 0.5
            result = frame_rgb.copy()
            result[mask] = render_rgb[mask]
        else:
            # Separate-render fallback (textured object or no hands this frame)
            render_rgb, render_alpha = render_mesh(
                mesh, rot_matrix, translation, scale, renderer, device,
                camera_frame=is_camera_frame_layout,
            )
            render_rgb = (render_rgb * 255).clip(0, 255).astype(np.uint8)
            mask = render_alpha > 0.5

            hand_rgb_list = []
            hand_mask_list = []
            if hand_data is not None:
                num_frames = hand_data["right_vertices"].shape[0]
                if frame_idx < num_frames:
                    for hand in hands_to_render:
                        verts = hand_data[f"{hand}_vertices"][frame_idx]
                        faces = hand_data[f"{hand}_faces"]
                        h_rgb, h_alpha = render_hand_mesh(verts, faces, renderer, device, color=hand_colors[hand])
                        h_rgb = (h_rgb * 255).clip(0, 255).astype(np.uint8)
                        hand_rgb_list.append(h_rgb)
                        hand_mask_list.append(h_alpha > 0.5)
                else:
                    print(f"  Warning: frame {frame_idx} out of range for hand meshes ({num_frames} frames)")

            result = frame_rgb.copy()
            if args.layer_order == "object_front":
                for h_rgb, h_mask in zip(hand_rgb_list, hand_mask_list):
                    result[h_mask] = h_rgb[h_mask]
                result[mask] = render_rgb[mask]
            else:
                result[mask] = render_rgb[mask]
                for h_rgb, h_mask in zip(hand_rgb_list, hand_mask_list):
                    result[h_mask] = h_rgb[h_mask]

        # Save frame
        output_path = os.path.join(args.output_base, f"frame_{frame_idx}.png")
        result_bgr = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
        cv2.imwrite(output_path, result_bgr)
        print(f"  [{i+1}/{len(objects)}] Saved frame {frame_idx}  scale={scale:.4f}")

    if cap is not None:
        cap.release()

    # Save frames as MP4 video
    files = glob.glob(os.path.join(args.output_base, "frame_*.png"))
    files.sort(key=lambda f: int(re.search(r"frame_(\d+)", f).group(1)))
    if files:
        video_path = os.path.join(args.output_base, "video.mp4")
        frame_pattern = os.path.join(args.output_base, "frame_%d.png")
        subprocess.run(
            [
                "ffmpeg", "-y", "-framerate", "30",
                "-i", frame_pattern,
                "-c:v", "libx264", "-pix_fmt", "yuv420p",
                video_path,
            ],
            check=True,
        )
        print(f"Saved video ({len(files)} frames) to {video_path}")


if __name__ == "__main__":
    main()
