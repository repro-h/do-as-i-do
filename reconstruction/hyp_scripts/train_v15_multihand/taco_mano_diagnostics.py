"""CPU-only geometry diagnostics; no checkpoint, cache, or GUI state is changed."""

from pathlib import Path

import numpy as np


def camera_rigidity(extrinsics):
    matrices = np.asarray(extrinsics, dtype=np.float64)
    if matrices.ndim != 3 or matrices.shape[1:] != (4, 4) or not len(matrices):
        raise ValueError(f"Expected nonempty (T,4,4) extrinsics: {matrices.shape}")
    if not np.isfinite(matrices).all():
        raise ValueError("Nonfinite camera extrinsics")
    rotation = matrices[:, :3, :3]
    singular = np.linalg.svd(rotation, compute_uv=False)
    determinants = np.linalg.det(rotation)
    orthogonal_error = np.max(np.abs(rotation.transpose(0, 2, 1) @ rotation - np.eye(3)))
    bottom_error = np.max(np.abs(matrices[:, 3, :] - [0, 0, 0, 1]))
    return {
        "frames": len(matrices),
        "determinant_min": float(determinants.min()),
        "determinant_max": float(determinants.max()),
        "singular_value_min": float(singular.min()),
        "singular_value_max": float(singular.max()),
        "max_rotation_orthogonality_error": float(orthogonal_error),
        "max_homogeneous_row_error": float(bottom_error),
        "proper_rigid_within_1e_4": bool(
            orthogonal_error < 1e-4 and bottom_error < 1e-4
            and np.max(np.abs(determinants - 1)) < 1e-4
        ),
    }


def render_shape_controls(variants, faces, path, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    faces = np.asarray(faces, dtype=np.int64)
    all_vertices = np.concatenate([v for _, v in variants])
    center = (all_vertices.min(axis=0) + all_vertices.max(axis=0)) / 2
    radius = max(float(np.ptp(all_vertices, axis=0).max()) * 0.55, 0.01)
    fig = plt.figure(figsize=(12, 3.8 * len(variants)))
    try:
        for row, (name, vertices) in enumerate(variants):
            triangles = np.asarray(vertices)[faces]
            normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
            normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)
            light = np.asarray([0.3, -0.4, 0.85])
            light /= np.linalg.norm(light)
            shade = 0.5 + 0.5 * np.abs(normals @ light)
            colors = shade[:, None] * np.asarray([0.55, 0.61, 0.51])
            for column, (elev, azim, label) in enumerate(((90, -90, "XY"), (0, -90, "XZ"), (0, 0, "YZ"))):
                ax = fig.add_subplot(len(variants), 3, row * 3 + column + 1, projection="3d")
                ax.add_collection3d(Poly3DCollection(
                    triangles, facecolors=colors, edgecolors="#202820", linewidths=0.22,
                ))
                ax.set_xlim(center[0] - radius, center[0] + radius)
                ax.set_ylim(center[1] - radius, center[1] + radius)
                ax.set_zlim(center[2] - radius, center[2] + radius)
                ax.set_box_aspect((1, 1, 1))
                ax.set_proj_type("ortho")
                ax.view_init(elev=elev, azim=azim)
                ax.set_axis_off()
                ax.set_title(f"{name}\n{label}", fontsize=10)
        fig.suptitle(title + "\nGlobal rotation removed; wrist centered; identical scale; not new GT", fontsize=11)
        fig.subplots_adjust(top=1.0 - 1.2 / fig.get_figheight(), bottom=0.01, left=0.01, right=0.99, hspace=0.12, wspace=0.01)
        fig.savefig(path, dpi=140)
    finally:
        plt.close(fig)


def export_rgb_frames(video_path, frames, out):
    import cv2

    frames = set(frames)
    if not frames:
        return []
    video_path = Path(video_path)
    if not video_path.is_file():
        raise FileNotFoundError(video_path)
    cap = cv2.VideoCapture(str(video_path))
    written = []
    try:
        for index in range(max(frames) + 1):
            ok, bgr = cap.read()
            if not ok:
                raise RuntimeError(f"Video ended at frame {index}; requested {sorted(frames)}")
            if index in frames:
                path = Path(out) / f"rgb_frame_{index:06d}.jpg"
                if not cv2.imwrite(str(path), bgr):
                    raise OSError(f"Cannot write {path}")
                written.append(str(path))
    finally:
        cap.release()
    return written
