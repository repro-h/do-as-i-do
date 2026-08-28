#!/usr/bin/env python3
"""Inspect one local TACO egocentric sequence without decoding its video."""

import argparse
import json
import pickle
from pathlib import Path

import cv2
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--taco-root", required=True)
    parser.add_argument("--triplet")
    parser.add_argument("--sequence")
    parser.add_argument("--decode-all", action="store_true")
    parser.add_argument("--out-json")
    return parser.parse_args()


def array_summary(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    array = np.asarray(value)
    result = {"shape": list(array.shape), "dtype": str(array.dtype)}
    if np.issubdtype(array.dtype, np.number):
        result["finite"] = bool(np.isfinite(array).all())
    return result


def pickle_summary(path):
    with path.open("rb") as handle:
        data = pickle.load(handle)
    result = {"type": type(data).__name__}
    if not isinstance(data, dict):
        result["value"] = array_summary(data)
        return result
    keys = sorted(data, key=str)
    result["count"] = len(keys)
    result["first_key"] = str(keys[0]) if keys else None
    result["last_key"] = str(keys[-1]) if keys else None
    if keys:
        sample = data[keys[0]]
        if isinstance(sample, dict):
            result["sample"] = {
                str(key): array_summary(value) for key, value in sample.items()
            }
        else:
            result["sample"] = array_summary(sample)
    return result


def choose_video(root, triplet, sequence):
    if bool(triplet) != bool(sequence):
        raise ValueError("Specify both --triplet and --sequence, or neither")
    if triplet:
        path = root / "Egocentric_RGB_Videos" / triplet / sequence / "color.mp4"
        if not path.is_file():
            raise FileNotFoundError(path)
        return path
    for path in sorted((root / "Egocentric_RGB_Videos").glob("*/*/color.mp4")):
        hand_root = root / "Hand_Poses" / path.parent.parent.name / path.parent.name
        camera_root = (
            root / "Egocentric_Camera_Parameters"
            / path.parent.parent.name / path.parent.name
        )
        required = [
            hand_root / "left_hand.pkl", hand_root / "right_hand.pkl",
            hand_root / "left_hand_shape.pkl", hand_root / "right_hand_shape.pkl",
            camera_root / "egocentric_intrinsic.txt",
            camera_root / "egocentric_frame_extrinsic.npy",
        ]
        if all(item.is_file() for item in required):
            return path
    raise RuntimeError("No complete egocentric TACO sequence found")


def video_summary(path, decode_all=False):
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open {path}")
    result = {
        "frames": int(round(capture.get(cv2.CAP_PROP_FRAME_COUNT))),
        "fps": float(capture.get(cv2.CAP_PROP_FPS)),
        "width": int(round(capture.get(cv2.CAP_PROP_FRAME_WIDTH))),
        "height": int(round(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))),
    }
    ok, frame = capture.read()
    if decode_all:
        decoded_frames = int(ok)
        while ok:
            ok, _ = capture.read()
            decoded_frames += int(ok)
        result["decoded_frames"] = decoded_frames
    capture.release()
    result["first_frame_readable"] = frame is not None
    return result


def main():
    args = parse_args()
    root = Path(args.taco_root).expanduser().resolve()
    video = choose_video(root, args.triplet, args.sequence)
    triplet, sequence = video.parent.parent.name, video.parent.name
    hand_root = root / "Hand_Poses" / triplet / sequence
    camera_root = root / "Egocentric_Camera_Parameters" / triplet / sequence
    intrinsics = np.loadtxt(camera_root / "egocentric_intrinsic.txt")
    extrinsics = np.load(
        camera_root / "egocentric_frame_extrinsic.npy", mmap_mode="r"
    )
    report = {
        "taco_root": str(root),
        "triplet": triplet,
        "sequence": sequence,
        "video_path": str(video),
        "video": video_summary(video, args.decode_all),
        "intrinsics": array_summary(intrinsics),
        "intrinsics_value": np.asarray(intrinsics).tolist(),
        "extrinsics": array_summary(extrinsics),
        "left_pose": pickle_summary(hand_root / "left_hand.pkl"),
        "right_pose": pickle_summary(hand_root / "right_hand.pkl"),
        "left_shape": pickle_summary(hand_root / "left_hand_shape.pkl"),
        "right_shape": pickle_summary(hand_root / "right_hand_shape.pkl"),
    }
    counts = {
        "video": report["video"].get(
            "decoded_frames", report["video"]["frames"]
        ),
        "extrinsics": int(extrinsics.shape[0]),
        "left_pose": int(report["left_pose"].get("count", 0)),
        "right_pose": int(report["right_pose"].get("count", 0)),
    }
    report["frame_counts"] = counts
    report["frame_counts_match"] = len(set(counts.values())) == 1
    text = json.dumps(report, indent=2)
    print(text)
    if args.out_json:
        output = Path(args.out_json).expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
