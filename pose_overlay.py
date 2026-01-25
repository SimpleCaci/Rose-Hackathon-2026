# pose_overlay.py
# Simple, readable OpenCV overlay for drawing pose joints + skeleton directly onto camera frames.

from __future__ import annotations
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union
import cv2
import numpy as np

# COCO / YOLO-Pose (17 keypoints) naming
JOINTS: List[str] = [
    "nose",
    "left_eye", "right_eye",
    "left_ear", "right_ear",
    "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow",
    "left_wrist", "right_wrist",
    "left_hip", "right_hip",
    "left_knee", "right_knee",
    "left_ankle", "right_ankle",
]

IDX: Dict[str, int] = {name: i for i, name in enumerate(JOINTS)}

# Pairs of joints to connect with lines
SKELETON: List[Tuple[str, str]] = [
    ("nose", "left_eye"), ("nose", "right_eye"),
    ("left_eye", "left_ear"), ("right_eye", "right_ear"),

    ("left_shoulder", "right_shoulder"),
    ("left_shoulder", "left_elbow"), ("left_elbow", "left_wrist"),
    ("right_shoulder", "right_elbow"), ("right_elbow", "right_wrist"),

    ("left_shoulder", "left_hip"), ("right_shoulder", "right_hip"),
    ("left_hip", "right_hip"),

    ("left_hip", "left_knee"), ("left_knee", "left_ankle"),
    ("right_hip", "right_knee"), ("right_knee", "right_ankle"),
]

# Input keypoint formats we accept:
# - numpy array/list shape (17,3): (x,y,conf)
# - numpy array/list shape (17,2): (x,y) (assumes conf=1.0)
# - dict: {"left_knee": (x,y,conf), ...}
Keypoints = Union[np.ndarray, Sequence[Sequence[float]], Dict[str, Tuple[float, ...]]]


def _normalize_keypoints(kps: Keypoints) -> np.ndarray:
    """
    Return a (17,3) array: [x, y, conf] in pixel coordinates.
    Missing/conf=0 joints will be ignored by the draw function.
    """
    if isinstance(kps, dict):
        arr = np.zeros((len(JOINTS), 3), dtype=np.float32)
        for name, val in kps.items():
            if name not in IDX:
                continue
            i = IDX[name]
            if len(val) == 2:
                x, y = val
                c = 1.0
            else:
                x, y, c = val[0], val[1], val[2]
            arr[i] = (float(x), float(y), float(c))
        return arr

    arr = np.asarray(kps, dtype=np.float32)

    if arr.shape == (len(JOINTS), 2):
        conf = np.ones((len(JOINTS), 1), dtype=np.float32)
        return np.concatenate([arr, conf], axis=1)

    if arr.shape == (len(JOINTS), 3):
        return arr

    raise ValueError(f"Expected (17,2), (17,3), or dict; got {arr.shape}.")


def draw_pose_on_frame(
    frame: np.ndarray,
    keypoints: Keypoints,
    *,
    min_conf: float = 0.3,
    draw_names: bool = False,
) -> np.ndarray:
    """
    Draw a pose (joints + skeleton) directly onto the given camera frame.

    - frame: the image from your camera (BGR)
    - keypoints: (17,3) or (17,2) in pixel coordinates (x,y[,conf])
    - min_conf: hide low-confidence joints/limbs
    """
    kps = _normalize_keypoints(keypoints)
    h, w = frame.shape[:2]

    def visible(i: int) -> bool:
        x, y, c = kps[i]
        return (c >= min_conf) and (0 <= x < w) and (0 <= y < h)

    def pt(i: int) -> Tuple[int, int]:
        x, y, _ = kps[i]
        return int(x), int(y)

    # 1) Draw limbs (lines) first
    for a, b in SKELETON:
        ia, ib = IDX[a], IDX[b]
        if visible(ia) and visible(ib):
            cv2.line(frame, pt(ia), pt(ib), (255, 255, 255), 2, lineType=cv2.LINE_AA)

    # 2) Draw joints (circles) on top
    for name in JOINTS:
        i = IDX[name]
        if visible(i):
            cv2.circle(frame, pt(i), 5, (0, 255, 0), -1, lineType=cv2.LINE_AA)
            if draw_names:
                x, y = pt(i)
                cv2.putText(
                    frame,
                    name,
                    (x + 8, y - 8),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    (255, 255, 255),
                    1,
                    lineType=cv2.LINE_AA,
                )

    return frame
