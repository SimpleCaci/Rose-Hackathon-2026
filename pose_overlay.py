# pose_tasks.py
# MediaPipe Tasks Pose Landmarker (NOT mp.solutions.*) + OpenCV drawing utilities.
# Works with: mediapipe>=0.10.x (Tasks API), opencv-python(-headless), numpy

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple, List

import cv2
import numpy as np

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


@dataclass
class PoseTasksConfig:
    model_path: str = "pose_landmarker_full.task"  # download this file (see notes below)
    # 0=most accurate/slowest, 4=fastest/least accurate (varies by model)
    num_poses: int = 1
    min_pose_detection_confidence: float = 0.5
    min_pose_presence_confidence: float = 0.5
    min_tracking_confidence: float = 0.5


class PoseTasksLandmarker:
    """
    Wrapper around MediaPipe Tasks PoseLandmarker for video frames.
    Use `detect_bgr(frame_bgr)` -> returns a list of 33 landmarks (x,y,z,visibility) normalized [0..1].
    """

    def __init__(self, cfg: PoseTasksConfig):
        base_options = python.BaseOptions(model_asset_path=cfg.model_path)
        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            num_poses=cfg.num_poses,
            min_pose_detection_confidence=cfg.min_pose_detection_confidence,
            min_pose_presence_confidence=cfg.min_pose_presence_confidence,
            min_tracking_confidence=cfg.min_tracking_confidence,
        )
        self._landmarker = vision.PoseLandmarker.create_from_options(options)
        self._timestamp_ms = 0

    def detect_bgr(self, frame_bgr: np.ndarray) -> Optional[List[mp.framework.formats.landmark_pb2.NormalizedLandmark]]:
        """
        Returns landmarks for the first detected pose (list length 33), or None if no pose found.
        """
        # MediaPipe Tasks expects RGB
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        # Timestamp must increase for VIDEO mode
        self._timestamp_ms += 33  # ~30fps
        result = self._landmarker.detect_for_video(mp_image, self._timestamp_ms)

        if not result.pose_landmarks:
            return None
        return result.pose_landmarks[0]  # first person


# ---- Drawing ----

# Use MediaPipe's built-in pose connections (33-landmark skeleton)
POSE_CONNECTIONS = mp.solutions.pose.POSE_CONNECTIONS  # just a constant; doesn't require mp.solutions Pose runtime

def draw_pose_landmarks_on_frame(
    frame_bgr: np.ndarray,
    landmarks: List[mp.framework.formats.landmark_pb2.NormalizedLandmark],
    *,
    min_visibility: float = 0.5,
) -> np.ndarray:
    """
    Draw 33 landmarks + skeleton onto the camera frame (in-place, returns same frame).
    """
    h, w = frame_bgr.shape[:2]

    # Convert normalized landmarks -> pixel coords
    pts = []
    vis = []
    for lm in landmarks:
        x_px = int(lm.x * w)
        y_px = int(lm.y * h)
        pts.append((x_px, y_px))
        # Some models provide visibility; if not, default to 1.0
        v = getattr(lm, "visibility", 1.0)
        vis.append(float(v))

    # Draw skeleton lines first
    for a, b in POSE_CONNECTIONS:
        if vis[a] >= min_visibility and vis[b] >= min_visibility:
            cv2.line(frame_bgr, pts[a], pts[b], (255, 255, 255), 2, lineType=cv2.LINE_AA)

    # Draw joints on top
    for i, (p, v) in enumerate(zip(pts, vis)):
        if v >= min_visibility:
            cv2.circle(frame_bgr, p, 4, (0, 255, 0), -1, lineType=cv2.LINE_AA)

    return frame_bgr
