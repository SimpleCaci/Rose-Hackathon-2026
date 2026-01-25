# pose_tasks.py
# MediaPipe Tasks Pose Landmarker + drawing (NO mp.solutions usage)

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# -----------------------------
# Pose skeleton connections (33 landmark format)
# These are the same connections MediaPipe uses internally.
# -----------------------------

POSE_EDGES = [
    (0, 1), (0, 4),
    (1, 2), (2, 3),
    (4, 5), (5, 6),

    (11, 12),
    (11, 13), (13, 15),
    (12, 14), (14, 16),

    (11, 23),
    (12, 24),
    (23, 24),

    (23, 25), (25, 27),
    (24, 26), (26, 28),

    (27, 31), (28, 32),
    (29, 31), (30, 32),

    (15, 17), (16, 18),
    (15, 19), (16, 20),
    (19, 21), (20, 22),
]


# -----------------------------
# Pose Detector (MediaPipe Tasks)
# -----------------------------

class PoseDetector:
    def __init__(self, model_path="pose_landmarker_full.task"):

        base_options = python.BaseOptions(model_asset_path=model_path)

        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            num_poses=1,
        )

        self.detector = vision.PoseLandmarker.create_from_options(options)
        self.timestamp = 0


    def detect(self, frame_bgr):
        """
        Input: OpenCV frame (BGR)
        Output: landmarks list (length 33) or None
        """

        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb
        )

        self.timestamp += 33

        result = self.detector.detect_for_video(mp_image, self.timestamp)

        if not result.pose_landmarks:
            return None

        return result.pose_landmarks[0]


# -----------------------------
# Drawing function
# -----------------------------

def draw_pose(frame, landmarks):
    """
    Draw pose dots + skeleton directly onto the camera frame.
    Works purely from Tasks output.
    """

    h, w = frame.shape[:2]

    # Convert normalized landmarks → pixel coordinates
    points = []
    for lm in landmarks:
        x = int(lm.x * w)
        y = int(lm.y * h)
        points.append((x, y))

    # Draw skeleton lines
    for a, b in POSE_EDGES:
        cv2.line(frame, points[a], points[b],
                 (255, 255, 255), 2)

    # Draw joints as green dots
    for (x, y) in points:
        cv2.circle(frame, (x, y),
                   4, (0, 255, 0), -1)

    return frame
