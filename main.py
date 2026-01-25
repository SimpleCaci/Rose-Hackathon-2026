# main.py
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, StreamingResponse
from starlette.requests import Request

import cv2
import threading
import time
import numpy as np

import mediapipe as mp

# ✅ Import your overlay drawer
from pose_overlay import draw_pose_on_frame, IDX

app = FastAPI(title="FastAPI Website Boilerplate")

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "title": "Home"})

@app.get("/health")
def health():
    return {"ok": True}


# -----------------------
# OpenCV camera streamer
# -----------------------

class Camera:
    def __init__(self, src: int = 0):
        self.cap = cv2.VideoCapture(src)
        if not self.cap.isOpened():
            raise RuntimeError("Could not open camera.")

        self.lock = threading.Lock()
        self.latest_jpeg = None
        self.running = True

        threading.Thread(target=self._reader_loop, daemon=True).start()

    def _reader_loop(self):
        while self.running:
            ok, frame = self.cap.read()
            if not ok:
                time.sleep(0.05)
                continue

            # For demo purposes, create fake keypoints
            h, w = frame.shape[:2]
            keypoints = np.zeros((17, 3), dtype=np.float32)
            keypoints[:, 2] = 1.0  # confidence = 1

            keypoints[IDX["nose"]] = (w * 0.5, h * 0.2, 1)
            keypoints[IDX["left_shoulder"]] = (w * 0.43, h * 0.35, 1)
            keypoints[IDX["right_shoulder"]] = (w * 0.57, h * 0.35, 1)
            keypoints[IDX["left_hip"]] = (w * 0.46, h * 0.60, 1)
            keypoints[IDX["right_hip"]] = (w * 0.54, h * 0.60, 1)
            keypoints[IDX["left_knee"]] = (w * 0.47, h * 0.80, 1)
            keypoints[IDX["right_knee"]] = (w * 0.53, h * 0.80, 1)

            frame = draw_pose_on_frame(frame, keypoints)

            ok, buf = cv2.imencode(".jpg", frame)
            if ok:
                with self.lock:
                    self.latest_jpeg = buf.tobytes()

            time.sleep(0.01)

    def get_jpeg(self):
        with self.lock:
            return self.latest_jpeg

    def close(self):
        self.running = False
        self.cap.release()


camera = None


@app.on_event("startup")
def startup():
    global camera
    camera = Camera(0)


@app.on_event("shutdown")
def shutdown():
    if camera:
        camera.close()


def mjpeg_generator():
    while True:
        frame = camera.get_jpeg()
        if frame is None:
            time.sleep(0.01)
            continue

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n"
            + frame
            + b"\r\n"
        )


@app.get("/video")
def video_feed():
    return StreamingResponse(
        mjpeg_generator(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )
