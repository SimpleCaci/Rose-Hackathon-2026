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

from pose_tasks import PoseTasksLandmarker, PoseTasksConfig, draw_pose_landmarks_on_frame


app = FastAPI(title="FastAPI Website Boilerplate")

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "title": "Home"})
<<<<<<< HEAD

=======
>>>>>>> 8777ac371c1b37d264d214a281ed1c8859cf8f7f

@app.get("/health")
def health():
    return {"ok": True}


class Camera:
    def __init__(self, src: int = 0):
        self.cap = cv2.VideoCapture(src)
        if not self.cap.isOpened():
            raise RuntimeError("Could not open camera/video source.")

        # Pose landmarker (Tasks API)
        self.pose = PoseTasksLandmarker(PoseTasksConfig(model_path="pose_landmarker_full.task"))

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

            # ---- Pose detect + draw on the camera frame ----
            landmarks = self.pose.detect_bgr(frame)
            if landmarks is not None:
                draw_pose_landmarks_on_frame(frame, landmarks, min_visibility=0.5)

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
        try:
            self.cap.release()
        except Exception:
            pass


camera = None


@app.on_event("startup")
def startup():
    global camera
    camera = Camera(src=0)  # if Codespaces, use a video file path instead of 0


@app.on_event("shutdown")
def shutdown():
    global camera
    if camera:
        camera.close()
        camera = None


def mjpeg_generator():
    while True:
        frame = camera.get_jpeg() if camera else None
        if frame is None:
            time.sleep(0.01)
            continue
        yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"


@app.get("/video")
def video_feed():
    return StreamingResponse(
        mjpeg_generator(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )
