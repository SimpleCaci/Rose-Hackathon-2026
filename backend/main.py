from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path
import json

ROOT = Path(__file__).resolve().parent.parent
WEB = ROOT / "web"

app = FastAPI()

app.mount("/web", StaticFiles(directory=str(WEB)), name="web")


@app.get("/")
def home():
    return FileResponse(str(WEB / "index.html"))


@app.get("/api/poses")
def poses():
    return json.loads((WEB / "poses.json").read_text(encoding="utf-8"))
