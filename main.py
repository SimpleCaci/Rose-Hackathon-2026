# main.py
# FastAPI Server

# Library Importations:
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from starlette.requests import Request

# Local Importations:



# FastAPI APP Initialization:
app = FastAPI(title="FastAPI Website Boilerplate")

# static + templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {"request": request, "title": "Home"},
    )


@app.get("/health")
def health():
    return {"ok": True}
