from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel
import beat_generator
import base64
import os

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class GenerateRequest(BaseModel):
    bpm: int
    instruments: list[str] = []

@app.post("/generate")
async def generate_beat(req: GenerateRequest):
    wav_io, events = beat_generator.generate_beat(req.bpm, req.instruments, duration_sec=15)
    wav_data = wav_io.read()
    b64_audio = base64.b64encode(wav_data).decode('utf-8')
    
    return JSONResponse(content={
        "status": "success",
        "audio": b64_audio,
        "events": events
    })

# SERVE FRONTEND (Must be last to avoid catching API routes)
# We assume 'frontend' folder is one level up from 'backend' or accessible.
# In production, structure might differ, but for this repo structure:
# Project/
#   backend/main.py
#   frontend/index.html

# Determine path to frontend relative to this file
current_dir = os.path.dirname(os.path.abspath(__file__))
frontend_dir = os.path.join(current_dir, "../frontend")

if os.path.exists(frontend_dir):
    app.mount("/", StaticFiles(directory=frontend_dir, html=True), name="static")

