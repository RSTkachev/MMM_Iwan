from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel
import uvicorn
import os

from model_service import ModelService

app = FastAPI()
model_service = ModelService()

@app.on_event("startup")
async def startup_event():
    model_service.load()

class GenerateRequest(BaseModel):
    image_path: str
    prompt: str
    negative_prompt: str | None = None
    width: int = 832
    height: int = 480
    num_frames: int = 121

@app.post("/generate")
def generate_video(req: GenerateRequest):
    video_path = model_service.predict(
        image_path=req.image_path,
        prompt=req.prompt,
        negative_prompt=req.negative_prompt,
        width=req.width,
        height=req.height,
        num_frames=req.num_frames,
    )

    return FileResponse(
        path=video_path,
        media_type="video/mp4",
        filename=os.path.basename(video_path),
    )

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000)
