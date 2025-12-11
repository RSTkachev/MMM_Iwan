from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse
import uvicorn
from model_service import ModelService
import os

app = FastAPI()

model_service = ModelService()

@app.on_event("startup")
async def startup_event():
    model_service.load()

@app.post("/inference_by_image")
async def inference_by_image(image: UploadFile = File(...)):
    # читаем изображение
    image_bytes = await image.read()

    # модель возвращает путь к видеофайлу
    video_path = model_service.predict(image_bytes)

    # отдаём видео как файл
    return FileResponse(
        path=video_path,
        media_type="video/mp4",
        filename=os.path.basename(video_path)
    )

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
