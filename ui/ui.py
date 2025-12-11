import gradio as gr
import requests
import os

API_URL = "http://localhost:8000/inference_by_image"

def call_inference(image):
    if image is None:
        return None

    # отправляем картинку
    image.save("temp.png")
    with open("temp.png", "rb") as f:
        r = requests.post(API_URL, files={"image": ("temp.png", f, "image/png")})

    # сохраняем видеофайл
    video_path = "result.mp4"
    with open(video_path, "wb") as f:
        f.write(r.content)

    return video_path  # отдаём Gradio путь к видео

app = gr.Interface(
    fn=call_inference,
    inputs=gr.Image(type="pil"),
    outputs=gr.Video(),
    title="WAN Video Generator",
    description="Загружай картинку — получай видео"
)

if __name__ == "__main__":
    app.launch(server_name="0.0.0.0")
