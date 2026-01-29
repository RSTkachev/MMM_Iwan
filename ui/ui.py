import gradio as gr
import requests
import numpy as np
import os
from diffusers.utils import export_to_video

API_URL = os.getenv(
    "API_URL",
    "http://wan_api:8000/generate"
)

DEMO_VIDEO_PATH = "result.mp4"


def call_inference(image, prompt):
    if image is None or not prompt:
        return None

    # PIL -> numpy -> list
    image_array = np.array(image, dtype=np.uint8).tolist()

    payload = {
        "image": image_array,
        "prompt": prompt,
        "width": 832,
        "height": 480,
        "num_frames": 121,
    }

    r = requests.post(API_URL, json=payload)
    r.raise_for_status()

    # API возвращает video array
    video_np = np.array(r.json()["video"], dtype=np.uint8)

    # ⬇️ КЛЮЧЕВОЕ: возвращаем массив напрямую
    return video_np


def demo_inference(image, prompt):
    # Demo оставляем файловым, Gradio это поддерживает
    return DEMO_VIDEO_PATH


with gr.Blocks(title="WAN Video Generator") as app:
    gr.Markdown("# WAN Video Generator")

    with gr.Row():
        image_input = gr.Image(type="pil", label="Input image")
        prompt_input = gr.Textbox(label="Prompt")

    with gr.Row():
        generate_btn = gr.Button("Generate")
        demo_btn = gr.Button("Demo")

    video_output = gr.Video(label="Output video")

    generate_btn.click(
        fn=call_inference,
        inputs=[image_input, prompt_input],
        outputs=video_output,
    )

    demo_btn.click(
        fn=demo_inference,
        inputs=[image_input, prompt_input],
        outputs=video_output,
    )

if __name__ == "__main__":
    app.launch(server_name="0.0.0.0")
