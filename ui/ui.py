import gradio as gr
import requests
import numpy as np
import os

API_URL = os.getenv(
    "API_URL",
    "http://localhost:8000/generate"
)
DEMO_VIDEO_PATH = "result.mp4"


def call_inference(image, prompt):
    if image is None or not prompt:
        return None

    image_array = np.array(image).tolist()

    payload = {
        "image": image_array,
        "prompt": prompt,
        "width": 832,
        "height": 480,
        "num_frames": 121,
    }

    r = requests.post(API_URL, json=payload)
    r.raise_for_status()

    video_path = "generated.mp4"
    with open(video_path, "wb") as f:
        f.write(r.content)

    return video_path


def demo_inference(image, prompt):
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
