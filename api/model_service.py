import os
import uuid
import requests
import numpy as np
from diffusers.utils import export_to_video


class ModelService:
    def __init__(self):
        self.output_dir = "./outputs"
        os.makedirs(self.output_dir, exist_ok=True)

        self.wan_url = "http://localhost:11222/generate_video"

    def load(self):
        print("[ModelService] WAN proxy service ready")

    def predict(
        self,
        image_array: list | np.ndarray,
        prompt: str,
        negative_prompt: str | None,
        width: int,
        height: int,
        num_frames: int,
    ) -> str:
        if isinstance(image_array, list):
            image_array = np.array(image_array, dtype=np.uint8)

        payload = {
            "image": image_array.tolist(),
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "width": width,
            "height": height,
            "num_frames": num_frames,
        }

        response = requests.post(self.wan_url, json=payload)
        response.raise_for_status()

        video_np = np.array(response.json()["video"])

        out_path = os.path.join(
            self.output_dir, f"{uuid.uuid4()}.mp4"
        )

        export_to_video(video_np, out_path, fps=24)

        return out_path
