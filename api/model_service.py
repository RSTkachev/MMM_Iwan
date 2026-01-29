import os
import uuid
import requests
import numpy as np


class ModelService:
    def __init__(self):
        self.output_dir = "./outputs"
        os.makedirs(self.output_dir, exist_ok=True)

        self.wan_url = "http://wan_model:11222/generate_video"

    def load(self):
        print("[ModelService] WAN proxy service ready")

    def predict(
        self,
        image: list | np.ndarray,
        prompt: str,
        negative_prompt: str | None,
        width: int,
        height: int,
        num_frames: int,
    ) -> str:
        if isinstance(image, list):
            image = np.array(image, dtype=np.uint8)

        payload = {
            "image": image.tolist(),
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "width": width,
            "height": height,
            "num_frames": num_frames,
        }

        response = requests.post(self.wan_url, json=payload)
        # response.raise_for_status()

        # video_np = np.array(response.json()["video"])



        video_np = np.array(response.json()["video"], dtype=np.uint8)

        return video_np


