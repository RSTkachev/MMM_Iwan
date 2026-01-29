from pydantic import BaseModel

import numpy as np
from PIL import Image
import torch
from diffusers import AutoencoderKLWan, WanImageToVideoPipeline

from fastapi import FastAPI, HTTPException


app = FastAPI()
model = None


class GenerateRequest(BaseModel):
    image: list[list[list[int]]]
    prompt: str
    negative_prompt: str
    height: int
    width: int
    num_frames: int


@app.on_event("startup")
def load_model():
    global model

    model_id = "Wan-AI/Wan2.2-TI2V-5B-Diffusers"
    
    vae = AutoencoderKLWan.from_pretrained(model_id, subfolder="vae", torch_dtype=torch.float32)
    model = WanImageToVideoPipeline.from_pretrained("Wan-AI/Wan2.2-TI2V-5B-Diffusers", torch_dtype=torch.bfloat16)
    model.transformer.load_lora_adapter('lora/epoch-12.safetensors', prefix=None)
    model.enable_model_cpu_offload() 
    

@app.get("/health")
def check_health():
    global model
    if model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Model not loaded"
        )
    return {"status": "OK"}


@app.post("/generate_video")
async def generate_video(request: GenerateRequest):
    global model

    image = np.array(request.image, dtype=np.uint8)
    prompt = request.prompt
    negative_prompt = request.negative_prompt
    height = request.height
    width = request.width
    num_frames = request.num_frames

    image = Image.fromarray(image)
    
    max_area = width * height
    
    aspect_ratio = image.height / image.width
    mod_value = model.vae_scale_factor_spatial * model.transformer.config.patch_size[1]
    height = round(np.sqrt(max_area * aspect_ratio)) // mod_value * mod_value
    width = round(np.sqrt(max_area / aspect_ratio)) // mod_value * mod_value
    image = image.resize((width, height))

    
    output = model(
        image=image,
        prompt=prompt,
        negative_prompt=negative_prompt,
        height=height,
        width=width,
        num_frames=num_frames,
        num_inference_steps=50,
        guidance_scale=3.5
    ).frames[0]

    
    return {"video": output.tolist()}