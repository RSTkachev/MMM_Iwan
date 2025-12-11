import os

import torch
from diffusers import AutoencoderKLWan, WanImageToVideoPipeline
from diffusers.utils import export_to_video, load_image

device = torch.device(
    'cuda' if torch.cuda.is_available()
    else 'mps' if torch.mps.is_available()
    else 'cpu'
)

model_id = "Wan-AI/Wan2.2-TI2V-5B-Diffusers"

vae = AutoencoderKLWan.from_pretrained(model_id, subfolder="vae", torch_dtype=torch.float32, cache_dir="./weights")
pipe = WanImageToVideoPipeline.from_pretrained(model_id, vae=vae, torch_dtype=torch.bfloat16, device_map='balanced', cache_dir="./weights")

image = load_image(
    "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/astronaut.jpg",
)
max_area = 1280 * 704
aspect_ratio = image.height / image.width
mod_value = pipe.vae_scale_factor_spatial * pipe.transformer.config.patch_size[1]
height = round(np.sqrt(max_area * aspect_ratio)) // mod_value * mod_value
width = round(np.sqrt(max_area / aspect_ratio)) // mod_value * mod_value
image = image.resize((width, height))
prompt = (
    "An astronaut hatching from an egg, on the surface of the moon, the darkness and depth of space realised in "
    "the background. High quality, ultrarealistic detail and breath-taking movie-like camera shot."
)
negative_prompt = "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards"

output = pipe(
    image=image,
    prompt=prompt,
    negative_prompt=negative_prompt,
    height=height, width=width,
    num_frames=121,
).frames[0]
export_to_video(output, "output.mp4", fps=24)
