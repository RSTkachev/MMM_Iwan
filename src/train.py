import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import torch
from diffusers import AutoencoderKLWan, WanImageToVideoPipeline
from diffusers.utils import export_to_video, load_image

if torch.cuda.device_count() != 1:
    raise Exception("Not correct number of GPUs")
else:
    device = torch.device('cuda')

import numpy as np
import cv2
from PIL import Image
from peft import LoraConfig, get_peft_model
from datasets import load_dataset
from accelerate import Accelerator
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm.auto import tqdm
from torchvision.transforms import Compose, ToTensor, Normalize


model_id = "Wan-AI/Wan2.2-TI2V-5B-Diffusers"

vae = AutoencoderKLWan.from_pretrained(model_id, subfolder="vae", torch_dtype=torch.float32, cache_dir="./weights")
pipe = WanImageToVideoPipeline.from_pretrained(model_id, vae=vae, torch_dtype=torch.bfloat16, cache_dir="./weights")
pipe.to(device)

image = Image.open('i2v_input.JPG')
prompt = "Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard. The fluffy-furred feline gazes directly at the camera with a relaxed expression. Blurred beach scenery forms the background featuring crystal-clear waters, distant green hills, and a blue sky dotted with white clouds. The cat assumes a naturally relaxed posture, as if savoring the sea breeze and warm sunlight. A close-up shot highlights the feline's intricate details and the refreshing atmosphere of the seaside."

output_dir = "./weights_finetuned"
lora_rank = 32
lora_alpha = 64
lora_dropout = 0.1
batch_size = 1
accumulation_steps = 4
epochs = 5
learning_rate = 1e-5
dataset_file = "train.jsonl"
num_frames = 121

if not os.path.exists(output_dir):
    os.mkdir(output_dir)

accelerator = Accelerator(mixed_precision="bf16", gradient_accumulation_steps=accumulation_steps)

vae = pipe.vae
text_encoder = pipe.text_encoder
scheduler = pipe.scheduler
image_processor = pipe.image_processor
dit = pipe.transformer

transform = Compose([
    ToTensor(),
    Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

lora_config = LoraConfig(
    r=lora_rank,
    lora_alpha=lora_alpha,
    target_modules=[
        "to_q", "to_k", "to_v", "to_out.0",
        "ffn.net.0.proj", "ffn.net.2",
        "proj_out"
    ],
    lora_dropout=lora_dropout,
    bias="none"
)

dit = get_peft_model(dit, lora_config)
dit.print_trainable_parameters()

optimizer = AdamW(dit.parameters(), lr=learning_rate)
dit, optimizer = accelerator.prepare(dit, optimizer)

def preprocess(example):
    global pipe
    global image_processor
    global transform
    global vae
    global text_encoder
    
    img = Image.open(example["image"]).convert("RGB")
    img = img.resize((1280, 704))
    
    cap = cv2.VideoCapture(example["video"])
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = Image.fromarray(frame).resize((1280, 704))
        frames.append(frame)
    cap.release()
    if len(frames) != num_frames:
        raise ValueError(f"Video must have {num_frames} frames")
    
    with torch.no_grad():
        if image_processor is not None:
            img_input = image_processor(img, return_tensors="pt").pixel_values.to(device)
        else:
            img_input = transform(img).unsqueeze(0).to(device)
        img_input = img_input.unsqueeze(2)

        if image_processor is not None:
            frame_tensors = [image_processor(f, return_tensors="pt").pixel_values.squeeze(0).to(device) for f in frames]
        else:
            frame_tensors = [transform(f).to(device) for f in frames]
        video_tensor = torch.stack(frame_tensors, dim=0).to(device)
        video_tensor = video_tensor.permute(1, 0, 2, 3).unsqueeze(0)

        if hasattr(pipe, 'image_encoder') and pipe.image_encoder is not None:
            img_latent = pipe.image_encoder(img_input).last_hidden_state.squeeze(0)
        else:
            img_latent = vae.encode(img_input).latent_dist.sample().squeeze(0)

        video_latents = vae.encode(video_tensor).latent_dist.sample().squeeze(0)

        if img_latent.dim() == 4 and img_latent.shape[1] == 1:
            img_latent = img_latent.squeeze(1)

    if "prompt" in example:
        tokens = pipe.tokenizer(example["prompt"], return_tensors="pt").input_ids.to(device)
        text_emb = text_encoder(tokens).last_hidden_state
    else:
        text_emb = None

    return {"img_latent": img_latent, "text_emb": text_emb, "video_latents": video_latents}

def custom_collate(batch):
    collated = {}
    for key in batch[0].keys():
        values = [item[key] for item in batch]
        if key == "text_emb":
            non_none_values = [v for v in values if v is not None]
            if non_none_values:
                max_seq = max(v.shape[1] for v in non_none_values)
                padded = []
                for v in values:
                    if v is None:
                        zero_shape = list(non_none_values[0].shape)
                        zero_shape[1] = max_seq
                        padded.append(torch.zeros(zero_shape, dtype=non_none_values[0].dtype, device=non_none_values[0].device))
                    else:
                        pad_amount = (0, 0, 0, max_seq - v.shape[1])
                        padded.append(torch.nn.functional.pad(v, pad_amount))
                collated[key] = torch.cat(padded, dim=0)
            else:
                collated[key] = None
        else:
            collated[key] = torch.stack(values)
    return collated

dataset = load_dataset("json", data_files=dataset_file, split="train")
processed_dataset = []
for sample in tqdm(dataset):
    processed_dataset.append(preprocess(sample))
dataloader = DataLoader(processed_dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate)

dit.train()
for epoch in range(epochs):
    progress = tqdm(dataloader)
    for step, batch in enumerate(progress):
        with accelerator.accumulate(dit):
            latents = batch["video_latents"]
            img_latent = batch["img_latent"]
            text_emb = batch["text_emb"]

            cond = text_emb

            noise = torch.randn_like(latents)
            timesteps = torch.randint(0, scheduler.config.num_train_timesteps, (batch_size,), device=latents.device)
            noisy_latents = scheduler.add_noise(latents, noise, timesteps)

            img_latent = img_latent.unsqueeze(2)
            img_noise = noise[:, :, :1, :, :]
            noisy_img = scheduler.add_noise(img_latent, img_noise, timesteps)
            noisy_latents[:, :, :1, :, :] = noisy_img

            model_pred = dit(noisy_latents, timesteps, encoder_hidden_states=cond)

            loss = torch.nn.functional.mse_loss(model_pred, noise)
            accelerator.backward(loss)
            optimizer.step()
            optimizer.zero_grad()

        progress.set_description(f"Epoch {epoch}, Loss: {loss.item():.4f}")

    accelerator.save_state(os.path.join(output_dir, f"checkpoint_{epoch}"))

dit.save_pretrained(os.path.join(output_dir, "lora_adapter_transformer"))
print("Fine-tuning завершён. LoRA сохранён в", output_dir)
