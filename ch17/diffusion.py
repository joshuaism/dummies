import torch
from diffusers import AutoPipelineForText2Image

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if device == "cuda" else torch.float32

pipe = AutoPipelineForText2Image.from_pretrained(
    "stable-diffusion-v1-5/stable-diffusion-v1-5",
    torch_dtype=dtype,
    use_safetensors=True,
).to(device)

image = pipe(
    "a large dog eating a book, cozy lamp light",
    num_inference_steps=25
).images[0]
image.save("datasets_and_media/txt2img.png")
print("saved datasets_and_media/txt2img.png")
