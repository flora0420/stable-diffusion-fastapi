import os
from typing import Union

import torch
from diffusers import StableDiffusionPipeline
from PIL.Image import Image

token = os.getenv("HUGGINGFACE_TOKEN")

# get your token at https://huggingface.co/settings/tokens
model_id =  "CompVis/stable-diffusion-v1-4" # "runwayml/stable-diffusion-v1-5" #
device =  "cuda" # "mps" 


pipe = StableDiffusionPipeline.from_pretrained(
    pretrained_model_name_or_path = model_id,
    revision="fp16",
    torch_dtype=torch.float16,
    use_auth_token=token,
)
pipe = pipe.to(device)
pipe.enable_attention_slicing() # < 64 GB of Ram

# prompt = "a photo of an astronaut riding a horse on mars"
# image = pipe(prompt).images[0]      
# image.save("astronaut_rides_horse.png")

def obtain_image(
    prompt: str,
    *,
    # seed: int | None = None,
    seed: Union[int, None] = None,
    num_inference_steps: int = 50,
    guidance_scale: float = 7.5,
) -> Image:
    generator = None if seed is None \
        else torch.Generator().manual_seed(seed)
    print(f"Using device: {pipe.device}")
    image: Image = pipe(
        prompt,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        generator=generator,
    ).images[0]
    return image


# image = obtain_image(prompt, num_inference_steps=2, seed=1024)
