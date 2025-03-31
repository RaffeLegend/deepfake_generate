from PIL import Image, ImageOps
import requests
import torch
from diffusers import StableDiffusionInstructPix2PixPipeline, EulerAncestralDiscreteScheduler
from PIL import Image

url = "https://huggingface.co/datasets/diffusers/diffusers-images-docs/resolve/main/mountain.png"

def download_image(url):
    image = Image.open(requests.get(url, stream=True).raw)
    image = ImageOps.exif_transpose(image)
    image = image.convert("RGB")
    return image

image = download_image(url)
prompt = "make the mountains snowy"

class MagicBrush():
    def __init__(self, weight="vinesmsuic/magicbrush-jul7"):
        self.pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
                        weight, 
                        torch_dtype=torch.float16
                    ).to("cuda")
        self.pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(self.pipe.scheduler.config)
        
    def infer_one_image(self, src_image, instruct_prompt, seed):
        generator = torch.manual_seed(seed)
        image = self.pipe(instruct_prompt, image=src_image, num_inference_steps=20, image_guidance_scale=1.5, guidance_scale=7, generator=generator).images[0]
        return image

model = MagicBrush()
image_output = model.infer_one_image(image, prompt, 42)
image_output