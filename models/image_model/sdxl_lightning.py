import torch
from utils.tools import save_image
from diffusers import StableDiffusionXLPipeline, UNet2DConditionModel, EulerDiscreteScheduler
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

from sd_embed.embedding_funcs import get_weighted_text_embeddings_sd3
from prompts.prompt import NEGATIVE_PROMPT, PROMPT_REALISTIC_VISION_NEGATIVE
from prompts.prompt_enhance import PromptEnhancer

from models.image_model.abstract import DiffusionModel

# define model playground
class SDXLLightning(DiffusionModel):
    def __init__(self, model_name):
        super().__init__()
        self.prompt_set = None
        self.model_name = model_name
        self.model_path = "ByteDance/SDXL-Lightning"
        self.torch_dtype = torch.float16
        self.variant = "fp16"
        # self.save_path = self.get_save_path()
        self.negative_prompt = ""
        self.base = "stabilityai/stable-diffusion-xl-base-1.0"
        self.ckpt = "sdxl_lightning_4step_unet.safetensors" 

    def init_model(self):
        self.unet = UNet2DConditionModel.from_config(self.base, subfolder="unet").to("cuda", torch.float16)
        self.unet.load_state_dict(load_file(hf_hub_download(self.model, self.ckpt), device="cuda"))
        self.model = StableDiffusionXLPipeline.from_pretrained(self.base, unet=self.unet, torch_dtype=torch.float16, variant="fp16").to("cuda")

        self.model.scheduler = EulerDiscreteScheduler.from_config(self.model.scheduler.config, timestep_spacing="trailing")

        # self.set_prompt_enhancer()

    def inference(self):
        for patch_data in self.data_sets:
            json_data = self.load_json(patch_data)
            output_path = self.get_output_path(patch_data)
            for data_info in json_data:
                index  = data_info["index"]
                prompt = data_info["prompt"]
                prompt = self.prompt_process(prompt, NEGATIVE_PROMPT)
                image = self.model(prompt=prompt, 
                                   num_inference_steps=4, 
                                   guidance_scale=0
                                   ).images[0]
                save_image(image, output_path, index)
        return 