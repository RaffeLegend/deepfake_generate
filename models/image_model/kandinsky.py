import torch
from utils.utils import save_image
from diffusers import Kandinsky3Pipeline

from sd_embed.embedding_funcs import get_weighted_text_embeddings_sd3
from globals.prompt import NEGATIVE_PROMPT, PROMPT_REALISTIC_VISION_NEGATIVE
from globals.prompt_enhance import PromptEnhancer

from models.image_model.abstract import DiffusionModel

# define model sd cascade
class Kandinsky3(DiffusionModel):
    def __init__(self, model_name):
        super().__init__()
        self.prompt_set = None
        self.model_name = model_name
        self.model_path = "kandinsky-community/kandinsky-3"
        self.torch_dtype = torch.float16
        self.variant = "fp16"
        # self.save_path = self.get_save_path()
        self.negative_prompt = ""

    def init_model(self):
        self.model = Kandinsky3Pipeline.from_pretrained(self.model_path, variant=self.variant, torch_dtype=self.torch_dtype)
        self.model.enable_model_cpu_offload()

    def inference(self):
        for patch_data in self.data_sets:
            json_data = self.load_json(patch_data)
            output_path = self.get_output_path(patch_data)
            for data_info in json_data:
                index  = data_info["index"]
                prompt = data_info["prompt"]
                image = self.model(prompt).images[0]
                save_image(image, output_path, index)
        return 