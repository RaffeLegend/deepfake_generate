import torch
from utils.tools import save_image

from diffusers import FluxPipeline
                      
from prompts.prompt import NEGATIVE_PROMPT
from models.image_model.abstract import DiffusionModel

# define model IterComp
class IterComp(DiffusionModel):
    def __init__(self, model_name):
        super().__init__()
        self.prompt_set = None
        self.model_name = model_name
        self.model_path = "comin/IterComp"
        self.torch_dtype = torch.float16
        self.variant = "fp16"
        # self.custom_pipeline="lpw_stable_diffusion"
        # self.save_path = self.get_save_path()

    def init_model(self):
        self.model = FluxPipeline.from_pretrained(
                                    self.model_path,
                                    torch_type=self.torch_dtype,
                                    )
        self.model.to("cuda")
        self.set_prompt_enhancer()

    def inference(self):
        for patch_data in self.data_sets:
            json_data = self.load_json(patch_data)
            output_path = self.get_output_path(patch_data)
            for data_info in json_data:
                index  = data_info["index"]
                prompt = data_info["prompt"]
                prompt = self.prompt_process(prompt, NEGATIVE_PROMPT)
                image = self.model(
                            prompt=prompt,
                            num_inference_steps=50,
                            guidance_scale=7.0,
                            ).images[0]
                save_image(image, output_path, index)
        return