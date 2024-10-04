import torch
from utils.tools import save_image
from diffusers import DiffusionPipeline
                      
from prompts.prompt import NEGATIVE_PROMPT, PROMPT_REALISTIC_VISION_NEGATIVE
from models.image_model.abstract import DiffusionModel

# define model absolute reality
class AbsoluteReality(DiffusionModel):
    def __init__(self, model_name):
        super().__init__()
        self.prompt_set = None
        self.model_name = model_name
        self.model_path = "Lykon/AbsoluteReality"
        self.torch_dtype = torch.float16
        self.variant = "fp16"
        # self.custom_pipeline="lpw_stable_diffusion"
        # self.save_path = self.get_save_path()

    def init_model(self):
        self.model = DiffusionPipeline.from_pretrained(
                                    self.model_path,
                                    )
        self.model.to("cuda")
        # self.set_prompt_enhancer()

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
                            negative_prompt=PROMPT_REALISTIC_VISION_NEGATIVE,
                            num_inference_steps=40,
                            guidance_scale=7.0,
                            ).images[0]
                save_image(image, output_path, index)
        return 
