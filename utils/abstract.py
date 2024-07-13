from utils.utils import save_image, is_folder
import torch
from diffusers import AutoPipelineForText2Image

# define abstract class
class StableDiffusion:
    def __init__(self) -> None:
        self.prompt = None
        self.height = None
        self.width  = None
        self.image  = None
        self.output = None

    def load_data(self, data):
        raise NotImplementedError("Subclasses should implement this!")

    def conduct(self, data):
        raise NotImplementedError("Subclasses should implement this!")
    
    def get_result(self, data):
        raise NotImplementedError("Subclasses should implement this!")

# define model sdxl
class StableDiffusionXLTurbo(StableDiffusion):
    def __init__(self, model_name):
        super().__init__()
        self.prompt_set = None
        self.model_name = model_name
        self.model_path = "stabilityai/" + model_name
        self.torch_dtype = torch.float16
        self.variant = "fp16"
        self.model = self.init_model()
        self.save_path = self.get_save_path()
    
    def get_save_path(self):
        folder_name = "./" + self.model_name.split("/")[1] + "_output"
        is_folder(folder_name)
        return folder_name

    def load_data(self, prompt_path):
        prompt_set = dict()
        prompt = ""
        with open(prompt_path, 'r') as f:
            for line in f:
                if line != "++++++++\n":
                    prompt += line
                else:
                    prompt_set.append("create image by the description: " + prompt)
                    prompt = ""
        print(f"Collect {len(prompt_set)} prompts to generate images")
        self.prompt_set = prompt_set

    def init_model(self):
        pipe = AutoPipelineForText2Image.from_pretrained(self.model_path, torch_dtype=self.torch_dtype, variant=self.variant)
        pipe.to("cuda")
        return pipe

    def inference(self):
        for prompt in self.prompt_set:
            image = self.model(prompt=prompt, num_inference_steps=3, guidance_scale=0.3).images[0]
            save_image(image, self.save_path, index)
            index += 1
        return 

# define the second model
# class ModelB(Model):
#     def train(self, data):
#         print("Training ModelB with data:", data)

#     def evaluate(self, data):
#         print("Evaluating ModelB with data:", data)
#         return "ModelB evaluation result"

# model factory
class ModelFactory:
    @staticmethod
    def get_model(model_name):
        if model_name == "sdxl-turbo":
            return StableDiffusionXLTurbo(model_name=model_name)
        else:
            raise ValueError(f"Unknown model name: {model_name}")
