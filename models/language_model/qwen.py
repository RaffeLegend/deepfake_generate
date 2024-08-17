import os
import torch
import json
from PIL import Image, UnidentifiedImageError

from transformers import AutoTokenizer, AutoModelForCausalLM

from prompts.prompt import PROMPT_GENERATE_DESCRIPTION
from models.language_model.abstract import LanguageModel

# define model Qwen2
class Qwen2(LanguageModel):
    def __init__(self, model_name):
        super().__init__()
        self.prompt_set = None
        self.model_name = model_name
        self.token_path = "MaziyarPanahi/calme-2.1-qwen2-72b"
        self.model_path = "MaziyarPanahi/calme-2.1-qwen2-72b"
        self.torch_dtype = torch.bfloat16
        self.variant = "fp16"
        self.prompt = PROMPT_GENERATE_DESCRIPTION
        self.image_size = 448
        self.generation_config = dict(
                            num_beams=1,
                            max_new_tokens=1024,
                            do_sample=False,
                            )
        # self.save_path = self.get_save_path()

    def init_model(self):

        self.tokenizer = AutoTokenizer.from_pretrained(self.token_path)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_path)
        self.model.cuda()
    
    def load_image(self, image_file):
        try:
            image = Image.open(image_file).convert('RGB')
            return self.image_processor(image)
        except (UnidentifiedImageError) as e:
            print(f"Failed to load the image {image_file}: {e}")
            return None

    def generate_description(self, image):
        # single-image single-round conversation
        question = self.prompt
        msgs = [{'role': 'user', 'content': [image, question]}]
        response = self.model.chat(
                        image=None,
                        msgs=msgs,
                        tokenizer=self.tokenizer,
                        )
        return response    

    # define the conduct step
    def conduct(self, path, name):
        image_path = os.path.join(path, name)
        image = self.load_image(image_path)
        if image is not None:
            image_des_format = image.to(self.torch_dtype).cuda()
            description = self.generate_description(image_des_format)
        else:
            description = ""
        return description

    def inference(self):

        data_sets = self.load_data()
        for data_set in data_sets:
            with open(data_set, 'r') as f:
                data = json.load(f)
                for image_info in data:
                    file_path = image_info["file_path"]
                    file_name = image_info["file_name"]
                    description = self.conduct(file_path, file_name)
                    image_info["prompt"] = description

            with open(data_set, 'w') as f:
                json.dump(data, f)

        return 