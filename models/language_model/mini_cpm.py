import os
import torch
import json
from PIL import Image, UnidentifiedImageError

from transformers import AutoTokenizer, AutoModel

from prompts.prompt import PROMPT_GENERATE_DESCRIPTION
from models.language_model.abstract import LanguageModel

# define model MiniCPM
class MiniCPM(LanguageModel):
    def __init__(self, model_name):
        super().__init__()
        self.prompt_set = None
        self.model_name = model_name
        self.token_path = "openbmb/MiniCPM-V-2_6"
        self.model_path = "openbmb/MiniCPM-V-2_6"
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
        self.tokenizer = AutoTokenizer.from_pretrained(self.token_path, trust_remote_code=True)

        self.model = AutoModel.from_pretrained(
                            self.model_path, 
                            trust_remote_code=True,
                            attn_implementation='sdpa', 
                            torch_dtype=self.torch_dtype,
                            ) # sdpa or flash_attention_2, no eager
        self.model.eval().cuda()
    
    def load_image(self, image_file):
        try:
            image = Image.open(image_file).convert('RGB')
        except (UnidentifiedImageError) as e:
            print(f"Failed to load the image {image_file}: {e}")
            return None
                
        return image
    
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