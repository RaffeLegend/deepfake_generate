import os
import json
import torch

from transformers import AutoModelForCausalLM, AutoTokenizer
import torchvision.transforms as T

from prompts.prompt import PROMPT_GENERATE_DESCRIPTION
from models.language_model.abstract import LanguageModel

# define model Stable Language Model 2 12B
class StableLanguageModel2_12B(LanguageModel):
    def __init__(self, model_name):
        super().__init__()
        self.prompt_set = None
        self.model_name = model_name
        self.token_path = "stabilityai/stablelm-2-12b"
        self.model_path = "stabilityai/stablelm-2-12b"
        self.torch_dtype = torch.float16
        self.variant = "fp16"
        self.prompt = PROMPT_GENERATE_DESCRIPTION
        # self.save_path = self.get_save_path()

    def init_model(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.token_path)
        self.model = AutoModelForCausalLM.from_pretrained(
                            self.model_path,
                            torch_dtype="auto",
                            )
        self.model.cuda()

    def generate_description_with_lm(self, image_des_format):
        inputs = self.tokenizer(image_des_format, return_tensors="pt").to(self.model.device)
        tokens = self.model.generate(
                        **inputs,
                        max_new_tokens=400,
                        temperature=0.70,
                        top_p=0.95,
                        do_sample=True,
                        )
        print(self.tokenizer.decode(tokens[0], skip_special_tokens=True))
        return self.tokenizer.decode(tokens[0], skip_special_tokens=True)

    # define the conduct step
    def conduct(self, path, name):
        image_path = os.path.join(path, name)
        image_des_format = self.load_image(image_path, max_num=6).to(self.torch_dtype).cuda()
        description = self.generate_description(image_des_format)
        return description

    def inference(self):

        data_sets = self.load_data()
        for data_set in data_sets:
            with open(data_set, 'rw') as f:
                data = json.load(f)
                for image_info in data:
                    file_path = image_info["file_path"]
                    file_name = image_info["file_name"]
                    description = self.condct(file_path, file_name)
                    image_info["prompt"] = description

                json.dump(data, f)