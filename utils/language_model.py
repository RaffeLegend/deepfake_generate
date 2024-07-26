import os
import torch
from utils.utils import save_image, is_folder
from transformers import AutoModelForCausalLM, AutoTokenizer
import json

from PIL import Image
import base64
from io import BytesIO

from globals.prompt import PROMPT_GENERATE_DESCRIPTION

# define abstract class
class StableLanguageModel:
    def __init__(self) -> None:
        self.prompt_set = None
        self.height = None
        self.width  = None
        self.model_name  = None
        self.model_path = None
        self.save_path = None
        self.model = None

    # Encode the image at the given path to a base64 string
    def encode_image(self, image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    # Get the base64 encoded description of an image at the given path
    def get_image_des_format(self, image_path):
        encoded_image = self.encode_image(image_path)
        return encoded_image

    # set output path
    def get_save_path(self, output_path, prompt_path, image_info_path):
        folder_path = os.path.join(output_path, self.model_name+ "_output")
        is_folder(folder_path)
        self.save_path = folder_path
        self.prompt_path = os.path.join(self.save_path, prompt_path)
        self.image_info_path = os.path.join(self.save_path, image_info_path)
        return folder_path

    # get the input image path
    def get_images_path(self, set_path):
    
        file_paths = list()
        index = 0

        for root, _, files in os.walk(set_path):
            for file in files:
                file_info = dict()
                file_info["index"] = str(index).zfill(8)
                file_info["file_name"] = file
                file_info["file_path"] = root
                file_info["prompt"] = ""
                file_paths.append(file_info)
                index += 1

        with open(self.image_info_path, "w") as f:
            json.dump(file_paths, f)

        return
      
    # Load data from Json
    def load_data(self):
        with open(self.image_info_path, 'r') as f:
            data = json.load(f)

        return data

    def init_model(self, data):
        raise NotImplementedError("Subclasses should implement this!")
    
    def get_result(self, data):
        raise NotImplementedError("Subclasses should implement this!")
    

# define model sdxl-turbo
class StableLanguageModel2_12B(StableLanguageModel):
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

    def inference(self):

        data = self.load_data()

        for image_info in data:
            file_path = image_info["file_path"]
            file_name = image_info["file_name"]
            image_des_format = self.get_image_des_format(os.path.join(file_path, file_name))
            description = self.generate_description_with_lm(image_des_format)
            image_info["prompt"] = description

        with open(self.image_info_path, 'w') as f:
            json.dump(data, f)

        return 
 

 # model factory
class LMModelFactory:
    @staticmethod
    def get_model(model_name):
        if model_name == "SL2_12B":
            return StableLanguageModel2_12B(model_name=model_name)
        else:
            raise ValueError(f"Unknown model name: {model_name}")