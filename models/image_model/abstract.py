import os
import json
from PIL import Image

from sd_embed.embedding_funcs import get_weighted_text_embeddings_sd3
from utils.tools import is_folder
from globals.define import *

from prompts.prompt import NEGATIVE_PROMPT, PROMPT_REALISTIC_VISION_NEGATIVE
from prompts.prompt_enhance import PromptEnhancer


# define abstract class
class DiffusionModel:
    def __init__(self) -> None:
        self.prompt_set = None
        self.height = None
        self.width  = None
        self.model_name  = None
        self.model_path = None
        self.save_path = None
        self.model = None
        self.prompt_enhancer = "gpt2"
        self.style = "photographic"
        self.prompt_post = False
        self.enhancer = None

    def set_prompt_enhancer(self):
        if self.prompt_enhancer == "gpt2":
            self.enhancer = PromptEnhancer()
            self.model.load_lora_weights(
                                LORA_NAME,
                                weight_name=LORA_WEIGHT_NAME,
                                adapter_name=ADAPTER_NAME,
                                )
            self.model.set_adapters([ADAPTER_NAME], adapter_weights=[0.2])
        else:
            self.enhancer = None

    def prompt_process(self, prompt, negative_prompt):
        if self.enhancer is not None:
            enhanced_prompt = self.enhancer(prompt, self.style)
        else:
            enhanced_prompt = prompt
        
        if self.prompt_post:
            return self.prompt_embedding(enhanced_prompt, negative_prompt)
        else:
            return enhanced_prompt

    # embedding the prompt
    def prompt_embedding(self, prompt, negative_prompt):
        prompt_set = get_weighted_text_embeddings_sd3(self.model, prompt = prompt, neg_prompt = negative_prompt)

        return prompt_set
    
    def get_save_path(self, output_path):
        folder_path = os.path.join(output_path, self.model_name+ "_output")
        is_folder(folder_path)
        self.save_path = folder_path
        return folder_path
    
    # Load data from Json
    def load_data(self, prompt_path, prompt_index):
        info_list = list()
        for root, _, files in os.walk(prompt_path):
            for file in files:
                info_path = os.path.join(root, file)
                info_list.append(info_path)

        info_list.sort()
        index_start = info_list.index(prompt_index + '.json')
        self.data_sets = info_list[:index_start]

        return info_list
    
    def load_json(self, prompt_json):
        with open(prompt_json, 'r') as f:
            data = json.load(f)
        return data
    
    def get_output_path(self, prompt_json):
        filename = os.path.splitext(os.path.basename(prompt_json))[0]
        output_path = os.path.join(self.save_path, filename)
        return output_path
    
    # loading image as input
    def load_image(self, path):
        return Image.open(path).convert("RGB")

    def conduct(self, data):
        raise NotImplementedError("Subclasses should implement this!")
    
    def get_result(self, data):
        raise NotImplementedError("Subclasses should implement this!")