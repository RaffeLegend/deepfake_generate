import os
import json
from PIL import Image

from deepfake_detection.utils.tools import is_folder
from deepfake_detection.globals.define import *


# define abstract class
class EditingModel:
    def __init__(self) -> None:
        self.prompt_set = None
        self.height = None
        self.width  = None
        self.model_name  = None
        self.model_path = None
        self.save_path = None
        self.model = None
    
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

        for index, path in enumerate(info_list):
            if prompt_index in path:
                self.data_sets = info_list[index:]
                break

        return info_list
    
    def load_json(self, prompt_json):
        with open(prompt_json, 'r') as f:
            data = json.load(f)
        return data
    
    def get_output_path(self, prompt_json):
        filename = os.path.splitext(os.path.basename(prompt_json))[0]
        output_path = os.path.join(self.save_path, filename)
        is_folder(output_path)
        return output_path
    
    # loading image as input
    def load_image(self, path):
        return Image.open(path).convert("RGB")

    def conduct(self, data):
        raise NotImplementedError("Subclasses should implement this!")
    
    def get_result(self, data):
        raise NotImplementedError("Subclasses should implement this!")