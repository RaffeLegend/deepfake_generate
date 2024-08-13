import os
import json

import torchvision.transforms as T

import base64
from io import BytesIO

from utils.tools import is_folder
from prompts.prompt import PROMPT_GENERATE_DESCRIPTION
from globals.define import IMAGENET_MEAN, IMAGENET_STD

# define abstract class
class LanguageModel:
    def __init__(self) -> None:
        self.prompt_set = None
        self.height = None
        self.width  = None
        self.model_name  = None
        self.model_path = None
        self.save_path = None
        self.model = None
        self.data_name = None
        self.save_size = None
        self.image_format = None

    # Encode the image at the given path to a base64 string
    def encode_image(self, image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    # Get the base64 encoded description of an image at the given path
    def get_image_des_format(self, image_path):
        encoded_image = self.encode_image(image_path)
        return encoded_image

    # set output path
    def get_save_path(self, output_path, data_name):
        self.data_name = data_name
        folder_path = os.path.join(output_path, self.model_name+ "_output")
        is_folder(folder_path)
        self.save_path = folder_path
        return
    
    # save json file
    def save_json(self, data, index):
        index = str(index).zfill(4)
        image_info_file_name = f"{self.data_name}_{index}.json"
        json_path = os.path.join(self.save_path, image_info_file_name)
        with open(json_path, "w") as f:
            json.dump(data, f)
        return

    # get the input image path
    def get_images_path(self, set_path, save_size, image_format):

        self.save_size = save_size
        self.image_format = image_format
    
        file_paths = list()
        index = 0
        index_file = 0

        for root, _, files in os.walk(set_path):
            for file in files:
                file_info = dict()
                file_info["index"]     = str(index).zfill(9)
                file_info["file_name"] = file
                file_info["file_path"] = root
                file_info["prompt"]    = ""
                file_info["text"]      = ""
                if file.split(".")[-1] == self.image_format:
                    file_paths.append(file_info)
                    index += 1

                    if index % self.save_size == 0 and index != 0:
                        self.save_json(file_paths, index_file)
                        index_file += 1
                        file_paths = list()

            self.save_json(file_paths, index_file)
        return
      
    # Load data from Json
    def load_data(self):
        info_list = list()
        for root, _, files in os.walk(self.save_path):
            for file in files:
                info_path = os.path.join(root, file)
                info_list.append(info_path)

        return info_list

    def init_model(self, data):
        raise NotImplementedError("Subclasses should implement this!")
    
    def get_result(self, data):
        raise NotImplementedError("Subclasses should implement this!")