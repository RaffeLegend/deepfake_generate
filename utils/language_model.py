import os
import torch
import math
from utils.utils import save_image, is_folder
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel
import json

import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode

from PIL import Image
import base64
from io import BytesIO

from globals.prompt import PROMPT_GENERATE_DESCRIPTION
from globals.define import IMAGENET_MEAN, IMAGENET_STD

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
        self.data_name = None
        self.save_size = None

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
    def save_json(self, index):
        index = str(index).zfill(4)
        image_info_file_name = f"{self.data_name}_{index}.json"
        json_path = os.path.join(self.save_path, image_info_file_name)
        with open(json_path, "w") as f:
            json.dump(json_path, f)
        return

    # get the input image path
    def get_images_path(self, set_path, save_size):

        self.save_size = save_size
    
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
                file_paths.append(file_info)
                index += 1

                if index % self.save_size == 0 and index != 0:
                    self.save_json(index_file)
                    index_file += 1
                    file_info = dict()

            self.save_json(index_file)
        return
      
    # Load data from Json
    def load_data(self):
        info_dict = ()
        for root, _, files in os.walk(self.save_path):
            for file in files:
                info_path = os.path.join(root, file)
                info_dict.append(info_path)

        return info_dict

    def init_model(self, data):
        raise NotImplementedError("Subclasses should implement this!")
    
    def get_result(self, data):
        raise NotImplementedError("Subclasses should implement this!")
    

# define model Stable Language Model 2 12B
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
 
# define model InternVL 2
class InternVL2(StableLanguageModel):
    def __init__(self, model_name):
        super().__init__()
        self.prompt_set = None
        self.model_name = model_name
        self.token_path = "OpenGVLab/InternVL2-8B"
        self.model_path = "OpenGVLab/InternVL2-8B"
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

    def split_model(self):
        device_map = {}
        world_size = torch.cuda.device_count()
        num_layers = {'Mini-InternVL-2B-V1-5': 24, 'Mini-InternVL-4B-V1-5': 32, 'InternVL-Chat-V1-5': 48}[self.model_name]
        # Since the first GPU will be used for ViT, treat it as half a GPU.
        num_layers_per_gpu = math.ceil(num_layers / (world_size - 0.5))
        num_layers_per_gpu = [num_layers_per_gpu] * world_size
        num_layers_per_gpu[0] = math.ceil(num_layers_per_gpu[0] * 0.5)
        layer_cnt = 0
        for i, num_layer in enumerate(num_layers_per_gpu):
            for j in range(num_layer):
                device_map[f'language_model.model.layers.{layer_cnt}'] = i
                layer_cnt += 1
        device_map['vision_model'] = 0
        device_map['mlp1'] = 0
        device_map['language_model.model.tok_embeddings'] = 0
        device_map['language_model.model.embed_tokens'] = 0
        device_map['language_model.output'] = 0
        device_map['language_model.model.norm'] = 0
        device_map['language_model.lm_head'] = 0
        device_map[f'language_model.model.layers.{num_layers - 1}'] = 0

        self.device_map = device_map
        return

    def init_model(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.token_path, trust_remote_code=True)
        # self.split_model()
        self.model = AutoModel.from_pretrained(
                            self.model_path,
                            torch_dtype=self.torch_dtype,
                            low_cpu_mem_usage=True,
                            trust_remote_code=True,
                            # device_map=self.device_map
                            ).eval()
        self.model.cuda()

    def build_transform(self):
        transform = T.Compose([
                            T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
                            T.Resize((self.image_size, self.image_size), interpolation=InterpolationMode.BICUBIC),
                            T.ToTensor(),
                            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
                            ])
        return transform
    
    def find_closest_aspect_ratio(self, aspect_ratio, target_ratios, width, height):
        best_ratio_diff = float('inf')
        best_ratio = (1, 1)
        area = width * height
        for ratio in target_ratios:
            target_aspect_ratio = ratio[0] / ratio[1]
            ratio_diff = abs(aspect_ratio - target_aspect_ratio)
            if ratio_diff < best_ratio_diff:
                best_ratio_diff = ratio_diff
                best_ratio = ratio
            elif ratio_diff == best_ratio_diff:
                if area > 0.5 * self.image_size * self.image_size * ratio[0] * ratio[1]:
                    best_ratio = ratio
        return best_ratio
    
    def dynamic_preprocess(self, image, min_num=1, max_num=6, use_thumbnail=False):
        image_size = self.image_size
        orig_width, orig_height = image.size
        aspect_ratio = orig_width / orig_height

        # calculate the existing image aspect ratio
        target_ratios = set(
            (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
            i * j <= max_num and i * j >= min_num)
        target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

        # find the closest aspect ratio to the target
        target_aspect_ratio = self.find_closest_aspect_ratio(
            aspect_ratio, target_ratios, orig_width, orig_height)

        # calculate the target width and height
        target_width = image_size * target_aspect_ratio[0]
        target_height = image_size * target_aspect_ratio[1]
        blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

        # resize the image
        resized_img = image.resize((target_width, target_height))
        processed_images = []
        for i in range(blocks):
            box = (
                (i % (target_width // image_size)) * image_size,
                (i // (target_width // image_size)) * image_size,
                ((i % (target_width // image_size)) + 1) * image_size,
                ((i // (target_width // image_size)) + 1) * image_size
            )
            # split the image
            split_img = resized_img.crop(box)
            processed_images.append(split_img)
        assert len(processed_images) == blocks
        if use_thumbnail and len(processed_images) != 1:
            thumbnail_img = image.resize((image_size, image_size))
            processed_images.append(thumbnail_img)
        return processed_images
    
    def load_image(self, image_file, max_num=6):
        image = Image.open(image_file).convert('RGB')
        transform = self.build_transform()
        images = self.dynamic_preprocess(image, use_thumbnail=True, max_num=max_num)
        pixel_values = [transform(image) for image in images]
        pixel_values = torch.stack(pixel_values)
        return pixel_values
    
    def generate_description(self, pixel_values):
        # single-image single-round conversation
        question = '<image>\n' + self.prompt
        response = self.model.chat(self.tokenizer, pixel_values, question, self.generation_config)
        print(f'Assistant: {response}')
        return response

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

        return 

 # model factory
class LMModelFactory:
    @staticmethod
    def get_model(model_name):
        if model_name == "SL2_12B":
            return StableLanguageModel2_12B(model_name=model_name)
        elif model_name == "internVL2":
            return InternVL2(model_name=model_name)
        else:
            raise ValueError(f"Unknown model name: {model_name}")