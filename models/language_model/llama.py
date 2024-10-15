import os
import torch
import math
import json
from PIL import Image, UnidentifiedImageError
import base64
from io import BytesIO

from transformers import AutoTokenizer, AutoModel
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode

from prompts.prompt import PROMPT_GENERATE_DESCRIPTION
from globals.define import IMAGENET_MEAN, IMAGENET_STD
from models.language_model.abstract import LanguageModel

# define model LLama
#To do

class LLama(LanguageModel):
    def __init__(self, model_name):
        super().__init__()
        self.prompt_set = None
        self.model_name = model_name
        self.token_path = "meta-llama/Llama-3.1-8B-Instruct"
        self.model_path = "meta-llama/Llama-3.1-8B-Instruct"
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
        try:
            image = Image.open(image_file).convert('RGB')
        except (UnidentifiedImageError) as e:
            print(f"Failed to load the image {image_file}: {e}")
            return None
                
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
        image = self.load_image(image_path, max_num=6)
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