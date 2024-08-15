import os
import torch
import json
from PIL import Image, UnidentifiedImageError

from transformers import AutoTokenizer

from open_flamingo import create_model_and_transforms
from huggingface_hub import hf_hub_download

from prompts.prompt import PROMPT_GENERATE_DESCRIPTION
from models.language_model.abstract import LanguageModel

# define model MiniCPM
class MiniCPM(LanguageModel):
    def __init__(self, model_name):
        super().__init__()
        self.prompt_set = None
        self.model_name = model_name
        self.token_path = "openflamingo/OpenFlamingo-4B-vitl-rpj3b"
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

        self.model, self.image_processor, self.tokenizer = create_model_and_transforms(
                            clip_vision_encoder_path="ViT-L-14",
                            clip_vision_encoder_pretrained="openai",
                            lang_encoder_path=self.token_path,
                            tokenizer_path=self.token_path,
                            cross_attn_every_n_layers=2
                            )

        # grab model checkpoint from huggingface hub
        checkpoint_path = hf_hub_download(self.model_path, "checkpoint.pt")
        self.model.load_state_dict(torch.load(checkpoint_path), strict=False)
        self.tokenizer = AutoTokenizer.from_pretrained(self.token_path, trust_remote_code=True)

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
        lang_x = self.tokenizer(self.prompt, return_tensors="pt",)
        generated_text = self.model.generate(
                                vision_x=image,
                                lang_x=lang_x["input_ids"],
                                attention_mask=lang_x["attention_mask"],
                                max_new_tokens=50,
                                )
        return self.tokenizer.decode(generated_text[0])

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