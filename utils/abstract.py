import os
import torch
import json
from PIL import Image
from utils.utils import save_image, is_folder
from diffusers import AutoPipelineForText2Image, StableDiffusion3Pipeline, \
                      StableCascadeDecoderPipeline, StableCascadePriorPipeline, \
                      Kandinsky3Pipeline, DiffusionPipeline, \
                      StableDiffusionImg2ImgPipeline
from sd_embed.embedding_funcs import get_weighted_text_embeddings_sd3
from globals.prompt import NEGATIVE_PROMPT, PROMPT_REALISTIC_VISION_NEGATIVE
from globals.prompt_enhance import PromptEnhancer


# define abstract class
class StableDiffusion:
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
                                "stabilityai/stable-diffusion-xl-base-1.0",
                                weight_name="sd_xl_offset_example-lora_1.0.safetensors",
                                adapter_name="offset",
                                )
            self.model.set_adapters(["offset"], adapter_weights=[0.2])
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
    def load_data(self, prompt_path):
        info_list = list()
        for root, _, files in os.walk(prompt_path):
            for file in files:
                info_path = os.path.join(root, file)
                info_list.append(info_path)

        self.data_sets = info_list
        return info_list
    
    def load_json(self, prompt_json):
        with open(prompt_json, 'r') as f:
            data = json.load(f)
        return data
    
    # loading image as input
    def load_image(self, path):
        return Image.open(path).convert("RGB")

    def conduct(self, data):
        raise NotImplementedError("Subclasses should implement this!")
    
    def get_result(self, data):
        raise NotImplementedError("Subclasses should implement this!")
    

# define model sdxl-turbo
class StableDiffusionXLTurbo(StableDiffusion):
    def __init__(self, model_name):
        super().__init__()
        self.prompt_set = None
        self.model_name = model_name
        self.model_path = "stabilityai/sdxl-turbo"
        self.torch_dtype = torch.float16
        self.variant = "fp16"
        # self.save_path = self.get_save_path()

    def init_model(self):
        self.model = AutoPipelineForText2Image.from_pretrained(self.model_path, torch_dtype=self.torch_dtype, variant=self.variant)
        self.model.to("cuda")

    def inference(self):
        for patch_data in self.data_sets:
            json_data = self.load_json(patch_data)
            for data_info in json_data:
                index  = data_info["index"]
                prompt = data_info["prompt"]
                image = self.model(prompt=prompt, num_inference_steps=3, guidance_scale=0.3).images[0]
                save_image(image, self.save_path, index)
            return 

# define model sdxl
class StableDiffusionXL(StableDiffusion):
    def __init__(self, model_name):
        super().__init__()
        self.prompt_set = None
        self.model_name = model_name
        self.model_path = "stabilityai/stable-diffusion-xl-base-1.0"
        self.torch_dtype = torch.float16
        self.variant = "fp16"
        # self.save_path = self.get_save_path()

    def init_model(self):
        self.model = AutoPipelineForText2Image.from_pretrained(self.model_path, torch_dtype=self.torch_dtype, variant=self.variant)
        self.model.to("cuda")

    def inference(self):
        for patch_data in self.data_sets:
            json_data = self.load_json(patch_data)
            for data_info in json_data:
                index  = data_info["index"]
                prompt = data_info["prompt"]
                image = self.model(prompt=prompt, num_inference_steps=40, guidance_scale=0.3).images[0]
                save_image(image, self.save_path, index)
        return 
    
# define model sd3-medium
class StableDiffusion3Medium(StableDiffusion):
    def __init__(self, model_name):
        super().__init__()
        self.prompt_set = None
        self.model_name = model_name
        self.model_path = "stabilityai/stable-diffusion-3-medium-diffusers"
        self.torch_dtype = torch.float16
        self.variant = "fp16"
        self.sampler = "K_DPMPP_2S_ANCESTRAL"
        self.enhancer = None
        self.prompt_post = True
        # self.custom_pipeline="lpw_stable_diffusion"
        # self.save_path = self.get_save_path()

    def init_model(self):
        self.model = StableDiffusion3Pipeline.from_pretrained(
                                    self.model_path, 
                                    torch_dtype=self.torch_dtype,
                                    )
        self.model.to("cuda")

    def inference(self):
        for patch_data in self.data_sets:
            json_data = self.load_json(patch_data)
            for data_info in json_data:
                index  = data_info["index"]
                prompt = data_info["prompt"]
                prompt_set = self.prompt_process(prompt, NEGATIVE_PROMPT)
                prompt_embeds, prompt_neg_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds = prompt_set
                image = self.model(
                             prompt_embeds=prompt_embeds,
                             pooled_prompt_embeds=pooled_prompt_embeds,
                             negative_prompt_embeds=prompt_neg_embeds,
                             negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
                             num_inference_steps=30,
                             guidance_scale=9.0,
                             ).images[0]
                save_image(image, self.save_path, index)
        return 

# define model sd cascade
class StableDiffusionCascade(StableDiffusion):
    def __init__(self, model_name):
        super().__init__()
        self.prompt_set = None
        self.model_name = model_name
        self.model_prior_path = "stabilityai/stable-cascade-prior"
        self.model_decoder_path = "stabilityai/stable-cascade"
        self.torch_dtype = torch.float16
        self.variant = "bf16"
        # self.save_path = self.get_save_path()
        self.negative_prompt = ""

    def init_model(self):
        self.prior = StableCascadePriorPipeline.from_pretrained(self.model_prior_path, variant=self.variant, torch_dtype=self.torch_dtype)
        self.decoder = StableCascadeDecoderPipeline.from_pretrained(self.model_decoder_path, variant=self.variant, torch_dtype=self.torch_dtype)

        # prior.enable_model_cpu_offload()
        # decoder.enable_model_cpu_offload()

    def inference(self):
        for patch_data in self.data_sets:
            json_data = self.load_json(patch_data)
            for data_info in json_data:
                index  = data_info["index"]
                prompt = data_info["prompt"]
                prior_output = self.prior(
                                 prompt=prompt,
                                 height=1024,
                                 width=1024,
                                 negative_prompt=self.negative_prompt,
                                 guidance_scale=4.0,
                                 num_images_per_prompt=1,
                                 num_inference_steps=20
                                )

                decoder_output = self.decoder(
                                 image_embeddings=prior_output.image_embeddings.to(torch.float16),
                                 prompt=prompt,
                                 negative_prompt=self.negative_prompt,
                                 guidance_scale=0.0,
                                 output_type="pil",
                                 num_inference_steps=10
                                ).images[0]

                save_image(decoder_output, self.save_path, index)
        return 
    
# define model sd cascade
class Kandinsky3(StableDiffusion):
    def __init__(self, model_name):
        super().__init__()
        self.prompt_set = None
        self.model_name = model_name
        self.model_path = "kandinsky-community/kandinsky-3"
        self.torch_dtype = torch.float16
        self.variant = "fp16"
        # self.save_path = self.get_save_path()
        self.negative_prompt = ""

    def init_model(self):
        self.model = Kandinsky3Pipeline.from_pretrained(self.model_path, variant=self.variant, torch_dtype=self.torch_dtype)
        self.model.enable_model_cpu_offload()

    def inference(self):
        for patch_data in self.data_sets:
            json_data = self.load_json(patch_data)
            for data_info in json_data:
                index  = data_info["index"]
                prompt = data_info["prompt"]
                image = self.model(prompt).images[0]
                save_image(image, self.save_path, index)
        return 

# define model sd cascade
class StableDiffusionXLwithRefiner(StableDiffusion):
    def __init__(self, model_name):
        super().__init__()
        self.prompt_set = None
        self.model_name = model_name
        self.model_base_path = "stabilityai/stable-diffusion-xl-base-1.0"
        self.model_refiner_path = "stabilityai/stable-diffusion-xl-refiner-1.0"
        self.torch_dtype = torch.float16
        self.variant = "fp16"
        # self.save_path = self.get_save_path()
        self.negative_prompt = ""

    def init_model(self):
        self.base = DiffusionPipeline.from_pretrained(
                            self.model_base_path, torch_dtype=self.torch_dtype, 
                            variant=self.variant, use_safetensors=True
                            ).to("cuda")

        self.refiner = DiffusionPipeline.from_pretrained(
                            self.model_refiner_path,
                            text_encoder_2=self.base.text_encoder_2,
                            vae=self.base.vae,
                            torch_dtype=self.torch_dtype,
                            use_safetensors=True,
                            variant=self.variant,
                            ).to("cuda")

    def inference(self):
        for patch_data in self.data_sets:
            json_data = self.load_json(patch_data)
            for data_info in json_data:
                index  = data_info["index"]
                prompt = data_info["prompt"]
                image = self.base(
                            prompt=prompt,
                            num_inference_steps=40,
                            denoising_end=0.8,
                            output_type="latent",
                            ).images
                image = self.refiner(
                            prompt=prompt,
                            num_inference_steps=40,
                            denoising_start=0.8,
                            image=image,
                            ).images[0]
                save_image(image, self.save_path, index)
        return 

# define model playground
class Playground(StableDiffusion):
    def __init__(self, model_name):
        super().__init__()
        self.prompt_set = None
        self.model_name = model_name
        self.model_path = "playgroundai/playground-v2.5-1024px-aesthetic"
        self.torch_dtype = torch.float16
        self.variant = "fp16"
        # self.save_path = self.get_save_path()
        self.negative_prompt = ""

    def init_model(self):
        self.model = DiffusionPipeline.from_pretrained(
                                        self.model_path,
                                        torch_dtype=self.torch_dtype,
                                        variant=self.variant,
                                        ).to("cuda")
        
        self.set_prompt_enhancer()

    def inference(self):
        for patch_data in self.data_sets:
            json_data = self.load_json(patch_data)
            for data_info in json_data:
                index  = data_info["index"]
                prompt = data_info["prompt"]
                prompt = self.prompt_process(prompt, NEGATIVE_PROMPT)
                image = self.model(prompt=prompt, 
                                   num_inference_steps=50, 
                                   guidance_scale=3
                                   ).images[0]
                save_image(image, self.save_path, index)
        return 

# define model realistic vision 6
class RealisticVision6(StableDiffusion):
    def __init__(self, model_name):
        super().__init__()
        self.prompt_set = None
        self.model_name = model_name
        self.model_path = "SG161222/Realistic_Vision_V6.0_B1_noVAE"
        self.torch_dtype = torch.float16
        self.variant = "fp16"
        # self.custom_pipeline="lpw_stable_diffusion"
        # self.save_path = self.get_save_path()

    def init_model(self):
        self.model = DiffusionPipeline.from_pretrained(
                                    self.model_path,
                                    )
        self.model.to("cuda")
        self.set_prompt_enhancer()

    def inference(self):
        for patch_data in self.data_sets:
            json_data = self.load_json(patch_data)
            for data_info in json_data:
                index  = data_info["index"]
                prompt = data_info["prompt"]
                prompt = self.prompt_process(prompt, NEGATIVE_PROMPT)
                image = self.model(
                            prompt=prompt,
                            negative_prompt=PROMPT_REALISTIC_VISION_NEGATIVE,
                            num_inference_steps=40,
                            guidance_scale=7.0,
                            ).images[0]
                save_image(image, self.save_path, index)
        return 

# define model absolute reality
class AbsoluteReality(StableDiffusion):
    def __init__(self, model_name):
        super().__init__()
        self.prompt_set = None
        self.model_name = model_name
        self.model_path = "Lykon/AbsoluteReality"
        self.torch_dtype = torch.float16
        self.variant = "fp16"
        # self.custom_pipeline="lpw_stable_diffusion"
        # self.save_path = self.get_save_path()

    def init_model(self):
        self.model = DiffusionPipeline.from_pretrained(
                                    self.model_path,
                                    )
        self.model.to("cuda")
        self.set_prompt_enhancer()

    def inference(self):
        for patch_data in self.data_sets:
            json_data = self.load_json(patch_data)
            for data_info in json_data:
                index  = data_info["index"]
                prompt = data_info["prompt"]
                prompt = self.prompt_process(prompt, NEGATIVE_PROMPT)
                image = self.model(
                            prompt=prompt,
                            negative_prompt=PROMPT_REALISTIC_VISION_NEGATIVE,
                            num_inference_steps=40,
                            guidance_scale=7.0,
                            ).images[0]
                save_image(image, self.save_path, index)
        return 

# define Stable Diffusion 2 for image to image task
class Image2ImageSD2(StableDiffusion):
    def __init__(self, model_name):
        super().__init__()
        self.prompt_set = None
        self.model_name = model_name
        self.model_path = "runwayml/stable-diffusion-v1-5"
        self.torch_dtype = torch.float16
        self.variant = "fp16"
        # self.custom_pipeline="lpw_stable_diffusion"
        # self.save_path = self.get_save_path()

    def init_model(self):
        self.model = StableDiffusionImg2ImgPipeline.from_pretrained(
                                    self.model_path,
                                    torch_dtype=self.torch_dtype,
                                    )
        self.model.to("cuda")

    def inference(self):
        for patch_data in self.data_sets:
            json_data = self.load_json(patch_data)
            for data_info in json_data:
                index  = data_info["index"]
                prompt = data_info["prompt"]
                path   = os.path.join(data_info["file_path"], data_info["file_name"])
                input  = self.load_image(path).resize((768, 512))
                image  = self.model(
                            prompt=prompt,
                            image=input,
                            strength=0.75,
                            guidance_scale=7.5,
                            ).images[0]
                save_image(image, self.save_path, index)
        return 
    
# define model Juggernaut XL
class JuggernautXL(StableDiffusion):
    def __init__(self, model_name):
        super().__init__()
        self.prompt_set = None
        self.model_name = model_name
        self.model_path = "RunDiffusion/Juggernaut-XI-v11"
        self.torch_dtype = torch.float16
        self.variant = "fp16"
        # self.custom_pipeline="lpw_stable_diffusion"
        # self.save_path = self.get_save_path()

    def init_model(self):
        self.model = DiffusionPipeline.from_pretrained(
                                    self.model_path,
                                    )
        self.model.to("cuda")
        self.set_prompt_enhancer()

    def inference(self):
        for patch_data in self.data_sets:
            json_data = self.load_json(patch_data)
            for data_info in json_data:
                index  = data_info["index"]
                prompt = data_info["prompt"]
                prompt = self.prompt_process(prompt, NEGATIVE_PROMPT)
                image = self.model(
                            prompt=prompt,
                            negative_prompt=PROMPT_REALISTIC_VISION_NEGATIVE,
                            num_inference_steps=40,
                            guidance_scale=7.0,
                            ).images[0]
                save_image(image, self.save_path, index)
        return

# model factory
class ModelFactory:
    @staticmethod
    def get_model(model_name):
        if model_name == "sdxl_turbo":
            return StableDiffusionXLTurbo(model_name=model_name)
        elif model_name == "sdxl":
            return StableDiffusionXL(model_name=model_name)
        elif model_name == "sd3_medium":
            return StableDiffusion3Medium(model_name=model_name)
        elif model_name == "sd_cascade":
            return StableDiffusionCascade(model_name=model_name)
        elif model_name == "kandinsky3":
            return Kandinsky3(model_name=model_name)
        elif model_name == "sdxl_refiner":
            return StableDiffusionXLwithRefiner(model_name=model_name)
        elif model_name == "playground":
            return Playground(model_name=model_name)
        elif model_name == "realistic_vision":
            return RealisticVision6(model_name=model_name)
        elif model_name == "absolute_reality":
            return AbsoluteReality(model_name=model_name)
        elif model_name == "sd2_i2i":
            return Image2ImageSD2(model_name=model_name)
        elif model_name == "juggernautxl":
            return JuggernautXL(model_name=model_name)
        else:
            raise ValueError(f"Unknown model name: {model_name}")
