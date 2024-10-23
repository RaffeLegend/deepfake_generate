import torch
import os
from PIL import Image
from utils.tools import save_image
from diffusers import AutoPipelineForText2Image, StableDiffusion3Pipeline,      \
                      StableCascadeDecoderPipeline, StableCascadePriorPipeline, \
                      DiffusionPipeline, StableDiffusionImg2ImgPipeline,        \
                      BitsAndBytesConfig, SD3Transformer2DModel                 
from transformers import T5EncoderModel
                      
from prompts.prompt import NEGATIVE_PROMPT
from models.image_model.abstract import DiffusionModel

# define model sdxl-turbo
class StableDiffusionXLTurbo(DiffusionModel):
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
            output_path = self.get_output_path(patch_data)
            for data_info in json_data:
                index  = data_info["index"]
                prompt = data_info["prompt"]
                image = self.model(prompt=prompt, num_inference_steps=3, guidance_scale=0.3).images[0]
                save_image(image, output_path, index)
            return 

# define model sdxl
class StableDiffusionXL(DiffusionModel):
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
            output_path = self.get_output_path(patch_data)
            for data_info in json_data:
                index  = data_info["index"]
                prompt = data_info["prompt"]
                image = self.model(prompt=prompt, num_inference_steps=40, guidance_scale=0.3).images[0]
                save_image(image, output_path, index)
        return 
    
# define model sd3-medium
class StableDiffusion3Medium(DiffusionModel):
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
            output_path = self.get_output_path(patch_data)
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
                save_image(image, output_path, index)
        return 

# define model sd cascade
class StableDiffusionCascade(DiffusionModel):
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
            output_path = self.get_output_path(patch_data)
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

                save_image(decoder_output, output_path, index)
        return 

# define model sd cascade
class StableDiffusionXLwithRefiner(DiffusionModel):
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
            output_path = self.get_output_path(patch_data)
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
                save_image(image, output_path, index)
        return 
    
# define Stable Diffusion 2 for image to image task
class Image2ImageSD2(DiffusionModel):
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
            output_path = self.get_output_path(patch_data)
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
                save_image(image, output_path, index)
        return
    
#define model SD3.5 large
class StableDiffusion3_5Large(DiffusionModel):
    def __init__(self, model_name):
        super().__init__()
        self.prompt_set = None
        self.model_name = model_name
        self.model_path = "stabilityai/stable-diffusion-3.5-large"
        self.torch_dtype = torch.bfloat16
        self.variant = "fp16"
        self.enhancer = None
        self.prompt_post = True
        self.quantized = True
        # self.custom_pipeline="lpw_stable_diffusion"
        # self.save_path = self.get_save_path()

    def init_model(self):
        if self.quantized:
            nf4_config = BitsAndBytesConfig(
                                    load_in_4bit=True,
                                    bnb_4bit_quant_type="nf4",
                                    bnb_4bit_compute_dtype=self.torch_dtype
                                    )
            model_nf4 = SD3Transformer2DModel.from_pretrained(
                                    self.model_path,
                                    subfolder="transformer",
                                    quantization_config=nf4_config,
                                    torch_dtype=self.torch_dtype
                                    )

            self.model = StableDiffusion3Pipeline.from_pretrained(
                                    self.model_path, 
                                    transformer=model_nf4,
                                    torch_dtype=self.torch_dtype
                                    )
            self.model.enable_model_cpu_offload()

        else:
            self.model = StableDiffusion3Pipeline.from_pretrained(
                                    self.model_path, 
                                    torch_dtype=self.torch_dtype,
                                    )
            self.model.to("cuda")

    def inference(self):
        for patch_data in self.data_sets:
            json_data = self.load_json(patch_data)
            output_path = self.get_output_path(patch_data)
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
                             num_inference_steps=28,
                             guidance_scale=3.5,
                             ).images[0]
                save_image(image, output_path, index)
        return
    

#define model SD3.5 large turbo
class StableDiffusion3_5LargeTurbo(DiffusionModel):
    def __init__(self, model_name):
        super().__init__()
        self.prompt_set = None
        self.model_name = model_name
        self.model_path = "stabilityai/stable-diffusion-3.5-large-turbo"
        self.torch_dtype = torch.bfloat16
        self.variant = "fp16"
        self.enhancer = None
        self.prompt_post = True
        self.quantized = True
        # self.custom_pipeline="lpw_stable_diffusion"
        # self.save_path = self.get_save_path()

    def init_model(self):
        if self.quantized:
            nf4_config = BitsAndBytesConfig(
                                    load_in_4bit=True,
                                    bnb_4bit_quant_type="nf4",
                                    bnb_4bit_compute_dtype=self.torch_dtype
                                    )
            model_nf4 = SD3Transformer2DModel.from_pretrained(
                                    self.model_path,
                                    subfolder="transformer",
                                    quantization_config=nf4_config,
                                    torch_dtype=self.torch_dtype
                                    )

            t5_nf4 = T5EncoderModel.from_pretrained("diffusers/t5-nf4", torch_dtype=self.torch_dtype)

            self.model = StableDiffusion3Pipeline.from_pretrained(
                                    self.model_path, 
                                    transformer=model_nf4,
                                    text_encoder_3=t5_nf4,
                                    torch_dtype=self.torch_dtype
                                    )
            self.model.enable_model_cpu_offload()
        else:
            self.model = StableDiffusion3Pipeline.from_pretrained(
                                    self.model_path, 
                                    torch_dtype=self.torch_dtype,
                                    )
            self.model.to("cuda")

    def inference(self):
        for patch_data in self.data_sets:
            json_data = self.load_json(patch_data)
            output_path = self.get_output_path(patch_data)
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
                             num_inference_steps=4,
                             guidance_scale=0,
                             ).images[0]
                save_image(image, output_path, index)
        return