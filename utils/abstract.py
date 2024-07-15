from utils.utils import save_image, is_folder
import torch
from diffusers import AutoPipelineForText2Image, StableDiffusion3Pipeline, \
                      StableCascadeDecoderPipeline, StableCascadePriorPipeline, \
                      Kandinsky3Pipeline, DiffusionPipeline
                      


# define abstract class
class StableDiffusion:
    def __init__(self) -> None:
        self.prompt_set = None
        self.height = None
        self.width  = None
        self.model_name  = None
        self.model_path = None
    
    def get_save_path(self):
        folder_name = "./" + self.model_name+ "_output"
        is_folder(folder_name)
        return folder_name
    
    def load_data(self, prompt_path):
        prompt_set = dict()
        prompt = ""
        with open(prompt_path, 'r') as f:
            for line in f:
                if line != "++++++++\n":
                    prompt += line
                else:
                    prompt_set.append("create image by the description: " + prompt)
                    prompt = ""
        print(f"Collect {len(prompt_set)} prompts to generate images")
        self.prompt_set = prompt_set

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
        self.model = self.init_model()
        self.save_path = self.get_save_path()

    def init_model(self):
        pipe = AutoPipelineForText2Image.from_pretrained(self.model_path, torch_dtype=self.torch_dtype, variant=self.variant)
        pipe.to("cuda")
        return pipe

    def inference(self):
        for prompt in self.prompt_set:
            image = self.model(prompt=prompt, num_inference_steps=3, guidance_scale=0.3).images[0]
            save_image(image, self.save_path, index)
            index += 1
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
        self.model = self.init_model()
        self.save_path = self.get_save_path()

    def init_model(self):
        pipe = AutoPipelineForText2Image.from_pretrained(self.model_path, torch_dtype=self.torch_dtype, variant=self.variant)
        pipe.to("cuda")
        return pipe

    def inference(self):
        for prompt in self.prompt_set:
            image = self.model(prompt=prompt, num_inference_steps=3, guidance_scale=0.3).images[0]
            save_image(image, self.save_path, index)
            index += 1
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
        self.model = self.init_model()
        self.save_path = self.get_save_path()

    def init_model(self):
        pipe = StableDiffusion3Pipeline.from_pretrained(self.model_path, torch_dtype=self.torch_dtype)
        pipe.to("cuda")
        return pipe

    def inference(self):
        for prompt in self.prompt_set:
            image = pipe(prompt=prompt,
                         negative_prompt="",
                         num_inference_steps=28,
                         guidance_scale=7.0,
                         ).images[0]
            save_image(image, self.save_path, index)
            index += 1
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
        self.model = self.init_model()
        self.save_path = self.get_save_path()
        self.negative_prompt = ""

    def init_model(self):
        self.prior = StableCascadePriorPipeline.from_pretrained(self.model_prior_path, variant=self.variant, torch_dtype=self.torch_dtype)
        self.decoder = StableCascadeDecoderPipeline.from_pretrained(self.model_decoder_path, variant=self.variant, torch_dtype=self.torch_dtype)

        # prior.enable_model_cpu_offload()
        # decoder.enable_model_cpu_offload()

    def inference(self):
        for prompt in self.prompt_set:
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
            index += 1
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
        self.model = self.init_model()
        self.save_path = self.get_save_path()
        self.negative_prompt = ""

    def init_model(self):
        self.pipeline = Kandinsky3Pipeline.from_pretrained(self.model_path, variant=self.variant, torch_dtype=self.torch_dtype)
        self.pipeline.enable_model_cpu_offload()

    def inference(self):
        for prompt in self.prompt_set:
            image = self.pipeline(prompt).images[0]
            save_image(image, self.save_path, index)
            index += 1
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
        self.model = self.init_model()
        self.save_path = self.get_save_path()
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
        for prompt in self.prompt_set:
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
            index += 1
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
        self.model = self.init_model()
        self.save_path = self.get_save_path()
        self.negative_prompt = ""

    def init_model(self):
        self.pipe = DiffusionPipeline.from_pretrained(
                                        self.model_path,
                                        torch_dtype=self.torch_dtype,
                                        variant=self.variant,
                                        ).to("cuda")

    def inference(self):
        for prompt in self.prompt_set:
            image = self.pipe(prompt=prompt, num_inference_steps=50, guidance_scale=3).images[0]
            save_image(image, self.save_path, index)
            index += 1
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
        else:
            raise ValueError(f"Unknown model name: {model_name}")
