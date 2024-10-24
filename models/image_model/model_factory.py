# model factory
class ModelFactory:
    @staticmethod
    def get_model(model_name):
        if model_name == "sdxl_turbo":
            from models.image_model.stable_diffusion import StableDiffusionXLTurbo
            return StableDiffusionXLTurbo(model_name=model_name)
        elif model_name == "sdxl":
            from models.image_model.stable_diffusion import StableDiffusionXL
            return StableDiffusionXL(model_name=model_name)
        elif model_name == "sd3_medium":
            from models.image_model.stable_diffusion import StableDiffusion3Medium
            return StableDiffusion3Medium(model_name=model_name)
        elif model_name == "sd3.5_large":
            from models.image_model.stable_diffusion import StableDiffusion3_5Large
            return StableDiffusion3_5Large(model_name=model_name)
        elif model_name == "sd3.5_large_turbo":
            from models.image_model.stable_diffusion import StableDiffusion3_5LargeTurbo
            return StableDiffusion3_5LargeTurbo(model_name=model_name)
        elif model_name == "sd_cascade":
            from models.image_model.stable_diffusion import StableDiffusionCascade
            return StableDiffusionCascade(model_name=model_name)
        elif model_name == "sdxl_lightning":
            from models.image_model.sdxl_lightning import SDXLLightning
            return SDXLLightning(model_name=model_name)
        elif model_name == "kandinsky3":
            from models.image_model.kandinsky import Kandinsky3
            return Kandinsky3(model_name=model_name)
        elif model_name == "sdxl_refiner":
            from models.image_model.stable_diffusion import StableDiffusionXLwithRefiner
            return StableDiffusionXLwithRefiner(model_name=model_name)
        elif model_name == "playground":
            from models.image_model.playground import Playground
            return Playground(model_name=model_name)
        elif model_name == "realistic_vision":
            from models.image_model.realistic_vision import RealisticVision6
            return RealisticVision6(model_name=model_name)
        elif model_name == "realism_riiwa":
            from models.image_model.realism_riiwa import RealismRiiwa
            return RealismRiiwa(model_name=model_name)
        elif model_name == "absolute_reality":
            from models.image_model.absolute_reality import AbsoluteReality
            return AbsoluteReality(model_name=model_name)
        elif model_name == "sd2_i2i":
            from models.image_model.stable_diffusion import Image2ImageSD2
            return Image2ImageSD2(model_name=model_name)
        elif model_name == "juggernautxl":
            from models.image_model.juggernaut import JuggernautXL
            return JuggernautXL(model_name=model_name)
        elif model_name == "flux":
            from models.image_model.flux import Flux
            return Flux(model_name=model_name)
        elif model_name == "flux_quantized":
            from models.image_model.flux import FluxQuantized
            return FluxQuantized(model_name=model_name)
        elif model_name == "flux_turbo":
            from models.image_model.flux import FluxTurbo
            return FluxTurbo(model_name=model_name)
        elif model_name == "open_flux":
            from models.image_model.flux import OpenFlux
            return OpenFlux(model_name=model_name)
        elif model_name == "flux_ghibsky":
            from models.image_model.flux import FluxGhibsky
            return FluxGhibsky(model_name=model_name)
        elif model_name == "flux_realism":
            from models.image_model.flux import FluxRealism
            return FluxRealism(model_name=model_name)
        elif model_name == "itercomp":
            from models.image_model.itercomp import IterComp
            return IterComp(model_name=model_name)
        elif model_name == "flux_antiblur":
            from models.image_model.flux import FluxAntiBlur
            return FluxAntiBlur(model_name=model_name)
        else:
            raise ValueError(f"Unknown model name: {model_name}")
