# model factory
class LMModelFactory:
    @staticmethod
    def get_model(model_name):
        if model_name == "SL2_12B":
            from models.language_model.stable_language_model import StableLanguageModel2_12B
            return StableLanguageModel2_12B(model_name=model_name)
        elif model_name == "internVL2":
            from models.language_model.internvl import InternVL2
            return InternVL2(model_name=model_name)
        elif model_name == "mini_cpm":
            from models.language_model.mini_cpm import MiniCPM
            return MiniCPM(model_name=model_name)
        else:
            raise ValueError(f"Unknown model name: {model_name}")