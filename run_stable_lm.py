import os
import argparse
from utils.abstract import LMModelFactory
from utils.exception import ModelExecuteError
from utils.utils import is_folder

# load args
def parse_args():
    parser = argparse.ArgumentParser(description="Model Training and Evaluation Framework")
    parser.add_argument("--models", nargs='+', type=str, required=True, 
                        help="choose model from [SL2_12B]")
    # parser.add_argument("--model", type=str, default="sdxl-turbo", help="model list")
    parser.add_argument("--prompt",  type=str, default="prompt.json", help="prompt path")
    parser.add_argument("--output_path",  type=str, default="./output", help="output path")
    parser.add_argument("--image_path",  type=str, default="", help="input image path")
    parser.add_argument("--image_info_path", type=str, default="image_info.json")
    return parser.parse_args()

if __name__ == "__main__":

    args = parse_args()
    model_list  = args.models
    image_path  = args.image_path
    output_path = args.output_path
    prompt_path = args.prompt
    image_info  = args.image_info_path

    is_folder(output_path)

    # model selection
    for model in model_list:
        print(f"running model '{model}' ....")
        try:
            model = LMModelFactory.get_model(model)

            # running pipeline to generate
            model.get_save_path(output_path, prompt_path, image_info)
            model.get_image_path(image_path)
            model.init_model()
            model.inference()
        except ModelExecuteError as e:
            print(e)
            continue
    
    print("Finished!")