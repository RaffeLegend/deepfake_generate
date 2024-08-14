import os
import argparse
from models.image_model.model_factory import ModelFactory
from utils.exception import ModelExecuteError
from utils.tools import is_folder

# load args
def parse_args():
    parser = argparse.ArgumentParser(description="Model Training and Evaluation Framework")
    parser.add_argument("--models", nargs='+', type=str, required=True, 
                        help="choose model from [sdxl_turbo, sdxl, sd3_medium, sd_cascade, kandinsky3, sdxl_refiner, playground]")
    # parser.add_argument("--model", type=str, default="sdxl-turbo", help="model list")
    parser.add_argument("--prompt",  type=str, default="./output/internVL2_output", help="prompt path")
    parser.add_argument("--output_path",  type=str, default="./output", help="output path")
    parser.add_argument("--image_path",  type=str, default="", help="input image path")
    parser.add_argument("--prompt_index", type=str, default=None, help="the index of starting prompt file")
    return parser.parse_args()

if __name__ == "__main__":

    args = parse_args()
    model_list  = args.models
    prompt_path = args.prompt
    output_path = args.output_path
    prompt_index = args.prompt_index

    is_folder(output_path)

    # model selection
    for model in model_list:
        print(f"running model '{model}' ....")
        try:
            model = ModelFactory.get_model(model)

            # running pipeline to generate
            model.get_save_path(output_path)
            model.load_data(prompt_path, prompt_index)
            model.init_model()
            model.inference()
        except ModelExecuteError as e:
            print(e)
            continue
    
    print("Finished!")