import os
import argparse
from utils.language_model import LMModelFactory
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
    parser.add_argument("--data_name", type=str, default="used", help="name of dataset")
    parser.add_argument("--save_batch_size", type=int, default=1000, help="save size of image info")

    return parser.parse_args()

if __name__ == "__main__":

    args = parse_args()
    model_list  = args.models
    image_path  = args.image_path
    output_path = args.output_path
    prompt_path = args.prompt
    data_name   = args.data_name
    save_size   = args.save_batch_size

    is_folder(output_path)

    # model selection
    for model in model_list:
        print(f"running model '{model}' ....")
        try:
            model = LMModelFactory.get_model(model)

            # running pipeline to generate
            model.get_save_path(output_path, prompt_path, data_name)
            model.get_images_path(image_path, save_size)
            model.init_model()
            model.inference()
        except ModelExecuteError as e:
            print(e)
            continue
    
    print("Finished!")