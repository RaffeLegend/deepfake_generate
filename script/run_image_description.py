import os
import sys
import argparse

from models.language_model.model_factory import LMModelFactory
from utils.exception import ModelExecuteError
from utils.tools import is_folder

# load args
def parse_args():
    parser = argparse.ArgumentParser(description="Model Training and Evaluation Framework")
    parser.add_argument("--models", nargs='+', type=str, required=True, 
                        help="choose model from [SL2_12B]")
    # parser.add_argument("--prompt",  type=str, default="prompt.json", help="prompt path")
    parser.add_argument("--output_path",  type=str, default="./output", help="output path")
    parser.add_argument("--image_path",  type=str, default="", help="input image path")
    parser.add_argument("--data_name", type=str, default="used", help="name of dataset")
    parser.add_argument("--save_batch_size", type=int, default=1000, help="save size of image info")
    parser.add_argument("--image_format", type=str, default="jpg", help="input image format")

    return parser.parse_args()

if __name__ == "__main__":

    args = parse_args()
    model_list  = args.models
    image_path  = args.image_path
    output_path = args.output_path
    data_name   = args.data_name
    save_size   = args.save_batch_size
    image_format = args.image_format

    is_folder(output_path)

    # model selection
    for model in model_list:
        print(f"running model '{model}' ....")
        try:
            model = LMModelFactory.get_model(model)

            # running pipeline to generate
            model.get_save_path(output_path, data_name)
            model.get_images_path(image_path, save_size, image_format)
            model.init_model()
            model.inference()
        except ModelExecuteError as e:
            print(e)
            continue
    
    print("Finished!")