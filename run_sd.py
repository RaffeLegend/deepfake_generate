import os
import argparse
from utils.abstract import ModelFactory

# load args
def parse_args():
    parser = argparse.ArgumentParser(description="Model Training and Evaluation Framework")
    parser.add_argument("--model", type=str, default="sdxl-turbo", help="model list")
    parser.add_argument("--prompt",  type=str, default="prompt.txt", help="model list")
    parser.add_argument("--output_path",  type=str, default="", help="model list")
    parser.add_argument("--image_path",  type=str, default="", help="model list")
    return parser.parse_args()

if __name__ == "__main__":

    args = parse_args()
    model = args.model
    prompt_path = args.prompt

    # model selection
    model = ModelFactory.get_model(model)

    # running pipeline to generate
    model.load_data(prompt_path)
    model.init_model()
    model.inference()
    
    print("Finished!")