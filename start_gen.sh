#!/bin/bash

#SBATCH -p gpulowbig -N 1 -n 16
#SBATCH -J enhancer
#SBATCH -o sdxl_log_image.out
#SBATCH -e sdxl_image.err
#SBATCH --gres=gpu:1
#SBATCH -t 24:00:00

# source activate sdxl

# define the path of the output images
# OUTPUT="output_images"

# make sure the folder of output exist
# mkdir -p $OUTPUT_DIR

# execute shell
# --model sdxl_turbo sdxl sd3_medium sd_cascade kandinsky3 sdxl_refiner playground
python run_image_generation.py --model flux  --prompt prompt_folder/  --output_path output_folder/

