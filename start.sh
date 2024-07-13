#!/bin/bash

#SBATCH -p gpu -N 1 -n 16
#SBATCH -J sdxl_refiner_generate
#SBATCH -o sdxl_log.out
#SBATCH -e sdxl.err
#SBATCH --gres=gpu:2

# source activate sdxl

# define the path of the output images
# OUTPUT="output_images"

# make sure the folder of output exist
# mkdir -p $OUTPUT_DIR

# execute shell
python run_sd.py --model sdxl-turbo --prompt prompt.txt

