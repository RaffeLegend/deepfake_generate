#!/bin/bash

#SBATCH -p gpulowbig -N 1 -n 16
#SBATCH -J sdxl_refiner_generate
#SBATCH -o sdxl_log.out
#SBATCH -e sdxl.err
#SBATCH --gres=gpu:1
#SBATCH -t 24:00:00

# source activate sdxl

# define the path of the output images
# OUTPUT="output_images"

# make sure the folder of output exist
# mkdir -p $OUTPUT_DIR

# execute shell
python run_image_description.py --model internVL2 --data_name flickr --image_path image_path/ --output_path output_path/