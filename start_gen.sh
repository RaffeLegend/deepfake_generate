#!/bin/bash

#SBATCH -p gpulowmed -N 1 -n 16
#SBATCH -J flux
#SBATCH -o sdxl_log.out
#SBATCH -e sdxl_error.err
#SBATCH --gres=gpu:1
#SBATCH -t 24:00:00

# source activate sdxl

# define the path of the output images
# OUTPUT="output_images"

# make sure the folder of output exist
# mkdir -p $OUTPUT_DIR

# execute shell
#python run_sd.py --model sdxl_turbo sdxl sd3_medium sd_cascade kandinsky3 sdxl_refiner playground --prompt prompt.txt
python run_image_generation.py --model flux_turbo  --prompt /mnt/data2/users/chengyh1/generated_results/prompt_results/coco_result  --prompt_index coco_0000  --output_path /mnt/data2/users/hilight/generated_results/dataset_collect/
