#!/bin/bash

#SBATCH --job-name=gpt2-debug
#SBATCH --output=logs/gpt2-debug_%j.log

#SBATCH --partition=gpu_devel
#SBATCH --gpus=1

#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=06:00:00

#SBATCH --mail-user=daniel.sh.joo@gmail.com
#SBATCH --mail-type=ALL

mkdir -p logs

echo "Loading modules..."
module load PyTorch/2.1.2-foss-2022b-CUDA-12.1.1

echo "Activating virtual environment..."
source venv/bin/activate

echo "Starting Python script..."
# Just run the python script directly. DataParallel handles the GPUs.
python train_gpt.py