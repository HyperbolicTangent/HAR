#!/bin/bash -l

# Slurm parameters
#SBATCH --job-name=tuning
#SBATCH --output=tuning-%j.%N.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --time=2-00:00:00
#SBATCH --mem=16G
#SBATCH --gpus=1
#SBATCH --qos=batch

# Activate everything you need
module load cuda/10.1
pyenv activate venv
# Run your python code
python3 tune.py