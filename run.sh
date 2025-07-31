#!/bin/bash
#SBATCH --job-name=nlp_project_job
#SBATCH --partition=gpu           # The GPU partition name
#SBATCH --gres=gpu:v100-sxm2:4    # Request 1 GPU
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=01:00:00           # HH:MM:SS
module load anaconda3/2022.05 cuda/12.1
export WANDB_API_KEY=a71e58ca60218e22883570553abe604279dd53cf
conda activate /home/dhopate.r/miniconda3/envs/nlp_project_venv
pip install -r requirements.txt
wandb login
python3 ./PyTorch_Files/encoder_run.py