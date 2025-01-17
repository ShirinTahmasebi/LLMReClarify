#!/usr/bin/env bash
#SBATCH -A PRJECT_CODE
#SBATCH -p alvis
#SBATCH --gpus-per-node=T4:1  -t 16:00:00
#SBATCH -o preprocessing.out
#SBATCH -e preprocessing.err


python3 -m venv .genrec_venv
source .genrec_venv/bin/activate
pip3 install -r requirements.txt


srun python evaluation.py

