#!/usr/bin/env bash
#SBATCH -A PRJECT_CODE
#SBATCH -p alvis
#SBATCH --gpus-per-node=A100:1  -t 15:00:00
#SBATCH --job-name=preprocessing
#SBATCH --output=preprocessing_%j.out
#SBATCH --error=preprocessing_%j.err
#SBATCH --ntasks=10

python3 -m venv .recbench_env
source .recbench_env/bin/activate
pip3 install -r requirements.txt

# srun python ./src/inference.py --model_name "LLMRec" --dataset_name "MOVIE_LENS" --start_index 0 --end_index 10
# srun python ./src/finetune.py --model_name "LLMRec" --dataset_name "MOVIE_LENS"

srun --exclusive --ntasks=1 python train_ranker.py --llm_gt_pos -1 &
srun --exclusive --ntasks=1 python train_ranker.py --llm_gt_pos 19 &
srun --exclusive --ntasks=1 python train_ranker.py --llm_gt_pos 18 &
srun --exclusive --ntasks=1 python train_ranker.py --llm_gt_pos 17 &
srun --exclusive --ntasks=1 python train_ranker.py --llm_gt_pos 16 &
srun --exclusive --ntasks=1 python train_ranker.py --llm_gt_pos 15 &
srun --exclusive --ntasks=1 python train_ranker.py --llm_gt_pos 14 &
srun --exclusive --ntasks=1 python train_ranker.py --llm_gt_pos 13 &
srun --exclusive --ntasks=1 python train_ranker.py --llm_gt_pos 12 &
srun --exclusive --ntasks=1 python train_ranker.py --llm_gt_pos 11 &
srun --exclusive --ntasks=1 python train_ranker.py --llm_gt_pos 10 &
srun --exclusive --ntasks=1 python train_ranker.py --llm_gt_pos 9 &
srun --exclusive --ntasks=1 python train_ranker.py --llm_gt_pos 8 &
srun --exclusive --ntasks=1 python train_ranker.py --llm_gt_pos 7 &
srun --exclusive --ntasks=1 python train_ranker.py --llm_gt_pos 6 &
srun --exclusive --ntasks=1 python train_ranker.py --llm_gt_pos 5 &
srun --exclusive --ntasks=1 python train_ranker.py --llm_gt_pos 4 &
srun --exclusive --ntasks=1 python train_ranker.py --llm_gt_pos 3 &
srun --exclusive --ntasks=1 python train_ranker.py --llm_gt_pos 2 &
srun --exclusive --ntasks=1 python train_ranker.py --llm_gt_pos 1


wait

echo "All scripts have completed."
