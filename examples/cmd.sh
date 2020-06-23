#!/bin/bash
#SBATCH --ntasks=1 # Note that ntasks=1 runs multiple jobs in an array
#SBATCH --array=1-4%4
#SBATCH --gres=gpu:1
#SBATCH -p p100
#SBATCH --mem=16G
#SBATCH -o ./slurm_output/%J.out # this is where the output goes
#SBATCH --cpus-per-gpu=16        # number of CPUs we want per GPU
#SBATCH --gpus-per-task=1       # number of GPUs we want per task

. /scratch/ssd001/home/limoyool/pytorch15.env
cmd_line=$(sed "${SLURM_ARRAY_TASK_ID}q;d" ${1})
$cmd_line