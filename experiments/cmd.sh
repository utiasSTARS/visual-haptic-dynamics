#!/bin/bash
#SBATCH --ntasks=1 # Note that ntasks=1 runs multiple jobs in an array
#SBATCH --array=1-12%12
#SBATCH --gres=gpu:1
#SBATCH --partition=rtx6000,t4v1,p100,t4v2
#SBATCH --mem=16G
#SBATCH --qos=normal
#SBATCH --output ./slurm_output/%J.out # this is where the output goes
#SBATCH --cpus-per-gpu=16        # number of CPUs we want per GPU
#SBATCH --gpus-per-task=1       # number of GPUs we want per task

. /scratch/ssd001/home/limoyool/pt151.env
cmd_line=$(sed "${SLURM_ARRAY_TASK_ID}q;d" ${1})
$cmd_line
