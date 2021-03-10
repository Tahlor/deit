#!/bin/bash
#SBATCH --gres=gpu:4
#SBATCH -C 'rhel7&pascal'
#SBATCH --mem-per-cpu 32000M
#SBATCH --ntasks 24
#SBATCH --output="/lustre/scratch/grp/fslg_internn/deit/output/baseline.slurm"
#SBATCH --time 36:00:00
#SBATCH --mail-user=taylornarchibald@gmail.com   # email address
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL

#%Module

module purge
module load cuda/10.1
module load cudnn/7.6

export PATH="/lustre/scratch/grp/fslg_internn/env/deit:$PATH"
eval "$(conda shell.bash hook)"
conda activate "/lustre/scratch/grp/fslg_internn/env/deit"


cd "/lustre/scratch/grp/fslg_internn/deit"
which python
python -m torch.distributed.launch --nproc_per_node=4 --use_env main.py --model deit_tiny_patch16_224 --epochs 50 --resume "./output/baseline/checkpoint.pth" --batch-size 256 --data-set 'CIFAR' --data-path "../data" --output_dir ./output/baseline
