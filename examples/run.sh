#!/bin/bash
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH -p compute
#SBATCH --time=7-0:0
#SBATCH --gres=gpu:4
#SBATCH --output=mrc/slurm/mrc_lexical_seed=44.out
#SBATCH --error=mrc/slurm/mrc_lexical_seed=44.err
singularity run --nv torch.sif \
	python $HOME/mrc/trainer.py --data_dir $HOME/data/RuNNE_lex --bert_config_dir $HOME/mrc/for_mrc/rubert \
	--max_length 192 --batch_size 16 --max_epochs 16 --gpus 4 --default_root_dir $HOME/mrc \
	--accumulate_grad_batches 1 --workers 4 --strategy ddp_spawn --num_nodes 1 --accelerator gpu \
	--seed 44
