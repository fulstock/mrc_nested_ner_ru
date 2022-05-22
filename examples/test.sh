#!/bin/bash
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH -p compute
#SBATCH --time=7-0:0
#SBATCH --gres=gpu:1
#SBATCH --output=mrc/slurm/tests/RuNNE_lex/77.out
#SBATCH --error=mrc/slurm/tests/RuNNE_lex/77.err
singularity run --nv torch.sif \
	python $HOME/mrc/tester.py --data_dir $HOME/data/RuNNE_lex --bert_config_dir $HOME/mrc/for_mrc/rubert \
	--max_length 192 --batch_size 16 --gpus 1 --default_root_dir $HOME/mrc \
	--workers 2 --accelerator gpu --pretrained_checkpoint $HOME/mrc/ckpt/model-RuNNE_lex-seed=77-220519_061614-epoch=15.ckpt \
	--seed 77
