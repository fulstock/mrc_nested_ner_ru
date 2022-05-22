#!/bin/bash
#SBATCH --output=br.out
#SBATCH --error=br.err
#SBATCH --cpus-per-task=20
#SBATCH --mem=128G
#SBATCH -p compute
#SBATCH --time=7-0:0
singularity run torch.sif python $HOME/mrc/prompts/lexical.py