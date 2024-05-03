#!/bin/bash -l

# Set SCC project
#$ -P ds598xz

# Assuming you want to run a job array with 8 combinations (2 batch sizes x 2 learning rates x 2 epochs, you would need to set this number manually)

# module load miniconda/4.9.2
# conda activate ayangrl

# python UNet_attention/train.py

module load miniconda
module load academic-ml/spring-2024

conda activate spring-2024-pyt

export PYTHONPATH="/projectnb/ds598xz/students/ayang903/final/sp2024_RL:$PYTHONPATH"

python UNet_attention/train.py

##qsub -pe omp 4 -P ds598xz -l gpus=1 run.sh