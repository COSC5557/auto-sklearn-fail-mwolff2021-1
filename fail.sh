#!/bin/bash
#SBATCH --account=arcc-students
##SBATCH --job-name pml_hpo
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=mwolff3@uwyo.edu
#SBATCH --time=0-01:00:00
#SBATCH --mem=32GB
python3 /home/mwolff3/fail.py > fail_3.out

