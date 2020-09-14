#!/usr/bin/env bash
#SBATCH --mincpus 2
#SBATCH --mem 4000
#SBATCH --exclude rockford,steele,hammer,conan,blomquist,wolfe,knatterton,holmes,lenssen,scuderi,matula,marlowe,poirot,monk


srun -u python remote_experiment.py "$@"
