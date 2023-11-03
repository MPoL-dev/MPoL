#!/bin/bash 

#SBATCH --nodes=1 
#SBATCH --ntasks=1 
#SBATCH --gpus=1
#SBATCH --mem=64GB 
#SBATCH --time=1:00:00 
#SBATCH --account=ipc5094_c_gpu
#SBATCH --partition=sla-prio

# load appropriate modules
module purge
module load ffmpeg/4.3.2
module load openmpi/4.1.1
module load anaconda3/2021.05

# load the virtual environment
# source /storage/home/ipc5094/Documents/scripts/RML_init.sh

pytest