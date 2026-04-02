#!/bin/bash
#SBATCH --job-name=testExoplasim        # Job name (testExoplasim)
#SBATCH --partition=batch             # Partition name (batch, highmem_p, or ...)
#SBATCH --nodes=1                     # Number of compute nodes for resources
#SBATCH --ntasks=16
#SBATCH --ntasks-per-node=16                    # 1 task (process) for below commands
## #SBATCH --cpus-per-task=16             # CPU core count per task, by default 1
#SBATCH --mem=60G                      # Memory per node (4GB); by default ...
#SBATCH --time=8:00:00                # Time limit hrs:min:sec or days-hours:mins
#SBATCH --output=%x_%j.out            # Standard output log, e.g., testBowtie2_12345.out
#SBATCH --mail-user=js42202@uga.edu  # Where to send mail
#SBATCH --mail-type=END,FAIL          # Mail events (BEGIN, END, FAIL, ALL)

set -e

module purge
module load GCCcore/11.3.0
module load OpenMPI/4.1.4-GCC-11.3.0
module load Python/3.10.4-GCCcore-11.3.0
module load HDF5/1.12.2-gompi-2022a

source ~/env/exoplasim_16cpu/bin/activate
cd exoplasimmodeling

srun python run_model.py
