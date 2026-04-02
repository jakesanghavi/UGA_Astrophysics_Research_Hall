#!/bin/bash
#SBATCH --job-name=testExoplasim        # Job name (testExoplasim)
#SBATCH --partition=batch             # Partition name (batch, highmem_p, or ...)
#SBATCH --nodes=1                     # Number of compute nodes for resources
#SBATCH --ntasks=1                    # 1 task (process) for below commands
#SBATCH --cpus-per-task=4             # CPU core count per task, by default 1
#SBATCH --mem=4G                      # Memory per node (4GB); by default ...
#SBATCH --time=1:00:00                # Time limit hrs:min:sec or days-hours:mins
#SBATCH --output=%x_%j.out            # Standard output log, e.g., testBowtie2_12345.out
#SBATCH --mail-user=js42202@uga.edu  # Where to send mail
#SBATCH --mail-type=END,FAIL          # Mail events (BEGIN, END, FAIL, ALL)

set -e

ml Python/3.10.8-GCCcore-12.2.0-bare
source ~/env/exoplasimenv/bin/activate
cd exoplasimmodeling

python veg_by_au_and_mass.py