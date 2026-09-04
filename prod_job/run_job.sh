#!/bin/bash
#SBATCH --job-name=testExoplasim        # Job name (testExoplasim)
#SBATCH --partition=batch             # Partition name (batch, highmem_p, or ...)
#SBATCH --nodes=1                     # Number of compute nodes for resources
#SBATCH --ntasks=16                   # Reserve 16 slots on the node for MPI ranks
#SBATCH --ntasks-per-node=16
## #SBATCH --cpus-per-task=16             # CPU core count per task, by default 1
#SBATCH --mem=60G                      # Memory per node (4GB); by default ...
#SBATCH --time=12:00:00                # Time limit hrs:min:sec or days-hours:mins
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

# Parallelism: run EXOPLASIM_WORKERS planets at once, each using EXOPLASIM_NCPUS
# MPI ranks. Keep WORKERS * NCPUS <= cores allocated on the node. model_helpers
# reads these env vars; here we size WORKERS to the allocation.
export EXOPLASIM_NCPUS=4
CORES=${SLURM_NTASKS:-16}
export EXOPLASIM_WORKERS=$(( CORES / EXOPLASIM_NCPUS ))

# Run the driver ONCE -- do NOT use `srun`. srun would launch one copy of
# run_model.py per task (16 copies), each spawning its own worker pool and
# mpiexec, which oversubscribes cores and clobbers shared output files.
# Instead, exoplasim launches its own `mpiexec -np NCPUS` per planet, and
# run_model.py's process pool runs EXOPLASIM_WORKERS planets concurrently.
python run_model.py
