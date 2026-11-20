#!/bin/bash
#SBATCH --job-name=pressureSweep
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=24                   # 6 pressures x 4 MPI ranks, all at once
#SBATCH --ntasks-per-node=24
#SBATCH --mem=60G
#SBATCH --time=24:00:00
#SBATCH --output=%x_%j.out
#SBATCH --mail-user=js42202@uga.edu
#SBATCH --mail-type=END,FAIL

# One planet at fixed mass, surface pressure varied with the air's composition
# held fixed. Writes pressure_sweep_<atmos>[_xfe<X_FE>].json with per-area GPP
# and global/land surface temperature for every pressure; plot it with
#   python plot_pressure_sweep.py pressure_sweep_n2.json
# Submit from prod_job/: sbatch run_pressure_sweep.sh
#
# ATMOS: n2 (Earth air) or mars (95% CO2). PHYSICS empty uses the mix's
# default (earth for n2, other for mars). X_FE empty is Earth-like iron
# (0.35); 0.25 is Mars-like.

set -e

module purge
module load GCCcore/11.3.0
module load OpenMPI/4.1.4-GCC-11.3.0
module load Python/3.10.4-GCCcore-11.3.0
module load HDF5/1.12.2-gompi-2022a

source ~/env/exoplasim_16cpu/bin/activate

ATMOS=n2
PHYSICS=
X_FE=
MASS=1
YEARS=30
PRESSURES=0.5,1,3,5,7,9

export EXOPLASIM_NCPUS=4
CORES=${SLURM_NTASKS:-24}
export EXOPLASIM_WORKERS=$(( CORES / EXOPLASIM_NCPUS ))

OUTPUT="pressure_sweep_${ATMOS}"
EXTRA=()
if [ -n "$PHYSICS" ]; then
    EXTRA+=(--physics "$PHYSICS")
fi
if [ -n "$X_FE" ]; then
    EXTRA+=(--x-fe "$X_FE")
    OUTPUT="${OUTPUT}_xfe${X_FE//./}"
fi
if [ "$MASS" != "1" ]; then
    OUTPUT="${OUTPUT}_m${MASS//./}"
fi

# Run once, not under srun (see run_job.sh). --max-minutes 0: no per-point
# cap; SLURM's --time is the limit for the whole job.
python run_gpp_diagnostics.py --axis pressure --atmos "$ATMOS" --masses "$MASS" \
    --pressures "$PRESSURES" --years "$YEARS" --max-minutes 0 \
    --output "${OUTPUT}.json" --no-plot "${EXTRA[@]}"
