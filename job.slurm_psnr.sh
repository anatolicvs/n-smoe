#!/bin/bash

USE_APPTAINER=false

while getopts ":m:o:a:" opt; do
  case $opt in
    m)
      MODEL_NAME=$OPTARG
      ;;
    o)
      OPTION_PATH=$OPTARG
      ;;
    a)
      USE_APPTAINER=true
      ;;
    \?)
      echo "Invalid option: -$OPTARG" >&2
      exit 1
      ;;
    :)
      echo "Option -$OPTARG requires an argument." >&2
      exit 1
      ;;
  esac
done

TODAYS_DATE=$(date +%d%m%Y%H%M)
JOB_NAME="${TODAYS_DATE}__${MODEL_NAME}_JOBID_"

NODES=1
NTASKS=1
CPUS_PER_TASK=32
GPUS=4
MEMORY="32G"
TIME="32:00:00"
MAIL_TYPE="ALL"
MAIL_USER="aytac@linux.com"

OUTPUT_DIR="/home/pb035507/slurm/output"
ERROR_DIR="/home/pb035507/slurm/error"
WORKDIR="/hpcwork/p0021791"
mkdir -p "$OUTPUT_DIR" "$ERROR_DIR" || { echo "Failed to create directories"; exit 1; }

# PARTITION
# c23ms: 632 total, 96 cores/node, 256 GB/node, Claix-2023 (small memory partition)
# c23mm: 160 total, 96 cores/node, 512 GB/node, Claix-2023 (medium memory partition)
# c23ml: 2 total, 96 cores/node, 1024 GB/node, Claix-2023 (large memory partition)
# c23g: 52 total, 96 cores/node, 256 GB/node, Claix-2023 GPU partition with four H100 GPUs per node
# c18m: 1240 total, 48 cores/node, 192 GB/node, default partition for the "default" project
# c18g: 54 total, 48 cores/node, 192 GB/node, 2 V100 gpus, request of volta gpu needed to submit to this partition
# devel: 8 total, 48 cores/node, 192 GB/node, Designed for testing jobs and programs. Maximum runtime: 1 Hour
# Has to be used without an project!

PARTITION="c23g"

sbatch <<-EOT
#!/usr/bin/zsh

#SBATCH -A p0021791
#SBATCH --time=$TIME
#SBATCH --partition=$PARTITION
#SBATCH --gres=gpu:$GPUS
#SBATCH -c $CPUS_PER_TASK
#SBATCH --mem-per-gpu=90G
#SBATCH --nodes=$NODES
#SBATCH --ntasks=$NTASKS
#SBATCH --mail-type=$MAIL_TYPE
#SBATCH --mail-user=$MAIL_USER
#SBATCH --output=${OUTPUT_DIR}/o-%x.%j.%N.out
#SBATCH --error=${ERROR_DIR}/e-%x.%j.%N.err
#SBATCH --job-name=$JOB_NAME

echo "Starting job at: \$(date)"
nvidia-smi

export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=ALL

if [ "$USE_APPTAINER" = true ]; then
  apptainer exec --nv --bind $HOME,$HPCWORK,$WORK,$WORKDIR $WORKDIR/cuda.sif \
    torchrun --standalone --nnodes=1 --nproc-per-node=$GPUS $PWD/main_train_psnr.py --opt=$OPTION_PATH --dist
else
  module load Python/3.10.4
  source $HOME/venv/bin/activate

  torchrun --standalone --nnodes=1 --nproc-per-node=$GPUS $PWD/main_train_psnr.py --opt=$OPTION_PATH --dist
fi

echo "Job completed at: $(date)"
EOT

echo "Job submission complete."