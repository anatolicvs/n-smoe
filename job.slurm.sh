#!/bin/bash

# Usage:
#   ./script.sh -m "your_model_name" -o "your_option_path"
#
# Description:
#   This script submits a batch job using sbatch with configurable MODEL_NAME and OPTION_PATH
#   parameters. Default values are provided for both if they are not specified.
#
# Options:
#   -m  MODEL_NAME     The name of the model to be used (default: lft_gan_discriminator_unet_muller_resizer_v4_angx5_scalex4)
#   -o  OPTION_PATH    The path to the training options JSON file (default: /home/pb035507/n-smoe/options/train_lft_gan.json)

while getopts ":m:o:" opt; do
  case $opt in
    m)
      MODEL_NAME=$OPTARG
      ;;
    o)
      OPTION_PATH=$OPTARG
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
CPUS_PER_TASK=16
GPUS=1
MEMORY="16G"
TIME="18:00:00"
MAIL_TYPE="ALL"
MAIL_USER="aytac@linux.com"

OUTPUT_DIR="/home/pb035507/slurm/output"
ERROR_DIR="/home/pb035507/slurm/error"

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
#SBATCH --mem=$MEMORY
#SBATCH --nodes=$NODES
#SBATCH --ntasks=$NTASKS
#SBATCH --mail-type=$MAIL_TYPE
#SBATCH --mail-user=$MAIL_USER
#SBATCH --output=${OUTPUT_DIR}/o-%x.%j.%N.out
#SBATCH --error=${ERROR_DIR}/e-%x.%j.%N.err
#SBATCH --job-name=$JOB_NAME

echo; nvidia-smi; echo

# DATASET_DIR="/tmp/pb035507/dataset/multicoil_train"

# if [ ! -d "$DATASET_DIR" ]; then
#   echo "Directory $DATASET_DIR does not exist"
#   exit 1
# fi

# ls -lt $DATASET_DIR

apptainer exec --nv --bind $HOME,$HPCWORK,$WORK $HOME/cuda_latest.sif python -u $PWD/main_train_gan.py --opt=$OPTION_PATH
EOT

echo "Job $JOB_ID"
