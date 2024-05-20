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

MODEL_NAME="lft_gan_discriminator_unet_muller_resizer_v11_angx5_scalex4"
OPTION_PATH="/work/pb035507/superresolution/lft_gan_discriminator_unet_muller_resizer_v11_angx5_scalex4/options/train_lft_gan_240501_122659.json"

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

WORKDIR=/work/pb035507

NODES=1
NTASKS=1
CPUS_PER_TASK=16
GPUS=1
MEMORY="24G"
TIME="1:00:00"
MAIL_TYPE="ALL"
MAIL_USER="aytac@linux.com"

OUTPUT_DIR="/home/pb035507/slurm/output"
ERROR_DIR="/home/pb035507/slurm/error"

TMP_DIR="/tmp/pb035507/login18-g-2_126567"
TARGET_DIR="/tmp/pb035507/login18-g-2_126567/Medical/mri"

mkdir -p "$OUTPUT_DIR" "$ERROR_DIR" || { echo "Failed to create directories"; exit 1; }

# Verify the target directory exists
if [ ! -d "$TARGET_DIR/data_for_train" ]; then
  echo "Target directory $TARGET_DIR/data_for_train does not exist."
  exit 1
fi

# PARTITION
# c23ms: 632 total, 96 cores/node, 256 GB/node, Claix-2023 (small memory partition)
# c23mm: 160 total, 96 cores/node, 512 GB/node, Claix-2023 (medium memory partition)
# c23ml: 2 total, 96 cores/node, 1024 GB/node, Claix-2023 (large memory partition)
# c23g: 52 total, 96 cores/node, 256 GB/node, Claix-2023 GPU partition with four H100 GPUs per node
# c18m: 1240 total, 48 cores/node, 192 GB/node, default partition for the "default" project
# c18g: 54 total, 48 cores/node, 192 GB/node, 2 V100 gpus, request of volta gpu needed to submit to this partition
# devel: 8 total, 48 cores/node, 192 GB/node, Designed for testing jobs and programs. Maximum runtime: 1 Hour
# Has to be used without a project!

PARTITION="c23g"

sbatch <<-EOT
#!/usr/bin/zsh

### Request BeeOND
#SBATCH --beeond

#SBATCH --account=p0021791
#SBATCH --time=$TIME
#SBATCH --partition=$PARTITION
#SBATCH --gres=gpu:$GPUS
#SBATCH --cpus-per-task=$CPUS_PER_TASK
#SBATCH --mem=$MEMORY
#SBATCH --nodes=$NODES
#SBATCH --ntasks=$NTASKS
#SBATCH --mail-type=$MAIL_TYPE
#SBATCH --mail-user=$MAIL_USER
#SBATCH --output=${OUTPUT_DIR}/o-%x.%j.%N.out
#SBATCH --error=${ERROR_DIR}/e-%x.%j.%N.err
#SBATCH --job-name=$JOB_NAME

# Debug: List contents of /tmp before execution
echo "Listing contents of /tmp before execution:"
ls -la /tmp

# Ensure the dataset directory is copied to the BeeOND mount
cp -r $TARGET_DIR $BEEOND/mri || { echo "Failed to copy files to BeeOND directory"; exit 1; }

# Change to the copied directory
cd $BEEOND/mri || { echo "Failed to change directory to $BEEOND/mri"; exit 1; }

# Debug: List contents of the working directory
echo "Listing contents of the working directory:"
ls -la

echo; nvidia-smi; echo

# Ensure the dataset directory exists inside the container
apptainer exec --nv --bind $BEEOND,$WORKDIR,$HOME $HOME/cuda_latest.sif bash -c '
  TARGET_DIR="/mnt/data_for_train"
  if [ ! -d "$TARGET_DIR" ]; then
    echo "Target directory $TARGET_DIR does not exist in the container."
    exit 1
  fi
  python -u $PWD/main_train_gan.py --opt=$OPTION_PATH
'

EOT

echo "Job $JOB_ID"
