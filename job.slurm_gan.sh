#!/bin/bash

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
CPUS_PER_TASK=32
GPUS=1
TIME="30:00:00"
MAIL_TYPE="ALL"
MAIL_USER="aytac@linux.com"

OUTPUT_DIR="/home/pb035507/slurm/output"
ERROR_DIR="/home/pb035507/slurm/error"
WORKDIR="/hpcwork/p0021791"

mkdir -p "$OUTPUT_DIR" "$ERROR_DIR" || { echo "Failed to create directories"; exit 1; }

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

echo "Starting job at: $(date)"
nvidia-smi
echo "Attempting to bind SquashFS container..."
apptainer exec --nv --bind $HOME,$HPCWORK,$WORK,$WORKDIR $HOME/cuda_latest.sif python -u $PWD/main_train_gan.py --opt=$OPTION_PATH
echo "Job completed at: $(date)"
EOT

echo "Job submission complete. Monitor job with squeue or check output/error files."
