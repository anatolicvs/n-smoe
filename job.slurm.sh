#!/bin/bash

# MODEL_NAME=lft_gan_discriminator_unet_muller_resizer_v4_angx5_scalex4
# MODEL_NAME=conv_nsmoe_muller_x4_v1_gan_grayscale
MODEL_NAME=lft_gan_discriminator_unet_muller_resizer_v4_angx5_scalex4
WORKDIR=/work/pb035507
# OPTION_PATH=/home/pb035507/works/hpc-training/n-smoe/options/train_lft_gan.json
# OPTION_PATH=/home/pb035507/works/hpc-training/n-smoe/options/train_umoe_muller__sr_gan_x4.json
# OPTION_PATH=/work/pb035507/superresolution/lft_gan_discriminator_unet_muller_resizer_v4_angx5_scalex4/options/train_lft_gan_240426_201426.json

# OPTION_PATH=/home/pb035507/works/hpc-training/n-smoe/options/train_convsmoe_gan.json
OPTION_PATH=/home/pb035507/n-smoe/options/train_lft_gan.json

JOB_NAME="${MODEL_NAME}"

NODES=1
NTASKS=1
CPUS_PER_TASK=16
GPUS=1
MEMORY="4G"
TIME="2:00:00"
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
#!/bin/bash
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

echo; export; echo; nvidia-smi; echo

apptainer exec --nv --bind $WORKDIR $HOME/cuda_latest.sif python -u $PWD/main_train_gan.py --opt=$OPTION_PATH
EOT

echo "Job $JOB_ID"