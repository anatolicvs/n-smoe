#!/usr/bin/zsh

# PARTITION DETAILS
# c23ms: 632 total, 96 cores/node, 256 GB/node, Claix-2023 (small memory partition)
# c23mm: 160 total, 96 cores/node, 512 GB/node, Claix-2023 (medium memory partition)
# c23ml: 2 total, 96 cores/node, 1024 GB/node, Claix-2023 (large memory partition)
# c23g: 52 total, 96 cores/node, 256 GB/node, Claix-2023 GPU partition with four H100 GPUs per node
# c18m: 1240 total, 48 cores/node, 192 GB/node, default partition for the "default" project
# c18g: 54 total, 48 cores/node, 192 GB/node, 2 V100 GPUs, requires Volta GPU request
# devel: 8 total, 48 cores/node, 192 GB/node, Designed for testing jobs and programs. Max runtime: 1 Hour

USE_APPTAINER=true
BUILT_VERSION="1.3"
DISTRIBUTED_TRAINING=true
GPUS=${GPUS:-4}  # Default to 4 if not set

while getopts ":m:o:a:dg:h" opt; do
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
    d)
      DISTRIBUTED_TRAINING=true
      ;;
    g)
      GPUS=$OPTARG
      ;;
    h)
      echo "Usage: $0 [-m model_name] [-o option_path] [-a] [-d] [-g num_gpus]"
      echo "  -m: Specify the model name"
      echo "  -o: Path to options file"
      echo "  -a: Use Apptainer"
      echo "  -d: Enable distributed training (only effective if GPUs > 1)"
      echo "  -g: Number of GPUs to use (default 4)"
      exit 0
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

if [ "$GPUS" -lt 1 ]; then
  echo "Error: Number of GPUs must be at least 1" >&2
  exit 1
fi

if [ "$DISTRIBUTED_TRAINING" = true ] && [ "$GPUS" -le 1 ]; then
  echo "Warning: Distributed training is enabled but only 1 GPU is specified. Disabling distributed training."
  DISTRIBUTED_TRAINING=false
fi

if [ -z "$MODEL_NAME" ]; then
  echo "Error: MODEL_NAME must be specified." >&2
  exit 1
fi

TODAYS_DATE=$(date +%d%m%Y%H%M)
JOB_NAME="${TODAYS_DATE}__${MODEL_NAME}"

NODES=1
NTASKS_PER_NODE=$GPUS
CPUS_PER_TASK=16
MEM_PER_GPU=90G
TOTAL_MEM=$((GPUS * 90))G

TIME="8:00:00"
MAIL_TYPE="ALL"
MAIL_USER="aytac@linux.com"

OUTPUT_DIR="/hpcwork/p0021791/slurm/output"
ERROR_DIR="/hpcwork/p0021791/slurm/error"
WORKDIR="/hpcwork/p0021791"

mkdir -p "$OUTPUT_DIR" "$ERROR_DIR" "/home/p0021791/tmp" || {
  echo "Failed to create output, error, or tmp directories" >&2
  exit 1
}

PARTITION="c23g"


get_idle_node() {
    # sinfo -N -p "$PARTITION" -h -o "%N %T" | grep -w "idle" | awk '{print $1; exit}'
    sinfo -N -p "c23g" -h -o "%N" --states=idle | head -n 1
    # sinfo -N -p "$PARTITION" -h -o "%N %T" | grep -w "idle" | awk '{print $1}' | head -n 1
}

# EXCLUDE_NODES="n23g0009"

# get_idle_node() {
#     sinfo -N -p "$PARTITION" -h -o "%N %T %G" | awk -v gpus="$GPUS" -v exclude_nodes="$EXCLUDE_NODES" '
#     $2 == "idle" {
#         split(exclude_nodes, excl_nodes, ",");
#         for (node in excl_nodes) {
#             if ($1 == excl_nodes[node]) {
#                 next;  # Skip the excluded nodes
#             }
#         }
#         split($3, gpu_info, ":");
#         if (gpu_info[2] >= gpus) {
#             print $1;
#             exit;
#         }
#     }'
# }

IDLE_NODE=$(get_idle_node)
if [ -z "$IDLE_NODE" ]; then
    echo "Error: No idle nodes found in partition $PARTITION" >&2
    exit 1
fi

JOB_SCRIPT=$(mktemp /home/p0021791/tmp/job_script.XXXXXX)
if [ ! -f "$JOB_SCRIPT" ]; then
    echo "Failed to create a temporary job script file." >&2
    exit 1
fi

# idle nodes n23g[0022-0029],r23g[0001-0005],w23g[0001-0003]

VISIBLE_DEVICES=$(seq -s, 0 $((GPUS - 1)))

SAVE_DIR="/hpcwork/p0021791/zoo/vir-n-smoe/x4/v3/gaussian_cauchy/"

if [ ! -d "$SAVE_DIR" ]; then
  mkdir -p "$SAVE_DIR" || {
    echo "Failed to create save directory at $SAVE_DIR" >&2
    exit 1
  }
fi

cat <<-EOT > "$JOB_SCRIPT"
#!/usr/bin/zsh

#SBATCH -A p0021791
#SBATCH --nodelist=$IDLE_NODE
#SBATCH --time=$TIME
#SBATCH --partition=$PARTITION
#SBATCH --gres=gpu:$GPUS
#SBATCH -c $CPUS_PER_TASK
# #SBATCH --mem=$TOTAL_MEM
#SBATCH --mem-per-gpu=90G
#SBATCH --nodes=$NODES
#SBATCH --ntasks-per-node=$NTASKS_PER_NODE
#SBATCH --mail-type=$MAIL_TYPE
#SBATCH --mail-user=$MAIL_USER
#SBATCH --output=${OUTPUT_DIR}/o-%x.%j.%N.out
#SBATCH --error=${ERROR_DIR}/e-%x.%j.%N.err
#SBATCH --job-name=$JOB_NAME

export CC=gcc
export CXX=g++

module load CUDA/12.6.1

echo "Starting job at: \$(date)"
nvidia-smi
echo "GPUs available: \$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)"

# export TORCH_DISTRIBUTED_DEBUG=INFO
export TORCH_NCCL_BLOCKING_WAIT=1
export NCCL_BLOCKING_WAIT=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
# export NCCL_DEBUG=INFO
# export NCCL_DEBUG_SUBSYS=ALL
export NCCL_TIMEOUT=1200
# export TORCH_DISTRIBUTED_DEBUG=DETAIL
export NCCL_P2P_LEVEL=PXB
export NCCL_P2P_DISABLE=1

WANDB_KEY_FILE="${WORKDIR}/wandb_api_key.txt"
if [ -f "\$WANDB_KEY_FILE" ]; then
  export WANDB_API_KEY=\$(cat "\$WANDB_KEY_FILE")
else
  echo "Error: WANDB API key file not found at \$WANDB_KEY_FILE" >&2
  exit 1
fi

if [ "$USE_APPTAINER" = true ]; then
  if [ -z "$HPCWORK" ] || [ -z "$WORK" ] || [ -z "$WORKDIR" ]; then
      echo "Error: Required environment variables (HPCWORK, WORK, WORKDIR) are not set." >&2
      exit 1
  fi
  apptainer exec --nv --bind $HOME,$HPCWORK,$WORK,$WORKDIR $WORKDIR/cuda_v${BUILT_VERSION}.sif \
    torchrun --standalone --nnodes=1 --nproc-per-node=$GPUS $PWD/train_SISR.py --config=$OPTION_PATH --save_dir=$SAVE_DIR
else
  module load Python/3.10.4
  source $WORKDIR/env/bin/activate
  torchrun --standalone --nnodes=1 --nproc-per-node=$GPUS $PWD/train_SISR.py --config=$OPTION_PATH --save_dir=$SAVE_DIR
fi

echo "Job completed at: \$(date)"
EOT

chmod +x "$JOB_SCRIPT"

job_id=$(sbatch "$JOB_SCRIPT" | awk '{print $4}')

echo "Job submission complete. Job ID: $job_id"

sleep 5
echo "Job status:"
squeue -j $job_id
