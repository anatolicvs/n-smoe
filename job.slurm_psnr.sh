#!/usr/bin/zsh

USE_APPTAINER=true
BUILT_VERSION="1.2"
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
NTASKS=1
CPUS_PER_TASK=16
MEMORY="32G"
TIME="24:00:00"
MAIL_TYPE="ALL"
MAIL_USER="aytac@linux.com"

OUTPUT_DIR="/hpcwork/p0021791/slurm/output"
ERROR_DIR="/hpcwork/p0021791/slurm/error"
WORKDIR="/hpcwork/p0021791"

mkdir -p "$OUTPUT_DIR" "$ERROR_DIR" || { echo "Failed to create output or error directories"; exit 1; }

if [ ! -w "$OUTPUT_DIR" ] || [ ! -w "$ERROR_DIR" ]; then
  echo "Error: Output or error directory is not writable." >&2
  exit 1
fi

mkdir -p "/home/p0021791/tmp" || { echo "Failed to create /home/p0021791/tmp directory"; exit 1; }

PARTITION="c23g"


get_idle_node() {
    # sinfo -N -p "$PARTITION" -h -o "%N %T" | grep -w "idle" | awk '{print $1; exit}'
    sinfo -N -p "$PARTITION" -h -o "%N %T" | grep -w "idle" | awk '{print $1}' | head -n 1
}

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

cat <<-EOT > "$JOB_SCRIPT"
#!/usr/bin/zsh

#SBATCH -A p0021791
#SBATCH --nodelist=$IDLE_NODE 
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

# module load CUDA/12.6.1

echo "Starting job at: \$(date)"
nvidia-smi
echo "GPUs available: \$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)"

# export TORCH_DISTRIBUTED_DEBUG=INFO
export NCCL_BLOCKING_WAIT=1
export TORCH_NCCL_BLOCKING_WAIT=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
# export NCCL_DEBUG=INFO
# export NCCL_DEBUG_SUBSYS=ALL
export NCCL_TIMEOUT=1800
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
    torchrun --standalone --nnodes=1 --nproc-per-node=$GPUS $PWD/main_train_psnr.py --opt=$OPTION_PATH $([ "$DISTRIBUTED_TRAINING" = true ] && echo "--dist")
else
  module load Python/3.10.4
  source $WORKDIR/env/bin/activate
  torchrun --standalone --nnodes=1 --nproc-per-node=$GPUS $PWD/main_train_psnr.py --opt=$OPTION_PATH $([ "$DISTRIBUTED_TRAINING" = true ] && echo "--dist")
fi

echo "Job completed at: \$(date)"
EOT

chmod +x "$JOB_SCRIPT"

job_id=$(sbatch "$JOB_SCRIPT" | awk '{print $4}')

echo "Job submission complete. Job ID: $job_id"

sleep 5

echo "Job status:"
squeue -j $job_id