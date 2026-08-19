#!/usr/bin/env bash
#SBATCH -J fno_ddp_2n8g
#SBATCH -p sched_mit_psfc_gpu_r8
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH -c 16                            # 16 CPUs per node
#SBATCH --mem=0
#SBATCH -t 08:00:00                      # 8 hours for 5000 epochs
#SBATCH -o %x_%j.out

# NOTE: Removed -u flag to avoid "unbound variable" errors from /etc/bashrc
set -eo pipefail

# =============================================================================
# Job info (printed before srun, so no conda needed here)
# =============================================================================
MASTER_ADDR=$(scontrol show hostnames "$SLURM_NODELIST" | head -n1)
MASTER_PORT=${MASTER_PORT:-29500}
export MASTER_ADDR MASTER_PORT

NUM_GPUS_PER_NODE=4
NNODES=$SLURM_NNODES

echo "=============================================="
echo "Job ID       : $SLURM_JOB_ID"
echo "Job name     : $SLURM_JOB_NAME"
echo "Partition    : $SLURM_JOB_PARTITION"
echo "Nodes        : $SLURM_NODELIST"
echo "NNODES       : $NNODES"
echo "GPUs per node: $NUM_GPUS_PER_NODE"
echo "Total GPUs   : $((NNODES * NUM_GPUS_PER_NODE))"
echo "CPUs per node: ${SLURM_CPUS_ON_NODE:-16}"
echo "Master       : $MASTER_ADDR:$MASTER_PORT"
echo "=============================================="

# =============================================================================
# Environment variables (exported to srun via --export=ALL)
# =============================================================================

# Threading optimization:
# OMP_NUM_THREADS: threads per DataLoader worker
# 16 CPUs / (4 workers × 4 GPUs) = 1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export TORCH_NUM_THREADS=1

# NCCL settings for multi-node communication
# Use the new env var name (old NCCL_ASYNC_ERROR_HANDLING is deprecated)
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

# Set NCCL debug level (INFO for debugging, WARN for production)
export NCCL_DEBUG=INFO

# Python fault handler for better stack traces on crashes
export PYTHONFAULTHANDLER=1

# =============================================================================
# Per-node hardware check (optional, can comment out for production)
# =============================================================================
echo ""
echo "=============================================="
echo "Per-node hardware/software check"
echo "=============================================="

# IMPORTANT: conda activate MUST be inside srun because srun spawns a new shell
# that does not inherit the parent's conda environment
srun --nodes=$NNODES --ntasks=$NNODES --ntasks-per-node=1 bash -lc '
# Disable unbound variable check for bashrc compatibility
set +u
source ~/.bashrc
conda activate torch
set -eo pipefail

echo "----- NODE: $(hostname) -----"
echo "[GPU]"
nvidia-smi -L
echo "[Python] $(which python) - $(python -V 2>&1)"
echo "[PyTorch]"
python -c "
import torch
print(f\"  torch={torch.__version__}, cuda={torch.version.cuda}\")
print(f\"  GPUs={torch.cuda.device_count()}, NCCL={torch.distributed.is_nccl_available()}\")
"
echo "------------------------------"
'

# =============================================================================
# Launch DDP training
# =============================================================================
echo ""
echo "=============================================="
echo "DDP Training: $NNODES nodes x $NUM_GPUS_PER_NODE GPUs = $((NNODES * NUM_GPUS_PER_NODE)) processes"
echo "=============================================="

# IMPORTANT: conda activate MUST be inside srun because srun spawns a new shell
# Each node runs one torchrun launcher, which spawns NUM_GPUS_PER_NODE workers
srun --nodes=$NNODES \
     --ntasks=$NNODES \
     --ntasks-per-node=1 \
     --cpus-per-task=16 \
     --export=ALL \
     bash -lc '
# Disable unbound variable check for bashrc compatibility
set +u
source ~/.bashrc
conda activate torch
set -eo pipefail

echo "[$(hostname)] Starting torchrun (node_rank=$SLURM_NODEID)..."

# Use rdzv_backend=c10d for robust multi-node rendezvous
# This is the recommended approach for torchrun >= 1.10
torchrun \
  --nnodes='"$NNODES"' \
  --nproc_per_node='"$NUM_GPUS_PER_NODE"' \
  --rdzv_backend=c10d \
  --rdzv_endpoint='"$MASTER_ADDR:$MASTER_PORT"' \
  Train_Server.py
'

echo ""
echo "=============================================="
echo "Job finished at $(date)"
echo "=============================================="
