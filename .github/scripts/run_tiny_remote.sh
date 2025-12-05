#!/bin/bash -l

#SBATCH --cpus-per-gpu=16
#SBATCH --gres=gpu:1
#SBATCH --job-name=tiny_remote_ci
#SBATCH --mem-per-gpu=125G
#SBATCH --nodes=1
#SBATCH --partition=a100
#SBATCH --time=00:10:00

#---------- SCRIPT ----------
export PROJECT_HOME_PATH=/storage_ssd_1/nano
export HF_HOME=$PROJECT_HOME_PATH/hf_cache
export HYDRA_FULL_ERROR=1
export PIXI_HOME=/storage_ssd_1/nano/pixi
export PATH="$PIXI_HOME/bin:$PATH"
export XDG_DATA_HOME="$PIXI_HOME/data"
export XDG_CACHE_HOME="$PIXI_HOME/cache"
export XDG_STATE_HOME="$PIXI_HOME/state"
cd "$PIXI_HOME"
eval "$(pixi shell-hook)"
cd -
#-------- SCRIPT END --------

export MASTER_ADDR=$(scontrol show hostname ${SLURM_NODELIST} | head -n 1)
export MASTER_PORT=$((40000 + ${SLURM_JOB_ID} % 10000))

srun torchrun --nnodes=${SLURM_NNODES}\
  --nproc-per-node=${SLURM_GPUS_ON_NODE} \
  --rdzv-id=${SLURM_JOBID} \
  --rdzv-backend=c10d \
  --rdzv-endpoint=${MASTER_ADDR}:${MASTER_PORT} \
  main.py \
    --config-path=configs \
    --config-name=tiny_remote
