# Test plan for the packed pixi env on lem (documents/notes/lem_packed_pixi_env.md).
# Not meant to be executed: copy the commands one at a time. Each is a single line.
# Hash below = this branch's pixi.lock; update_pixi.py prints the current one.

# --- 1. Pack [laptop, repo root] -------------------------------------------------------
# Dry run first: prints the hash, the pack path and the sbatch flags; touches nothing.
pixi run python update_pixi.py --config-path configs/jm/feature --config-name h4_lem_packed_env +dry_run=true
# Submits one batch job (1 GPU, ~20 min) and waits. Ends with "✓ Packed .../nano-env-<hash>.tar".
pixi run python update_pixi.py --config-path configs/jm/feature --config-name h4_lem_packed_env
# In the printed log tail expect: "imports ok: torch 2.7.1+cu128, cuda available: True", "torchrun ok", "pixi_unpack_env: ready in <N>s".
# Alternative that doesn't block while the pack job queues: add +wait=false, then give run_exp.py the printed ++infrastructure.slurm.dependency=afterok:<id> (see the note).

# --- 2. Look at the result [lem login node] ---------------------------------------------
ls -la /lustre/pd03/plgrid/plgllmefficont3/nano/pretraining/$USER/pixi_packs/ /lustre/pd03/plgrid/plgllmefficont3/nano/pretraining/$USER/pixi_packs/builds/*/
grep -E "pixi_unpack_env|imports ok|torchrun ok|packed:|done:" /lustre/pd03/plgrid/plgllmefficont3/nano/pretraining/$USER/pixi_packs/builds/*/slurm-*.out
lfs quota -h -p 649 /lustre/pd03

# --- 3. Interactive check [lem login node -> compute node] ------------------------------
srun -p plgrid-lem-gpu-h100 -A plgllmefficont3 -N 1 --gres=gpu:hopper:1 -c 8 --mem=32G -t 00:30:00 --extra=FORCE_RM_TMPDIR --pty bash -l
# Inside the srun shell: where node-local disk is and how big (expect TMPDIR_LOCAL=/mnt/lscratch/slurm/<jobid>, ~40G).
echo "TMPDIR=$TMPDIR TMPDIR_LOCAL=$TMPDIR_LOCAL"; df -h "$TMPDIR_LOCAL"; which tar zstd pigz unsquashfs
export PIXI_PACK_DIR=/lustre/pd03/plgrid/plgllmefficont3/nano/pretraining/$USER/pixi_packs PIXI_LOCK_HASH=cf69ad40a59f5c76
# The job's exact unpack step, from the copy the pack job staged (on failure it exits this shell, by design).
source "$(ls -td $PIXI_PACK_DIR/builds/*/ | head -1)pixi_unpack_env.sh"
python -c "import torch, hydra, lm_eval; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name())"; which python torchrun
du -sh "$PIXI_ENV_DIR"; df -h "$TMPDIR_LOCAL"
# Leaving the shell runs the script's trap, which deletes the env.
exit

# --- 4. Smoke run [laptop, repo root] ---------------------------------------------------
# 200 steps of h4 on 2 GPUs, evals and saves at 100 and 200. Pass: the log starts with
# "pixi_unpack_env: ... ready in <N>s", then trains, logs to W&B, evals and saves step_99/step_199 to checkpoints_test.
pixi run python run_exp.py --config-path configs/jm/feature --config-name h4_lem_packed_env

# --- 5. Check the smoke job [lem login node] --------------------------------------------
sacct -X -u $USER -n -S today --name=h4_lem_packed_env --format=JobID,State,Elapsed,ExitCode
grep -E "pixi_unpack_env|Traceback|Error" /lustre/pd03/plgrid/plgllmefficont3/nano/pretraining/$USER/llmrandom_cemetery/h4_lem_packed_env_*/slurm-*.out | head -20

# --- 6. Optional, before scaling up: 2-node smoke run [laptop, repo root] ---------------
# 1 GPU per node, storage:local:40G on each. Pass: "unpacking ... on 2 node(s)", then torchrun trains across both nodes.
pixi run python run_exp.py --config-path configs/jm/feature --config-name h4_lem_packed_env_2node
