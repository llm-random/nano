#!/bin/bash
# Bank gates (documents/notes/ame_bank_gates_plan.md): run evaluation scripts on one
# LEM H100, from a committed state of the repo -- the LEM counterpart of
# parity_audit's parity_submit.sh + parity_score_runs.sh, which target entropy.
#
#   documents/scripts/bank_gates_lem_eval.sh COMMIT NAME TIME -- SCRIPT [ARGS...] [:: SCRIPT [ARGS...]]
#
# COMMIT is git-archived (once per commit) to $BG/code/<short>, with EVAL_COMMIT
# beside it. Each SCRIPT is a path relative to the repo root; commands separated by
# `::` run one after the other in the same job, each with `--out $BG/out/NAME_<job>`
# appended. $BG is the bank-gates area on LEM's Lustre, which also holds the
# held-out set (data/parity_batches.pt, copied from entropy's parity dump).
# Arguments may use $BG and $CEM (the cemetery); they are expanded on the node.
# An argument with a space stays one argument. A failed command is reported in the
# log and counted; if any failed, the job exits 1, so Slurm reports it as FAILED.
# Prints the job id. Log: $BG/logs/NAME_<job>.out.
set -euo pipefail
COMMIT=$1; NAME=$2; TIME=$3; shift 3
[ "$1" = "--" ] && shift
REPO=$(git -C "$(dirname "$0")" rev-parse --show-toplevel)
FULL=$(git -C "$REPO" rev-parse "$COMMIT")
SHORT=${FULL:0:8}
BG_REMOTE='/lustre/pd03/plgrid/plgllmefficont3/nano/$USER/bank_gates'
if ! ssh -o BatchMode=yes lem "test -f $BG_REMOTE/code/$SHORT/EVAL_COMMIT"; then
  git -C "$REPO" archive "$FULL" | ssh -o BatchMode=yes lem \
    "mkdir -p $BG_REMOTE/code/$SHORT $BG_REMOTE/logs $BG_REMOTE/out && tar -x -C $BG_REMOTE/code/$SHORT && echo $FULL > $BG_REMOTE/code/$SHORT/EVAL_COMMIT"
fi
# The commands, one `check` line each. Every argument is double-quoted with only
# \, " and ` escaped: an argument with a space stays one, and $BG and $CEM still
# expand on the node.
quote() { local a=${1//\\/\\\\}; a=${a//\"/\\\"}; a=${a//\`/\\\`}; printf '"%s"' "$a"; }
COMMANDS=""
CMD=""
for ARG in "$@"; do
  if [ "$ARG" = "::" ]; then
    if [ -n "$CMD" ]; then COMMANDS+="check$CMD"$'\n'; fi
    CMD=""
  else
    CMD+=" $(quote "$ARG")"
  fi
done
if [ -n "$CMD" ]; then COMMANDS+="check$CMD"$'\n'; fi
ssh -o BatchMode=yes lem "cat > $BG_REMOTE/logs/$NAME.sbatch && sbatch --parsable $BG_REMOTE/logs/$NAME.sbatch" <<SBATCH
#!/bin/bash
#SBATCH --partition=plgrid-lem-gpu-h100
#SBATCH --account=plgllmefficont3
#SBATCH --nodes=1
#SBATCH --gres=gpu:hopper:1
#SBATCH --cpus-per-gpu=16
#SBATCH --mem-per-gpu=228G
#SBATCH --exclude=r16-11
#SBATCH --time=$TIME
#SBATCH --job-name=$NAME
#SBATCH --output=/lustre/pd03/plgrid/plgllmefficont3/nano/%u/bank_gates/logs/${NAME}_%j.out
# The environment of configs/_cluster/lem.yaml's job script.
export PROJECT_HOME_PATH=/lustre/pd03/plgrid/plgllmefficont3/nano
export HF_HOME=\$PROJECT_HOME_PATH/hf_cache
export HYDRA_FULL_ERROR=1
export USER_HOME_PATH=\$PROJECT_HOME_PATH/\$USER
export HF_DATASETS_CACHE="\$USER_HOME_PATH/hf_datasets"
export HF_MODULES_CACHE="\$USER_HOME_PATH/hf_modules"
export HF_HUB_CACHE="\$USER_HOME_PATH/hf_cache"
export HF_DATASETS_TRUST_REMOTE_CODE=1
export PIXI_HOME=\$USER_HOME_PATH/pixi
export PATH="\$HOME/.pixi/bin:\$PATH"
export XDG_DATA_HOME="\$USER_HOME_PATH/data"
export XDG_CACHE_HOME="\$USER_HOME_PATH/cache"
export XDG_STATE_HOME="\$USER_HOME_PATH/state"
export RATTLER_CACHE_DIR="\$XDG_CACHE_HOME/rattler"
export PIXI_CACHE_DIR="\$XDG_CACHE_HOME/pixi"
cd "\$PIXI_HOME" && eval "\$(pixi shell-hook)" && cd - > /dev/null
export BG=\$USER_HOME_PATH/bank_gates
export CEM=\$USER_HOME_PATH/llmrandom_cemetery
OUT=\$BG/out/${NAME}_\$SLURM_JOB_ID
mkdir -p \$OUT
cd \$BG/code/$SHORT
echo "commit \$(cat EVAL_COMMIT), job \$SLURM_JOB_ID on \$(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
FAILED=0
check() { python "\$@" --out "\$OUT" || { echo "CHECK FAILED: \$*"; FAILED=\$((FAILED + 1)); }; }
$COMMANDS
if [ "\$FAILED" -gt 0 ]; then echo "\$FAILED check(s) failed"; exit 1; fi
SBATCH
