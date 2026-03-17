NEPTUNE_PROJECT="pmtest/llm-random"   # workspace/project
# DATA_DIR="/home/janek/Downloads/janeks/data_dir"
# FILES_DIR="/home/janek/Downloads/janeks/files_dir"
DATA_DIR="data_dir"
FILES_DIR="files_dir"

WANDB_ENTITY="ideas_cv"
NAME_PREFIX="llm-random"

# 1) export only the listed runs (regex matches EXACTLY these IDs)
uv run neptune-exporter export \
  --exporter neptune2 \
  -p "$NEPTUNE_PROJECT" \
  -r "^(LLMRANDOM-42561|LLMRANDOM-41946|LLMRANDOM-41945|LLMRANDOM-41944|LLMRANDOM-41943|LLMRANDOM-41942|LLMRANDOM-41941|LLMRANDOM-41938|LLMRANDOM-41930|LLMRANDOM-41907|LLMRANDOM-41906|LLMRANDOM-41899|LLMRANDOM-41898|LLMRANDOM-41897|LLMRANDOM-41896|LLMRANDOM-41895|LLMRANDOM-41894|LLMRANDOM-41893|LLMRANDOM-41892|LLMRANDOM-41885|LLMRANDOM-41884|LLMRANDOM-41882|LLMRANDOM-41816|LLMRANDOM-41815|LLMRANDOM-41814|LLMRANDOM-41813|LLMRANDOM-41812|LLMRANDOM-41811|LLMRANDOM-41810|LLMRANDOM-41809|LLMRANDOM-41808|LLMRANDOM-41807|LLMRANDOM-41806)$" \
  -d "$DATA_DIR" \
  -f "$FILES_DIR"

echo "data_dir: $DATA_DIR"
echo "files_dir: $FILES_DIR"


# 2) summary (sanity check what was exported)
uv run neptune-exporter summary --data-path "$DATA_DIR"

# 3) load into W&B
uv run neptune-exporter load \
  --loader wandb \
  --wandb-entity $WANDB_ENTITY \
  --name-prefix $NAME_PREFIX \
  -d $DATA_DIR \
  -f $FILES_DIR

