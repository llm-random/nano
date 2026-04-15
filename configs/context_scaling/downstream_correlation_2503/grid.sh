#!/bin/bash
# Grid: MQA × {k12, k16, k20} × {seq_len 1024, 2048, 4096} @ 40B tokens
# + compute-optimal runs for k12 & k16
#
# tokens_per_batch = seq_len * batch_size = 131072 (constant)
# seq_len 1024 → batch_size 128
# seq_len 2048 → batch_size 64
# seq_len 4096 → batch_size 32

DIR="context_scaling/downstream_correlation_2503"
N_STEPS_OPT_K12=100001
N_STEPS_OPT_K16=190001

# --- 40B token runs: k12, k16, k20 × seq_len 1024, 2048, 4096 ---

# k12
python run_exp.py --config-path configs --config-name $DIR/k12 \
  common.sequence_length=1024 common.batch_size=128

python run_exp.py --config-path configs --config-name $DIR/k12 \
  common.sequence_length=2048 common.batch_size=64

python run_exp.py --config-path configs --config-name $DIR/k12 \
  common.sequence_length=4096 common.batch_size=32

# k16
python run_exp.py --config-path configs --config-name $DIR/k16 \
  common.sequence_length=1024 common.batch_size=128

python run_exp.py --config-path configs --config-name $DIR/k16 \
  common.sequence_length=2048 common.batch_size=64

python run_exp.py --config-path configs --config-name $DIR/k16 \
  common.sequence_length=4096 common.batch_size=32

# k20
python run_exp.py --config-path configs --config-name $DIR/k20 \
  common.sequence_length=1024 common.batch_size=128

python run_exp.py --config-path configs --config-name $DIR/k20 \
  common.sequence_length=2048 common.batch_size=64

python run_exp.py --config-path configs --config-name $DIR/k20 \
  common.sequence_length=4096 common.batch_size=32

# --- Compute-optimal runs: k12 & k16 ---

python run_exp.py --config-path configs --config-name $DIR/k12 \
  trainer.n_steps=$N_STEPS_OPT_K12 trainer.lm_eval_interval=10000

python run_exp.py --config-path configs --config-name $DIR/k16 \
  trainer.n_steps=$N_STEPS_OPT_K16 trainer.lm_eval_interval=19000
