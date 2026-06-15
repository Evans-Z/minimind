#!/usr/bin/env bash
set -euo pipefail

# FineWeb-Edu packed-data launcher for trainer/train_pretrain_scale.py.
# This follows scripts/launch_pretrain_scale.sh, but defaults are tuned for the
# preprocessed FineWeb-Edu packed directory and explicit max_seq_len matching.
#
# Example:
#   scripts/launch_fineweb_pretrain_scale.sh \
#     --size-preset mhc_1b_balm \
#     --backend fsdp2 \
#     --nproc-per-node 8 \
#     -- --learning_rate 2e-4 --batch_size 8 --accumulation_steps 16
#
# Extra args after "--" are passed directly to trainer/train_pretrain_scale.py.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

SCRIPT_PATH="trainer/train_pretrain_scale.py"
TRAIN_CONFIG_PATH="${TRAIN_CONFIG_PATH:-configs/train_fineweb_config.yaml}"
YAML_PATH="${YAML_PATH:-configs/model_scale_presets.yaml}"
SIZE_PRESET="${SIZE_PRESET:-dense_64m}"
BACKEND="${BACKEND:-fsdp2}"

DATA_PATH="${DATA_PATH:-/data/share/lixing/mhc/data/fineweb-edu-packed-100BT}"
MODEL_PATH="${MODEL_PATH:-model}"
SAVE_DIR="${SAVE_DIR:-/data/share/lixing/mhc/out}"
SAVE_WEIGHT="${SAVE_WEIGHT:-pretrain_fineweb100bt}"
TENSORBOARD_LOGDIR="${TENSORBOARD_LOGDIR:-runs/pretrain_fineweb100bt}"
TB_RUN_TAG="${TB_RUN_TAG:-fineweb100bt}"

MAX_SEQ_LEN="${MAX_SEQ_LEN:-768}"
EPOCHS="${EPOCHS:-1}"
BATCH_SIZE="${BATCH_SIZE:-32}"
ACCUMULATION_STEPS="${ACCUMULATION_STEPS:-8}"
LEARNING_RATE="${LEARNING_RATE:-5e-4}"
WARMUP_RATIO="${WARMUP_RATIO:-0.01}"
MIN_LR_RATIO="${MIN_LR_RATIO:-0.1}"
NUM_WORKERS="${NUM_WORKERS:-8}"
LOG_INTERVAL="${LOG_INTERVAL:-100}"
SAVE_INTERVAL="${SAVE_INTERVAL:-1000}"
DTYPE="${DTYPE:-bfloat16}"
FSDP2_RESHARD_AFTER_FORWARD="${FSDP2_RESHARD_AFTER_FORWARD:-1}"

NPROC_PER_NODE="${NPROC_PER_NODE:-}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-29500}"
NNODES="${NNODES:-1}"
NODE_RANK="${NODE_RANK:-0}"
CUDA_VISIBLE_DEVICES_OPT="${CUDA_VISIBLE_DEVICES_OPT:-}"
DIST_TIMEOUT_SECONDS="${DIST_TIMEOUT_SECONDS:-1800}"

RESUME=0
USE_WANDB=0
USE_TENSORBOARD=1
USE_COMPILE=0
DRY_RUN=0
TRAIN_CONFIG_EXTRA_ARGS=()

load_train_config_if_exists() {
  local cfg_path="$1"
  if [[ ! -f "$cfg_path" ]]; then
    return 0
  fi
  local rendered
  rendered="$(python - "$cfg_path" <<'PY'
import shlex
import sys

cfg_path = sys.argv[1]
try:
    import yaml
except Exception as e:
    raise SystemExit(f"PyYAML is required to load train config: {e}")

with open(cfg_path, "r", encoding="utf-8") as f:
    data = yaml.safe_load(f) or {}

def pick(section, key, default=None):
    sec = data.get(section, {})
    if isinstance(sec, dict) and key in sec:
        return sec[key]
    return data.get(key, default)

def emit(name, value):
    if value is None:
        return
    if isinstance(value, bool):
        value = "1" if value else "0"
    print(f"{name}={shlex.quote(str(value))}")

emit("YAML_PATH", pick("presets", "yaml_path"))
emit("SIZE_PRESET", pick("presets", "size_preset"))

emit("BACKEND", pick("launch", "backend"))
emit("NPROC_PER_NODE", pick("launch", "nproc_per_node"))
emit("NNODES", pick("launch", "nnodes"))
emit("NODE_RANK", pick("launch", "node_rank"))
emit("MASTER_ADDR", pick("launch", "master_addr"))
emit("MASTER_PORT", pick("launch", "master_port"))
emit("CUDA_VISIBLE_DEVICES_OPT", pick("launch", "cuda_visible_devices"))
emit("DIST_TIMEOUT_SECONDS", pick("launch", "dist_timeout_seconds"))
emit("DRY_RUN", pick("launch", "dry_run"))

emit("DATA_PATH", pick("run", "data_path"))
emit("MODEL_PATH", pick("run", "model_path"))
emit("SAVE_DIR", pick("run", "save_dir"))
emit("SAVE_WEIGHT", pick("run", "save_weight"))
emit("TENSORBOARD_LOGDIR", pick("run", "tensorboard_logdir"))
emit("TB_RUN_TAG", pick("run", "tb_run_tag"))
emit("RESUME", pick("run", "resume"))
emit("USE_WANDB", pick("run", "use_wandb"))
emit("USE_TENSORBOARD", pick("run", "use_tensorboard"))
emit("USE_COMPILE", pick("run", "use_compile"))

emit("MAX_SEQ_LEN", pick("train", "max_seq_len"))
emit("EPOCHS", pick("train", "epochs"))
emit("BATCH_SIZE", pick("train", "batch_size"))
emit("ACCUMULATION_STEPS", pick("train", "accumulation_steps"))
emit("LEARNING_RATE", pick("train", "learning_rate"))
emit("WARMUP_RATIO", pick("train", "warmup_ratio"))
emit("MIN_LR_RATIO", pick("train", "min_lr_ratio"))
emit("NUM_WORKERS", pick("train", "num_workers"))
emit("LOG_INTERVAL", pick("train", "log_interval"))
emit("SAVE_INTERVAL", pick("train", "save_interval"))
emit("DTYPE", pick("train", "dtype"))
emit("FSDP2_RESHARD_AFTER_FORWARD", pick("train", "fsdp2_reshard_after_forward"))

overrides = pick("train", "overrides", default={})
if overrides is None:
    overrides = {}
if not isinstance(overrides, dict):
    raise SystemExit("train.overrides must be a mapping")

extra_args = []
for k, v in overrides.items():
    flag = f"--{k}"
    if isinstance(v, bool):
        if v:
            extra_args.append(flag)
    elif isinstance(v, list):
        for item in v:
            extra_args.extend([flag, str(item)])
    else:
        extra_args.extend([flag, str(v)])
array_expr = " ".join(shlex.quote(x) for x in extra_args)
print(f"TRAIN_CONFIG_EXTRA_ARGS=({array_expr})")
PY
)"
  eval "$rendered"
}

print_help() {
  cat <<'EOF'
Usage: scripts/launch_fineweb_pretrain_scale.sh [options] [-- extra_train_args...]

Options:
  --train-config <path>        FineWeb training config YAML
  --yaml <path>                Model preset YAML path
  --size-preset <name>         Size preset key, e.g. dense_64m, dense_1b, mhc_1b_balm
  --backend <ddp|fsdp2>        Distributed backend
  --nproc-per-node <int>       Processes per node; auto-detect if omitted
  --nnodes <int>               Number of nodes
  --node-rank <int>            Node rank
  --master-addr <addr>         Master address
  --master-port <port>         Master port
  --dist-timeout-seconds <int> Distributed process-group timeout
  --cuda-visible-devices <ids> Set CUDA_VISIBLE_DEVICES, e.g. 0,1,2,3

  --data-path <path>           FineWeb packed directory
  --model-path <path>          Tokenizer/model path
  --save-dir <path>            Save directory
  --save-weight <name>         Save weight prefix
  --max-seq-len <int>          Must match preprocessing --seq-len
  --epochs <int>
  --batch-size <int>
  --accumulation-steps <int>
  --learning-rate <float>
  --warmup-ratio <float>
  --min-lr-ratio <float>
  --num-workers <int>
  --log-interval <int>
  --save-interval <int>
  --dtype <bfloat16|float16>
  --tb-run-tag <tag>
  --tensorboard-logdir <path>

  --resume                     Enable --from_resume 1
  --wandb                      Enable --use_wandb
  --tensorboard                Enable TensorBoard (default)
  --no-tensorboard             Disable TensorBoard
  --compile                    Enable torch.compile
  --dry-run                    Print command only
  -h, --help                   Show this help

Notes:
  1) This script intentionally does not pass --context_preset because context
     presets override --max_seq_len inside train_pretrain_scale.py.
  2) Extra args after "--" are forwarded directly to train_pretrain_scale.py.
EOF
}

# Pre-scan --train-config so YAML defaults load before normal CLI parse.
for ((i = 1; i <= $#; i++)); do
  if [[ "${!i}" == "--train-config" ]]; then
    next_i=$((i + 1))
    if [[ $next_i -le $# ]]; then
      TRAIN_CONFIG_PATH="${!next_i}"
    fi
  fi
done
load_train_config_if_exists "$TRAIN_CONFIG_PATH"

EXTRA_ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --train-config) TRAIN_CONFIG_PATH="$2"; shift 2 ;;
    --yaml) YAML_PATH="$2"; shift 2 ;;
    --size-preset) SIZE_PRESET="$2"; shift 2 ;;
    --backend) BACKEND="$2"; shift 2 ;;
    --nproc-per-node) NPROC_PER_NODE="$2"; shift 2 ;;
    --nnodes) NNODES="$2"; shift 2 ;;
    --node-rank) NODE_RANK="$2"; shift 2 ;;
    --master-addr) MASTER_ADDR="$2"; shift 2 ;;
    --master-port) MASTER_PORT="$2"; shift 2 ;;
    --dist-timeout-seconds) DIST_TIMEOUT_SECONDS="$2"; shift 2 ;;
    --cuda-visible-devices) CUDA_VISIBLE_DEVICES_OPT="$2"; shift 2 ;;
    --data-path) DATA_PATH="$2"; shift 2 ;;
    --model-path) MODEL_PATH="$2"; shift 2 ;;
    --save-dir) SAVE_DIR="$2"; shift 2 ;;
    --save-weight) SAVE_WEIGHT="$2"; shift 2 ;;
    --max-seq-len) MAX_SEQ_LEN="$2"; shift 2 ;;
    --epochs) EPOCHS="$2"; shift 2 ;;
    --batch-size) BATCH_SIZE="$2"; shift 2 ;;
    --accumulation-steps) ACCUMULATION_STEPS="$2"; shift 2 ;;
    --learning-rate) LEARNING_RATE="$2"; shift 2 ;;
    --warmup-ratio) WARMUP_RATIO="$2"; shift 2 ;;
    --min-lr-ratio) MIN_LR_RATIO="$2"; shift 2 ;;
    --num-workers) NUM_WORKERS="$2"; shift 2 ;;
    --log-interval) LOG_INTERVAL="$2"; shift 2 ;;
    --save-interval) SAVE_INTERVAL="$2"; shift 2 ;;
    --dtype) DTYPE="$2"; shift 2 ;;
    --tb-run-tag) TB_RUN_TAG="$2"; shift 2 ;;
    --tensorboard-logdir) TENSORBOARD_LOGDIR="$2"; shift 2 ;;
    --resume) RESUME=1; shift ;;
    --wandb) USE_WANDB=1; shift ;;
    --tensorboard) USE_TENSORBOARD=1; shift ;;
    --no-tensorboard) USE_TENSORBOARD=0; shift ;;
    --compile) USE_COMPILE=1; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) print_help; exit 0 ;;
    --)
      shift
      EXTRA_ARGS+=("$@")
      break
      ;;
    *)
      echo "Unknown option: $1" >&2
      echo "Use --help for supported arguments." >&2
      exit 1
      ;;
  esac
done

if [[ "$BACKEND" != "ddp" && "$BACKEND" != "fsdp2" ]]; then
  echo "Invalid --backend: $BACKEND (expected ddp or fsdp2)" >&2
  exit 1
fi

if [[ -n "$CUDA_VISIBLE_DEVICES_OPT" ]]; then
  export CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES_OPT"
fi

if [[ -z "$NPROC_PER_NODE" ]]; then
  if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    NPROC_PER_NODE="$(python - <<'PY'
import os
v = os.environ.get("CUDA_VISIBLE_DEVICES", "")
print(len([x for x in v.split(",") if x.strip()]))
PY
)"
  else
    NPROC_PER_NODE="$(python - <<'PY'
import torch
print(torch.cuda.device_count() if torch.cuda.is_available() else 1)
PY
)"
  fi
fi

if [[ "$NPROC_PER_NODE" -lt 1 ]]; then
  echo "nproc_per_node must be >= 1, got: $NPROC_PER_NODE" >&2
  exit 1
fi

if [[ "$NNODES" -eq 1 ]]; then
  export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
  export NCCL_NVLS_ENABLE="${NCCL_NVLS_ENABLE:-0}"
  export NCCL_MNNVL_ENABLE="${NCCL_MNNVL_ENABLE:-0}"
fi
export MINIMIND_DIST_TIMEOUT_SECONDS="$DIST_TIMEOUT_SECONDS"

TRAIN_ARGS=(
  "$SCRIPT_PATH"
  --model_config_yaml "$YAML_PATH"
  --size_preset "$SIZE_PRESET"
  --dist_backend "$BACKEND"
  --save_weight "$SAVE_WEIGHT"
  --save_dir "$SAVE_DIR"
  --model_path "$MODEL_PATH"
  --data_path "$DATA_PATH"
  --max_seq_len "$MAX_SEQ_LEN"
  --epochs "$EPOCHS"
  --batch_size "$BATCH_SIZE"
  --accumulation_steps "$ACCUMULATION_STEPS"
  --learning_rate "$LEARNING_RATE"
  --warmup_ratio "$WARMUP_RATIO"
  --min_lr_ratio "$MIN_LR_RATIO"
  --num_workers "$NUM_WORKERS"
  --log_interval "$LOG_INTERVAL"
  --save_interval "$SAVE_INTERVAL"
  --dtype "$DTYPE"
  --fsdp2_reshard_after_forward "$FSDP2_RESHARD_AFTER_FORWARD"
)

if [[ "${#TRAIN_CONFIG_EXTRA_ARGS[@]}" -gt 0 ]]; then
  TRAIN_ARGS+=("${TRAIN_CONFIG_EXTRA_ARGS[@]}")
fi

if [[ "$RESUME" -eq 1 ]]; then
  TRAIN_ARGS+=(--from_resume 1)
fi
if [[ "$USE_WANDB" -eq 1 ]]; then
  TRAIN_ARGS+=(--use_wandb)
fi
if [[ "$USE_TENSORBOARD" -eq 1 ]]; then
  TRAIN_ARGS+=(--use_tensorboard)
fi
if [[ -n "$TB_RUN_TAG" ]]; then
  TRAIN_ARGS+=(--tb_run_tag "$TB_RUN_TAG")
fi
if [[ -n "$TENSORBOARD_LOGDIR" ]]; then
  TRAIN_ARGS+=(--tensorboard_logdir "$TENSORBOARD_LOGDIR")
fi
if [[ "$USE_COMPILE" -eq 1 ]]; then
  TRAIN_ARGS+=(--use_compile 1)
fi
if [[ "${#EXTRA_ARGS[@]}" -gt 0 ]]; then
  TRAIN_ARGS+=("${EXTRA_ARGS[@]}")
fi

TORCHRUN_CMD=(
  torchrun
  --nproc_per_node "$NPROC_PER_NODE"
  --nnodes "$NNODES"
  --node_rank "$NODE_RANK"
  --master_addr "$MASTER_ADDR"
  --master_port "$MASTER_PORT"
  "${TRAIN_ARGS[@]}"
)

echo "== FineWeb Pretrain Launch Summary =="
echo "root_dir:           $ROOT_DIR"
echo "script:             $SCRIPT_PATH"
echo "train_config:       $TRAIN_CONFIG_PATH"
echo "backend:            $BACKEND"
echo "yaml:               $YAML_PATH"
echo "size_preset:        $SIZE_PRESET"
echo "max_seq_len:        $MAX_SEQ_LEN"
echo "nproc_per_node:     $NPROC_PER_NODE"
echo "nnodes/node_rank:   $NNODES/$NODE_RANK"
echo "master:             $MASTER_ADDR:$MASTER_PORT"
echo "dist_timeout_sec:   $MINIMIND_DIST_TIMEOUT_SECONDS"
echo "resume:             $RESUME"
echo "wandb:              $USE_WANDB"
echo "tensorboard:        $USE_TENSORBOARD"
echo "tb_logdir:          ${TENSORBOARD_LOGDIR:-<trainer default>}"
echo "compile:            $USE_COMPILE"
echo "cuda_visible:       ${CUDA_VISIBLE_DEVICES:-<not set>}"
echo "NCCL_IB_DISABLE:    ${NCCL_IB_DISABLE:-<not set>}"
echo "NCCL_NVLS_ENABLE:   ${NCCL_NVLS_ENABLE:-<not set>}"
echo "NCCL_MNNVL_ENABLE:  ${NCCL_MNNVL_ENABLE:-<not set>}"
echo "save_weight:        $SAVE_WEIGHT"
echo "save_dir:           $SAVE_DIR"
echo "model_path:         $MODEL_PATH"
echo "data_path:          $DATA_PATH"
echo
echo "Command:"
printf ' %q' "${TORCHRUN_CMD[@]}"
echo

if [[ "$DRY_RUN" -eq 1 ]]; then
  echo "[dry-run] Exit without launching."
  exit 0
fi

exec "${TORCHRUN_CMD[@]}"
