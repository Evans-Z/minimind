#!/usr/bin/env bash
set -euo pipefail

# Comprehensive launcher for trainer/train_pretrain_scale.py
# Supports:
# - YAML size/context presets
# - DDP / FSDP2 backend
# - single-node torchrun defaults
# - resume toggle
# - arbitrary extra args passthrough
#
# Example:
# ./scripts/launch_pretrain_scale.sh \
#   --size-preset dense_1b \
#   --context-preset ctx_1024 \
#   --backend fsdp2 \
#   --nproc-per-node 8 \
#   --resume
#
# Any args after "--" are passed directly to train_pretrain_scale.py.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

SCRIPT_PATH="trainer/train_pretrain_scale.py"
TRAIN_CONFIG_PATH="configs/train_scale_config.yaml"
# Runtime defaults are intentionally empty: prefer train_config.yaml as source of truth.
YAML_PATH=""
SIZE_PRESET=""
CONTEXT_PRESET=""
BACKEND=""
SAVE_WEIGHT=""
SAVE_DIR=""
MODEL_PATH=""
DATA_PATH=""
TB_RUN_TAG=""
TENSORBOARD_LOGDIR=""
CUDA_VISIBLE_DEVICES_OPT=""

NPROC_PER_NODE=""
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-29500}"
NNODES="${NNODES:-1}"
NODE_RANK="${NODE_RANK:-0}"

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
emit("CONTEXT_PRESET", pick("presets", "context_preset"))

emit("BACKEND", pick("launch", "backend"))
emit("NPROC_PER_NODE", pick("launch", "nproc_per_node"))
emit("NNODES", pick("launch", "nnodes"))
emit("NODE_RANK", pick("launch", "node_rank"))
emit("MASTER_ADDR", pick("launch", "master_addr"))
emit("MASTER_PORT", pick("launch", "master_port"))
emit("CUDA_VISIBLE_DEVICES_OPT", pick("launch", "cuda_visible_devices"))
emit("DRY_RUN", pick("launch", "dry_run"))

emit("SAVE_WEIGHT", pick("run", "save_weight"))
emit("SAVE_DIR", pick("run", "save_dir"))
emit("MODEL_PATH", pick("run", "model_path"))
emit("DATA_PATH", pick("run", "data_path"))
emit("TB_RUN_TAG", pick("run", "tb_run_tag"))
emit("TENSORBOARD_LOGDIR", pick("run", "tensorboard_logdir"))
emit("RESUME", pick("run", "resume"))
emit("USE_WANDB", pick("run", "use_wandb"))
emit("USE_TENSORBOARD", pick("run", "use_tensorboard"))
emit("USE_COMPILE", pick("run", "use_compile"))

overrides = pick("train", "overrides", default={})
if not isinstance(overrides, dict):
    overrides = data.get("train_overrides", {})
if not isinstance(overrides, dict):
    raise SystemExit("train.overrides (or train_overrides) must be a mapping")

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
Usage: scripts/launch_pretrain_scale.sh [options] [-- extra_train_args...]

Options:
  --train-config <path>        Training launcher config YAML (default: configs/train_scale_config.yaml)
  --yaml <path>                Preset YAML path (default from train-config)
  --size-preset <name>         Size preset key (default from train-config)
  --context-preset <name>      Context preset key (default from train-config)
  --backend <ddp|fsdp2>        Distributed backend (default from train-config)
  --nproc-per-node <int>       Processes per node (auto-detect if omitted)
  --nnodes <int>               Number of nodes (default: 1)
  --node-rank <int>            Node rank (default: 0)
  --master-addr <addr>         Master address (default: 127.0.0.1)
  --master-port <port>         Master port (default: 29500)

  --save-weight <name>         Save weight prefix (default from train-config)
  --save-dir <path>            Save directory (default from train-config)
  --model-path <path>          Tokenizer/model path for init_model (default from train-config)
  --data-path <path>           Data file path (default from train-config)
  --tb-run-tag <tag>           TensorBoard run tag
  --tensorboard-logdir <path>  TensorBoard log directory
  --cuda-visible-devices <ids> Set CUDA_VISIBLE_DEVICES explicitly (e.g. 0,1,2,3)

  --resume                     Enable --from_resume 1
  --wandb                      Enable --use_wandb
  --tensorboard                Enable --use_tensorboard (default: enabled)
  --no-tensorboard             Disable TensorBoard
  --compile                    Enable --use_compile 1
  --dry-run                    Print final command only
  -h, --help                   Show this help message

Notes:
  1) Extra args after "--" are forwarded to trainer/train_pretrain_scale.py
  2) Launch from repo root or anywhere (script auto-cd to repo root)

Example:
  scripts/launch_pretrain_scale.sh \
    --size-preset dense_2b \
    --context-preset ctx_2048 \
    --backend fsdp2 \
    --nproc-per-node 8 \
    --resume \
    -- --learning_rate 2e-4 --batch_size 8 --accumulation_steps 16
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
    --context-preset) CONTEXT_PRESET="$2"; shift 2 ;;
    --backend) BACKEND="$2"; shift 2 ;;
    --nproc-per-node) NPROC_PER_NODE="$2"; shift 2 ;;
    --nnodes) NNODES="$2"; shift 2 ;;
    --node-rank) NODE_RANK="$2"; shift 2 ;;
    --master-addr) MASTER_ADDR="$2"; shift 2 ;;
    --master-port) MASTER_PORT="$2"; shift 2 ;;
    --save-weight) SAVE_WEIGHT="$2"; shift 2 ;;
    --save-dir) SAVE_DIR="$2"; shift 2 ;;
    --model-path) MODEL_PATH="$2"; shift 2 ;;
    --data-path) DATA_PATH="$2"; shift 2 ;;
    --tb-run-tag) TB_RUN_TAG="$2"; shift 2 ;;
    --tensorboard-logdir) TENSORBOARD_LOGDIR="$2"; shift 2 ;;
    --cuda-visible-devices) CUDA_VISIBLE_DEVICES_OPT="$2"; shift 2 ;;
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

required_vars=(YAML_PATH SIZE_PRESET CONTEXT_PRESET BACKEND SAVE_WEIGHT SAVE_DIR MODEL_PATH DATA_PATH)
missing=()
for var_name in "${required_vars[@]}"; do
  if [[ -z "${!var_name}" ]]; then
    missing+=("$var_name")
  fi
done
if [[ "${#missing[@]}" -gt 0 ]]; then
  echo "Missing required launcher settings: ${missing[*]}" >&2
  echo "Set them in --train-config (recommended) or pass explicit CLI options." >&2
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
print(len([x for x in v.split(",") if x.strip() != ""]))
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

TRAIN_ARGS=(
  "$SCRIPT_PATH"
  --model_config_yaml "$YAML_PATH"
  --size_preset "$SIZE_PRESET"
  --context_preset "$CONTEXT_PRESET"
  --dist_backend "$BACKEND"
  --save_weight "$SAVE_WEIGHT"
  --save_dir "$SAVE_DIR"
  --model_path "$MODEL_PATH"
  --data_path "$DATA_PATH"
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

echo "== Launch Summary =="
echo "root_dir:           $ROOT_DIR"
echo "script:             $SCRIPT_PATH"
echo "train_config:       $TRAIN_CONFIG_PATH"
echo "backend:            $BACKEND"
echo "yaml:               $YAML_PATH"
echo "size_preset:        $SIZE_PRESET"
echo "context_preset:     $CONTEXT_PRESET"
echo "nproc_per_node:     $NPROC_PER_NODE"
echo "nnodes/node_rank:   $NNODES/$NODE_RANK"
echo "master:             $MASTER_ADDR:$MASTER_PORT"
echo "resume:             $RESUME"
echo "wandb:              $USE_WANDB"
echo "tensorboard:        $USE_TENSORBOARD"
echo "tb_logdir:          ${TENSORBOARD_LOGDIR:-<trainer default>}"
echo "compile:            $USE_COMPILE"
echo "cuda_visible:       ${CUDA_VISIBLE_DEVICES:-<not set>}"
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
