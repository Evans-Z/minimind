#!/usr/bin/env bash
set -euo pipefail

# FineWeb-Edu 100BT packed preprocessing launcher.
# Tune NUM_SHARDS/BATCH_SIZE if CPU, memory, or disk I/O becomes the bottleneck.

INPUT_DIR="${INPUT_DIR:-/data/share/lixing/mhc/data/fineweb-edu/sample-100BT}"
OUTPUT_DIR="${OUTPUT_DIR:-/data/share/lixing/mhc/data/fineweb-edu-packed-100BT}"
TOKENIZER_PATH="${TOKENIZER_PATH:-model}"
LOG_DIR="${LOG_DIR:-logs/fineweb_preprocess_100bt}"

SEQ_LEN="${SEQ_LEN:-768}"
BATCH_SIZE="${BATCH_SIZE:-2048}"
NUM_SHARDS="${NUM_SHARDS:-48}"
TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

mkdir -p "${LOG_DIR}" "${OUTPUT_DIR}"

echo "input_dir:               ${INPUT_DIR}"
echo "output_dir:              ${OUTPUT_DIR}"
echo "tokenizer_path:          ${TOKENIZER_PATH}"
echo "seq_len:                 ${SEQ_LEN}"
echo "batch_size:              ${BATCH_SIZE}"
echo "num_shards:              ${NUM_SHARDS}"
echo "tokenizers_parallelism:  ${TOKENIZERS_PARALLELISM}"
echo "log_dir:                 ${LOG_DIR}"
echo

for shard_index in $(seq 0 "$((NUM_SHARDS - 1))"); do
  log_file="${LOG_DIR}/shard_${shard_index}.log"
  echo "starting shard ${shard_index}/${NUM_SHARDS}, log: ${log_file}"
  python scripts/preprocess_fineweb_edu.py \
    --input-dir "${INPUT_DIR}" \
    --output-dir "${OUTPUT_DIR}" \
    --tokenizer-path "${TOKENIZER_PATH}" \
    --seq-len "${SEQ_LEN}" \
    --batch-size "${BATCH_SIZE}" \
    --num-shards "${NUM_SHARDS}" \
    --shard-index "${shard_index}" \
    --tokenizers-parallelism "${TOKENIZERS_PARALLELISM}" \
    --no-progress-bar \
    > "${log_file}" 2>&1 &
done

echo
echo "All shards launched. Waiting for completion..."
wait
echo "FineWeb-Edu preprocessing complete: ${OUTPUT_DIR}"
