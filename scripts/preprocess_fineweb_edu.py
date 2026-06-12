#!/usr/bin/env python3
import argparse
import json
import os
import time

import numpy as np
from datasets import load_dataset, load_from_disk
from transformers import AutoTokenizer


DATASET_NAME = "HuggingFaceFW/fineweb-edu"


def normalize_sample(sample: str) -> str:
    sample = sample.strip()
    if sample.startswith("sample-"):
        return sample
    return f"sample-{sample}"


def token_dtype(vocab_size: int):
    return np.uint16 if vocab_size <= np.iinfo(np.uint16).max else np.uint32


def load_fineweb(args):
    if args.input_dir:
        return load_from_disk(args.input_dir)
    return load_dataset(
        DATASET_NAME,
        name=normalize_sample(args.sample),
        split=args.split,
        cache_dir=args.cache_dir,
        num_proc=None if args.streaming else args.num_proc,
        streaming=args.streaming,
    )


def format_int(value: int) -> str:
    return f"{value:,}"


def format_duration(seconds: float) -> str:
    seconds = max(int(seconds), 0)
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours > 0:
        return f"{hours:d}h{minutes:02d}m{secs:02d}s"
    if minutes > 0:
        return f"{minutes:d}m{secs:02d}s"
    return f"{secs:d}s"


def output_paths(output_dir: str, num_shards: int, shard_index: int):
    if num_shards == 1:
        return os.path.join(output_dir, "train.bin"), os.path.join(output_dir, "meta.json")
    shard_name = f"{shard_index:05d}_of_{num_shards:05d}"
    return os.path.join(output_dir, f"train_{shard_name}.bin"), os.path.join(output_dir, f"meta_{shard_name}.json")


def main():
    parser = argparse.ArgumentParser(
        description="Tokenize and pack FineWeb-Edu into fixed-length MiniMind pretraining blocks.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--sample", type=str, default="10BT", choices=["10BT", "100BT", "sample-10BT", "sample-100BT"])
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--cache-dir", type=str, default=os.path.join("dataset", "hf_cache"))
    parser.add_argument("--input-dir", type=str, default="", help="Optional datasets.save_to_disk directory to read instead of HF.")
    parser.add_argument("--output-dir", type=str, default=os.path.join("dataset", "fineweb_edu_packed"))
    parser.add_argument("--tokenizer-path", type=str, default="model")
    parser.add_argument("--seq-len", type=int, default=768, help="Packed training sequence length.")
    parser.add_argument("--text-field", type=str, default="text")
    parser.add_argument("--num-proc", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=1024, help="Documents per tokenizer batch.")
    parser.add_argument("--num-shards", type=int, default=1, help="Total preprocessing shards.")
    parser.add_argument("--shard-index", type=int, default=0, help="Current shard index, from 0 to num_shards-1.")
    parser.add_argument("--streaming", action="store_true", help="Stream from HF instead of materializing the dataset first.")
    parser.add_argument("--max-docs", type=int, default=0, help="Stop after N documents; 0 means all.")
    parser.add_argument("--drop-remainder", action="store_true", help="Drop the final partial packed sequence.")
    parser.add_argument("--progress-interval", type=int, default=10000)
    parser.add_argument("--no-progress-bar", action="store_true", help="Disable single-line progress bar output.")
    parser.add_argument(
        "--tokenizers-parallelism",
        type=str,
        default="true",
        choices=["true", "false"],
        help="Set TOKENIZERS_PARALLELISM for fast tokenizer batch encoding.",
    )
    args = parser.parse_args()

    if args.seq_len < 2:
        raise ValueError("--seq-len must be at least 2")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be at least 1")
    if args.num_shards < 1:
        raise ValueError("--num-shards must be at least 1")
    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError("--shard-index must be in [0, num_shards)")
    if args.input_dir and args.streaming:
        raise ValueError("--input-dir cannot be combined with --streaming")

    os.environ["TOKENIZERS_PARALLELISM"] = args.tokenizers_parallelism
    os.makedirs(args.output_dir, exist_ok=True)
    bin_path, meta_path = output_paths(args.output_dir, args.num_shards, args.shard_index)
    if os.path.exists(bin_path) or os.path.exists(meta_path):
        raise FileExistsError(f"{bin_path} or {meta_path} already exists")

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    bos_id = tokenizer.bos_token_id
    eos_id = tokenizer.eos_token_id
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else eos_id
    if bos_id is None or eos_id is None:
        raise ValueError("Tokenizer must define bos_token_id and eos_token_id")

    dtype = token_dtype(len(tokenizer))
    ds = load_fineweb(args)
    if args.num_shards > 1:
        ds = ds.shard(num_shards=args.num_shards, index=args.shard_index, contiguous=True)
    try:
        target_docs = len(ds)
    except TypeError:
        target_docs = 0
    if args.max_docs > 0:
        target_docs = min(target_docs, args.max_docs) if target_docs > 0 else args.max_docs

    total_docs = 0
    total_tokens = 0
    total_sequences = 0
    buffer = []
    start_time = time.time()

    def print_progress(force_newline=False):
        elapsed = max(time.time() - start_time, 1e-9)
        docs_per_sec = total_docs / elapsed
        tokens_per_sec = total_tokens / elapsed
        prefix = f"shard={args.shard_index}/{args.num_shards}"
        stats = (
            f"{prefix} docs={format_int(total_docs)} sequences={format_int(total_sequences)} "
            f"tokens={format_int(total_tokens)} docs/s={docs_per_sec:.1f} tokens/s={tokens_per_sec:.1f}"
        )
        if target_docs > 0:
            progress = min(total_docs / target_docs, 1.0)
            eta = (target_docs - total_docs) / max(docs_per_sec, 1e-9)
            bar_width = 30
            filled = int(bar_width * progress)
            bar = "#" * filled + "-" * (bar_width - filled)
            stats = f"[{bar}] {progress * 100:6.2f}% eta={format_duration(eta)} {stats}"
        else:
            stats = f"elapsed={format_duration(elapsed)} eta=unknown {stats}"

        if args.no_progress_bar:
            print(stats, flush=True)
        else:
            print("\r" + stats, end="\n" if force_newline else "", flush=True)

    def write_token_batch(texts, out):
        nonlocal total_docs, total_tokens, total_sequences, buffer
        encoded = tokenizer(texts, add_special_tokens=False).input_ids
        for doc_tokens in encoded:
            if not doc_tokens:
                continue

            buffer.extend([bos_id])
            buffer.extend(doc_tokens)
            buffer.extend([eos_id])
            total_docs += 1
            total_tokens += len(doc_tokens) + 2

            while len(buffer) >= args.seq_len:
                block = np.asarray(buffer[: args.seq_len], dtype=dtype)
                block.tofile(out)
                del buffer[: args.seq_len]
                total_sequences += 1

            if args.progress_interval > 0 and total_docs % args.progress_interval == 0:
                print_progress()

    with open(bin_path, "wb") as out:
        batch_texts = []
        for row in ds:
            if args.max_docs > 0 and total_docs + len(batch_texts) >= args.max_docs:
                break
            text = row.get(args.text_field)
            if text is None:
                continue

            batch_texts.append(str(text))
            if len(batch_texts) >= args.batch_size:
                write_token_batch(batch_texts, out)
                batch_texts = []

        if batch_texts:
            write_token_batch(batch_texts, out)

        if buffer and not args.drop_remainder:
            padded = buffer + [pad_id] * (args.seq_len - len(buffer))
            np.asarray(padded, dtype=dtype).tofile(out)
            total_sequences += 1

    if args.progress_interval > 0:
        print_progress(force_newline=True)

    meta = {
        "format": "minimind_packed_pretrain",
        "source": args.input_dir or DATASET_NAME,
        "sample": normalize_sample(args.sample) if not args.input_dir else "",
        "split": args.split,
        "tokenizer_path": args.tokenizer_path,
        "seq_len": args.seq_len,
        "dtype": np.dtype(dtype).name,
        "bin_file": os.path.basename(bin_path),
        "num_sequences": total_sequences,
        "num_tokens": total_sequences * args.seq_len,
        "docs_read": total_docs,
        "num_shards": args.num_shards,
        "shard_index": args.shard_index,
        "text_field": args.text_field,
        "bos_token_id": bos_id,
        "eos_token_id": eos_id,
        "pad_token_id": pad_id,
        "drop_remainder": args.drop_remainder,
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
        f.write("\n")

    elapsed = time.time() - start_time
    print("\n== FineWeb-Edu Packed Dataset ==")
    print(f"output_dir:     {args.output_dir}")
    print(f"bin_file:       {os.path.basename(bin_path)}")
    print(f"shard:          {args.shard_index}/{args.num_shards}")
    print(f"docs_read:      {format_int(total_docs)}")
    print(f"sequences:      {format_int(total_sequences)}")
    print(f"tokens_written: {format_int(total_sequences * args.seq_len)}")
    print(f"dtype:          {np.dtype(dtype).name}")
    print(f"elapsed_sec:    {elapsed:.1f}")


if __name__ == "__main__":
    main()
