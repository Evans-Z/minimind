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
    parser.add_argument("--streaming", action="store_true", help="Stream from HF instead of materializing the dataset first.")
    parser.add_argument("--max-docs", type=int, default=0, help="Stop after N documents; 0 means all.")
    parser.add_argument("--drop-remainder", action="store_true", help="Drop the final partial packed sequence.")
    parser.add_argument("--progress-interval", type=int, default=10000)
    args = parser.parse_args()

    if args.seq_len < 2:
        raise ValueError("--seq-len must be at least 2")
    if args.input_dir and args.streaming:
        raise ValueError("--input-dir cannot be combined with --streaming")

    os.makedirs(args.output_dir, exist_ok=True)
    bin_path = os.path.join(args.output_dir, "train.bin")
    meta_path = os.path.join(args.output_dir, "meta.json")
    if os.path.exists(bin_path) or os.path.exists(meta_path):
        raise FileExistsError(f"{args.output_dir} already contains train.bin or meta.json")

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    bos_id = tokenizer.bos_token_id
    eos_id = tokenizer.eos_token_id
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else eos_id
    if bos_id is None or eos_id is None:
        raise ValueError("Tokenizer must define bos_token_id and eos_token_id")

    dtype = token_dtype(len(tokenizer))
    ds = load_fineweb(args)

    total_docs = 0
    total_tokens = 0
    total_sequences = 0
    buffer = []
    start_time = time.time()

    with open(bin_path, "wb") as out:
        for row in ds:
            if args.max_docs > 0 and total_docs >= args.max_docs:
                break
            text = row.get(args.text_field)
            if text is None:
                continue

            doc_tokens = tokenizer(str(text), add_special_tokens=False).input_ids
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
                elapsed = max(time.time() - start_time, 1e-9)
                print(
                    f"docs={format_int(total_docs)} sequences={format_int(total_sequences)} "
                    f"tokens={format_int(total_tokens)} docs/s={total_docs / elapsed:.1f}",
                    flush=True,
                )

        if buffer and not args.drop_remainder:
            padded = buffer + [pad_id] * (args.seq_len - len(buffer))
            np.asarray(padded, dtype=dtype).tofile(out)
            total_sequences += 1

    meta = {
        "format": "minimind_packed_pretrain",
        "source": args.input_dir or DATASET_NAME,
        "sample": normalize_sample(args.sample) if not args.input_dir else "",
        "split": args.split,
        "tokenizer_path": args.tokenizer_path,
        "seq_len": args.seq_len,
        "dtype": np.dtype(dtype).name,
        "num_sequences": total_sequences,
        "num_tokens": total_sequences * args.seq_len,
        "docs_read": total_docs,
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
    print(f"docs_read:      {format_int(total_docs)}")
    print(f"sequences:      {format_int(total_sequences)}")
    print(f"tokens_written: {format_int(total_sequences * args.seq_len)}")
    print(f"dtype:          {np.dtype(dtype).name}")
    print(f"elapsed_sec:    {elapsed:.1f}")


if __name__ == "__main__":
    main()
