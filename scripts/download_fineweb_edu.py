#!/usr/bin/env python3
import argparse
import os

from datasets import load_dataset


DATASET_NAME = "HuggingFaceFW/fineweb-edu"


def normalize_sample(sample: str) -> str:
    sample = sample.strip()
    if sample.startswith("sample-"):
        return sample
    return f"sample-{sample}"


def main():
    parser = argparse.ArgumentParser(
        description="Download FineWeb-Edu samples with HuggingFace datasets.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--sample",
        type=str,
        choices=["10BT", "100BT", "sample-10BT", "sample-100BT"],
        default="10BT",
        help="FineWeb-Edu sample config to download.",
    )
    parser.add_argument("--split", type=str, default="train", help="Dataset split to load.")
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=os.path.join("dataset", "hf_cache"),
        help="HuggingFace datasets cache directory.",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default="",
        help="Optional directory for datasets.save_to_disk output. If empty, only the HF cache is populated.",
    )
    parser.add_argument("--num-proc", type=int, default=8, help="Number of processes for dataset preparation.")
    parser.add_argument(
        "--streaming",
        action="store_true",
        help="Use streaming mode for a quick connectivity/preview check without downloading the full sample.",
    )
    parser.add_argument(
        "--preview-rows",
        type=int,
        default=0,
        help="Print the first N rows after loading. Useful with --streaming.",
    )
    args = parser.parse_args()

    config_name = normalize_sample(args.sample)
    print(f"dataset:   {DATASET_NAME}")
    print(f"config:    {config_name}")
    print(f"split:     {args.split}")
    print(f"cache_dir: {args.cache_dir}")

    ds = load_dataset(
        DATASET_NAME,
        name=config_name,
        split=args.split,
        cache_dir=args.cache_dir,
        num_proc=None if args.streaming else args.num_proc,
        streaming=args.streaming,
    )

    print(ds)

    if args.preview_rows > 0:
        for i, row in enumerate(ds):
            text = str(row.get("text", ""))
            print(f"\n== Row {i} ==")
            print(text[:500].replace("\n", "\\n"))
            if i + 1 >= args.preview_rows:
                break

    if args.save_dir:
        if args.streaming:
            raise ValueError("--save-dir cannot be used with --streaming")
        save_path = os.path.join(args.save_dir, config_name)
        print(f"\nsaving dataset to: {save_path}")
        ds.save_to_disk(save_path, num_proc=args.num_proc)
        print("save complete")

    print("\ndone")


if __name__ == "__main__":
    main()
