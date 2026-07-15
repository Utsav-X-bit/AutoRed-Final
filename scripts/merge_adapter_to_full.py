#!/usr/bin/env python3
"""
Merge a PEFT LoRA adapter into its base model and save a standalone full model.

This is a workaround for vLLM 0.8.5 offline LoRA silently failing to apply some
PEFT adapters. Using a merged full model removes the LoRA path entirely and
lets vLLM load the adapter behavior as if it were a base checkpoint.

Example (run on CUDA HPC worker; CPU offload is used for the merge step):
    python scripts/merge_adapter_to_full.py \
        --base-model Orenguteng/Llama-3.1-8B-Lexi-Uncensored-V2 \
        --adapter experiment/results/planner_sft_v2_contract_anchor/checkpoint-27 \
        --output-dir experiment/results/planner_sft_v2_contract_anchor/checkpoint-27_merged
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge a PEFT LoRA adapter into a full model")
    parser.add_argument("--base-model", required=True, help="Base model id or path")
    parser.add_argument("--adapter", required=True, help="PEFT adapter directory")
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where the merged full model will be saved",
    )
    parser.add_argument(
        "--dtype",
        default="bfloat16",
        choices=["bfloat16", "float16", "float32"],
        help="Torch dtype for the merged model",
    )
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Use only locally cached HF files",
    )
    parser.add_argument(
        "--device-map",
        default="auto",
        choices=["auto", "cpu", "cuda:0", "cuda"],
        help="Device map for the merge step (default: auto). Use 'cpu' if GPUs are occupied.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype = getattr(torch, args.dtype)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[MERGE] Loading base model: {args.base_model} ({args.dtype})")
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=dtype,
        device_map=args.device_map,
        local_files_only=args.local_files_only,
        trust_remote_code=True,
    )

    print(f"[MERGE] Loading adapter: {args.adapter}")
    model = PeftModel.from_pretrained(
        base_model,
        args.adapter,
        local_files_only=args.local_files_only,
        trust_remote_code=True,
    )

    print("[MERGE] Merging and unloading adapter...")
    merged = model.merge_and_unload()

    print(f"[MERGE] Saving merged model to: {output_dir}")
    merged.save_pretrained(output_dir, safe_serialization=True)

    # Save the adapter tokenizer (it may contain the tuned chat template / special tokens)
    tokenizer_dir = Path(args.adapter) if Path(args.adapter).exists() else args.base_model
    print(f"[MERGE] Saving tokenizer from: {tokenizer_dir}")
    tokenizer = AutoTokenizer.from_pretrained(
        str(tokenizer_dir),
        local_files_only=args.local_files_only,
        trust_remote_code=True,
    )
    tokenizer.save_pretrained(output_dir)

    print(f"[MERGE] Done. Merged model is at: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
