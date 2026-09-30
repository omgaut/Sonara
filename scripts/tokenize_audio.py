#!/usr/bin/env python
from __future__ import annotations

import argparse
import os
from itertools import islice
from pathlib import Path

import torch
from tqdm import tqdm

from sonora.codecs import EncodecCodec
from sonora.tokenization.tokenize_dataset import discover_clips, tokenize_files


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tokenize processed audio clips with a neural audio codec.")
    parser.add_argument("--input-dir", type=Path, required=True, help="Directory containing processed WAV clips.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for saved token .pt files.")
    parser.add_argument("--codec", choices=["encodec_24khz"], default="encodec_24khz")
    parser.add_argument("--bandwidth", type=float, default=6.0, help="EnCodec bandwidth in kbps.")
    parser.add_argument("--device", default=None, help="Device override, e.g. cpu, cuda, cuda:0, or mps.")
    parser.add_argument("--torch-cache-dir", type=Path, default=Path(".cache/torch"), help="Local cache for codec weights.")
    parser.add_argument("--max-files", type=int, default=None, help="Limit files for a small test run.")
    parser.add_argument("--decode-output-dir", type=Path, default=None, help="Optional directory for reconstructions.")
    parser.add_argument("--decode-examples", type=int, default=0, help="Number of examples to reconstruct.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.torch_cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("TORCH_HOME", str(args.torch_cache_dir))
    try:
        import certifi

        os.environ.setdefault("SSL_CERT_FILE", certifi.where())
    except ImportError:
        pass

    files = discover_clips(args.input_dir)
    if args.max_files is not None:
        files = list(islice(files, args.max_files))

    if not files:
        raise RuntimeError(f"No audio clips found under {args.input_dir}")

    if args.codec == "encodec_24khz":
        codec = EncodecCodec(bandwidth=args.bandwidth, device=args.device)
    else:
        raise ValueError(f"Unsupported codec: {args.codec}")

    print(f"Codec:        {codec.name}")
    print(f"Device:       {codec.device}")
    print(f"Bandwidth:    {codec.bandwidth} kbps")
    print(f"Input clips:  {len(files)}")
    print(f"CUDA ready:   {torch.cuda.is_available()}")

    count, manifest_path = tokenize_files(
        tqdm(files, desc="Tokenizing clips"),
        output_dir=args.output_dir,
        codec=codec,
        decode_output_dir=args.decode_output_dir,
        decode_examples=args.decode_examples,
    )

    print(f"Wrote {count} token files to {args.output_dir}")
    print(f"Wrote manifest to {manifest_path}")
    if args.decode_output_dir and args.decode_examples:
        print(f"Wrote reconstructions to {args.decode_output_dir}")


if __name__ == "__main__":
    main()
