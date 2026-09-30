#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
from pathlib import Path

from tqdm import tqdm

from sonora.data.preprocessing import discover_audio_files, iter_audio_clips, save_waveform


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare fixed-length audio clips for Sonora.")
    parser.add_argument("--input-dir", type=Path, required=True, help="Directory containing raw audio.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for processed clips.")
    parser.add_argument("--sample-rate", type=int, default=24_000, help="Target sample rate.")
    parser.add_argument("--clip-seconds", type=float, default=5.0, help="Output clip duration.")
    parser.add_argument("--hop-seconds", type=float, default=None, help="Distance between clip starts.")
    parser.add_argument(
        "--keep-last",
        action="store_true",
        help="Pad and keep the final partial clip instead of dropping it.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    files = discover_audio_files(args.input_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    metadata_path = args.output_dir / "metadata.csv"
    rows: list[dict[str, str | int | float]] = []

    clip_counter = 0
    iterator = iter_audio_clips(
        files,
        target_sample_rate=args.sample_rate,
        clip_seconds=args.clip_seconds,
        hop_seconds=args.hop_seconds,
        drop_last=not args.keep_last,
    )

    for source_path, source_clip_index, waveform in tqdm(iterator, desc="Preparing clips"):
        clip_name = f"{clip_counter:08d}.wav"
        clip_path = args.output_dir / clip_name
        save_waveform(clip_path, waveform, args.sample_rate)

        rows.append(
            {
                "clip_path": str(clip_path),
                "source_path": str(source_path),
                "source_clip_index": source_clip_index,
                "sample_rate": args.sample_rate,
                "duration_seconds": args.clip_seconds,
                "num_samples": waveform.shape[-1],
            }
        )
        clip_counter += 1

    with metadata_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "clip_path",
                "source_path",
                "source_clip_index",
                "sample_rate",
                "duration_seconds",
                "num_samples",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Found {len(files)} audio files")
    print(f"Wrote {clip_counter} clips to {args.output_dir}")
    print(f"Wrote metadata to {metadata_path}")


if __name__ == "__main__":
    main()

