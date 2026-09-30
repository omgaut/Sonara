#!/usr/bin/env python
from __future__ import annotations

import argparse
import random
from pathlib import Path

import soundfile as sf
import torch
from torch.utils.data import DataLoader, TensorDataset

from sonora.data.preprocessing import (
    discover_audio_files,
    load_audio,
    save_waveform,
    split_fixed_length,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Step 1 audio pipeline sanity checks.")
    parser.add_argument("--input-dir", type=Path, required=True, help="Directory containing source audio.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for sanity WAV outputs.")
    parser.add_argument("--target-sample-rate", type=int, default=24_000)
    parser.add_argument("--clip-seconds", type=float, default=5.0)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-source-files", type=int, default=200)
    parser.add_argument("--max-examples", type=int, default=256)
    parser.add_argument("--save-examples", type=int, default=5)
    parser.add_argument("--random-seed", type=int, default=1337)
    return parser.parse_args()


def has_bad_values(waveform: torch.Tensor) -> tuple[bool, bool]:
    return torch.isnan(waveform).any().item(), torch.isinf(waveform).any().item()


def is_unexpectedly_silent(waveform: torch.Tensor, threshold: float = 1e-4) -> bool:
    return waveform.abs().mean().item() < threshold


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    files = discover_audio_files(args.input_dir)
    rng = random.Random(args.random_seed)
    sampled_files = files.copy()
    rng.shuffle(sampled_files)
    sampled_files = sampled_files[: args.max_source_files]
    trans_files = sorted(args.input_dir.rglob("*.trans.txt"))
    expected_samples = int(round(args.target_sample_rate * args.clip_seconds))

    if not sampled_files:
        raise RuntimeError(f"No audio files found under {args.input_dir}")

    original_infos = [sf.info(path) for path in sampled_files]
    original_sample_rates = sorted({info.samplerate for info in original_infos})
    original_channels = sorted({info.channels for info in original_infos})

    processed_clips: list[torch.Tensor] = []
    source_speakers: set[str] = set()
    source_utterances: set[str] = set()
    source_chapters: set[str] = set()
    silent_count = 0
    clipped_count = 0
    nan_count = 0
    inf_count = 0

    for path in sampled_files:
        parts = path.parts
        if len(parts) >= 3:
            source_speakers.add(parts[-3])
            source_chapters.add(parts[-2])
            source_utterances.add(path.stem)

        audio = load_audio(path, target_sample_rate=args.target_sample_rate)
        clips = split_fixed_length(
            audio.waveform,
            sample_rate=audio.sample_rate,
            clip_seconds=args.clip_seconds,
            drop_last=True,
        )

        for clip in clips:
            if clip.shape != (1, expected_samples):
                raise RuntimeError(f"Bad clip shape {tuple(clip.shape)} from {path}")
            if clip.dtype != torch.float32:
                raise RuntimeError(f"Bad dtype {clip.dtype} from {path}")

            has_nan, has_inf = has_bad_values(clip)
            nan_count += int(has_nan)
            inf_count += int(has_inf)
            silent_count += int(is_unexpectedly_silent(clip))
            clipped_count += int(clip.abs().max().item() > 1.0)

            processed_clips.append(clip)
            if len(processed_clips) >= args.max_examples:
                break

        if len(processed_clips) >= args.max_examples:
            break

    if not processed_clips:
        raise RuntimeError("No full-length clips were produced. Try --clip-seconds 1 or --keep-last in preparation.")

    save_count = min(args.save_examples, len(processed_clips))
    for index in range(save_count):
        save_waveform(args.output_dir / f"sanity_check_{index:02d}.wav", processed_clips[index], args.target_sample_rate)

    dataset = TensorDataset(torch.stack(processed_clips))
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    batch = next(iter(loader))[0]

    print(f"Dataset: {args.input_dir.name}")
    print(f"Input directory:       {args.input_dir}")
    print(f"Audio files found:     {len(files)}")
    print(f"Audio files checked:   {len(sampled_files)}")
    print(f"Speakers checked:      {len(source_speakers)}")
    print(f"Chapters checked:      {len(source_chapters)}")
    print(f"Utterances checked:    {len(source_utterances)}")
    print(f"Transcript files seen: {len(trans_files)}")
    print("Transcript exposed downstream: False")
    print()
    print(f"Original sample rates: {original_sample_rates}")
    print(f"Original channels:     {original_channels}")
    print(f"Target sample rate:    {args.target_sample_rate} Hz")
    print(f"Segment duration:      {args.clip_seconds} s")
    print()
    print("Single waveform:")
    print(f"shape = {list(processed_clips[0].shape)}")
    print(f"dtype = {processed_clips[0].dtype}")
    print(f"NaN = {nan_count > 0}")
    print(f"Inf = {inf_count > 0}")
    print(f"max_abs = {processed_clips[0].abs().max().item():.4f}")
    print()
    print("Batch:")
    print(f"shape = {list(batch.shape)}")
    print()
    print(f"Processed examples checked: {len(processed_clips)}")
    print(f"Silent examples:            {silent_count}")
    print(f"Clipped examples:           {clipped_count}")
    print(f"Saved sanity WAVs:          {save_count} in {args.output_dir}")
    print()
    print("Manual check: listen to the saved sanity_check_*.wav files before moving to Step 2.")


if __name__ == "__main__":
    main()
