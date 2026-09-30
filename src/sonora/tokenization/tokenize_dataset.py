from __future__ import annotations

from pathlib import Path
from typing import Iterable

import soundfile as sf
import torch

from sonora.codecs.base import AudioCodec
from sonora.data.preprocessing import discover_audio_files, load_audio, save_waveform
from sonora.tokenization.storage import save_tokens


def discover_clips(input_dir: Path) -> list[Path]:
    """Find clip audio files for tokenization."""
    return discover_audio_files(input_dir)


def tokenize_files(
    files: Iterable[Path],
    output_dir: Path,
    codec: AudioCodec,
    decode_output_dir: Path | None = None,
    decode_examples: int = 0,
) -> tuple[int, Path | None]:
    """Tokenize audio files and optionally reconstruct a few examples."""
    output_dir = Path(output_dir)
    decode_output_dir = Path(decode_output_dir) if decode_output_dir else None
    manifest_path = output_dir / "manifest.csv"
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = ["token_path,source_clip,codec_name,sample_rate,frame_rate,num_codebooks,codebook_size,bandwidth,num_frames\n"]
    count = 0

    for path in files:
        audio = load_audio(path, target_sample_rate=codec.sample_rate)
        tokens = codec.encode(audio.waveform)
        token_path = output_dir / f"{path.stem}.pt"
        save_tokens(token_path, tokens, source_clip=path)

        rows.append(
            ",".join(
                [
                    str(token_path),
                    str(path),
                    tokens.codec_name,
                    str(tokens.sample_rate),
                    str(tokens.frame_rate),
                    str(tokens.num_codebooks),
                    str(tokens.codebook_size),
                    str(tokens.bandwidth),
                    str(tokens.codes.shape[-1]),
                ]
            )
            + "\n"
        )

        if decode_output_dir is not None and count < decode_examples:
            reconstruction = codec.decode(tokens)
            decode_path = decode_output_dir / f"{path.stem}_reconstruction.wav"
            save_waveform(decode_path, reconstruction, codec.sample_rate)

        count += 1

    manifest_path.write_text("".join(rows))
    return count, manifest_path


def inspect_audio_file(path: Path) -> tuple[int, int, int]:
    """Return sample rate, channels, and samples for a file."""
    info = sf.info(path)
    return info.samplerate, info.channels, info.frames

