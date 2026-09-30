from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch
import torchaudio
import soundfile as sf


AUDIO_EXTENSIONS = {".flac", ".wav", ".mp3", ".ogg", ".m4a"}


@dataclass(frozen=True)
class AudioClip:
    waveform: torch.Tensor
    sample_rate: int
    source_path: Path


def discover_audio_files(input_dir: Path) -> list[Path]:
    """Find supported audio files under an input directory."""
    input_dir = Path(input_dir)
    return sorted(
        path
        for path in input_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in AUDIO_EXTENSIONS
    )


def load_audio(path: Path, target_sample_rate: int, mono: bool = True) -> AudioClip:
    """Load audio as a waveform shaped [channels, samples]."""
    path = Path(path)
    audio, sample_rate = sf.read(path, dtype="float32", always_2d=True)
    waveform = torch.from_numpy(audio).transpose(0, 1).contiguous()

    if mono:
        waveform = to_mono(waveform)

    if sample_rate != target_sample_rate:
        waveform = torchaudio.functional.resample(
            waveform,
            orig_freq=sample_rate,
            new_freq=target_sample_rate,
        )
        sample_rate = target_sample_rate

    waveform = normalize_peak(waveform)
    return AudioClip(waveform=waveform, sample_rate=sample_rate, source_path=path)


def to_mono(waveform: torch.Tensor) -> torch.Tensor:
    """Average channels into a single channel."""
    if waveform.ndim != 2:
        raise ValueError(f"Expected waveform shape [channels, samples], got {tuple(waveform.shape)}")
    if waveform.shape[0] == 1:
        return waveform
    return waveform.mean(dim=0, keepdim=True)


def normalize_peak(waveform: torch.Tensor, target_peak: float = 0.95, eps: float = 1e-8) -> torch.Tensor:
    """Scale waveform so its absolute peak is at most target_peak."""
    peak = waveform.abs().max()
    if peak < eps:
        return waveform
    return waveform * (target_peak / peak).clamp(max=1.0)


def split_fixed_length(
    waveform: torch.Tensor,
    sample_rate: int,
    clip_seconds: float,
    hop_seconds: float | None = None,
    drop_last: bool = True,
) -> list[torch.Tensor]:
    """Split a waveform into fixed-length clips.

    Args:
        waveform: Tensor shaped [channels, samples].
        sample_rate: Audio sample rate in Hz.
        clip_seconds: Duration of each output clip.
        hop_seconds: Distance between clip starts. Defaults to clip_seconds.
        drop_last: If true, skip trailing audio shorter than clip_seconds.
    """
    if waveform.ndim != 2:
        raise ValueError(f"Expected waveform shape [channels, samples], got {tuple(waveform.shape)}")
    if clip_seconds <= 0:
        raise ValueError("clip_seconds must be positive")

    hop_seconds = clip_seconds if hop_seconds is None else hop_seconds
    if hop_seconds <= 0:
        raise ValueError("hop_seconds must be positive")

    clip_samples = int(round(clip_seconds * sample_rate))
    hop_samples = int(round(hop_seconds * sample_rate))
    total_samples = waveform.shape[-1]

    clips: list[torch.Tensor] = []
    for start in range(0, total_samples, hop_samples):
        end = start + clip_samples
        clip = waveform[:, start:end]

        if clip.shape[-1] < clip_samples:
            if drop_last:
                break
            clip = torch.nn.functional.pad(clip, (0, clip_samples - clip.shape[-1]))

        clips.append(clip.contiguous())

        if end >= total_samples and drop_last:
            break

    return clips


def save_waveform(path: Path, waveform: torch.Tensor, sample_rate: int) -> None:
    """Write a waveform to disk as audio."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    audio = waveform.detach().cpu().transpose(0, 1).numpy()
    sf.write(path, audio, sample_rate)


def iter_audio_clips(
    files: Iterable[Path],
    target_sample_rate: int,
    clip_seconds: float,
    hop_seconds: float | None = None,
    drop_last: bool = True,
) -> Iterable[tuple[Path, int, torch.Tensor]]:
    """Yield fixed-length clips as (source_path, clip_index, waveform)."""
    for path in files:
        audio = load_audio(path, target_sample_rate=target_sample_rate)
        clips = split_fixed_length(
            audio.waveform,
            sample_rate=audio.sample_rate,
            clip_seconds=clip_seconds,
            hop_seconds=hop_seconds,
            drop_last=drop_last,
        )
        for clip_index, clip in enumerate(clips):
            yield audio.source_path, clip_index, clip
