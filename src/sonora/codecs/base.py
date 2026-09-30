from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch


@dataclass(frozen=True)
class CodecTokens:
    """Discrete audio tokens produced by a neural codec."""

    codes: torch.Tensor
    sample_rate: int
    frame_rate: int
    num_codebooks: int
    codebook_size: int
    bandwidth: float
    codec_name: str


class AudioCodec(Protocol):
    """Shared interface for codec backends.

    Keeping this interface small lets later code use EnCodec today and a custom
    Sonora codec later without changing the Transformer or token datasets.
    """

    name: str
    sample_rate: int
    frame_rate: int
    num_codebooks: int
    codebook_size: int
    bandwidth: float

    def encode(self, waveform: torch.Tensor) -> CodecTokens:
        """Convert waveform tensor shaped [channels, samples] into tokens."""

    def decode(self, tokens: CodecTokens) -> torch.Tensor:
        """Convert codec tokens back into waveform shaped [channels, samples]."""

