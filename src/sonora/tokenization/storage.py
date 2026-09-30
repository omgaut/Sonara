from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from sonora.codecs.base import CodecTokens


def save_tokens(path: Path, tokens: CodecTokens, source_clip: Path) -> None:
    """Save codec tokens and minimal provenance metadata."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "codes": tokens.codes,
        "sample_rate": tokens.sample_rate,
        "frame_rate": tokens.frame_rate,
        "num_codebooks": tokens.num_codebooks,
        "codebook_size": tokens.codebook_size,
        "bandwidth": tokens.bandwidth,
        "codec_name": tokens.codec_name,
        "source_clip": str(source_clip),
    }
    torch.save(payload, path)


def load_tokens(path: Path) -> CodecTokens:
    """Load codec tokens saved by `save_tokens`."""
    payload = torch.load(path, map_location="cpu")
    return CodecTokens(
        codes=payload["codes"].to(torch.long),
        sample_rate=int(payload["sample_rate"]),
        frame_rate=int(payload["frame_rate"]),
        num_codebooks=int(payload["num_codebooks"]),
        codebook_size=int(payload["codebook_size"]),
        bandwidth=float(payload["bandwidth"]),
        codec_name=str(payload["codec_name"]),
    )

