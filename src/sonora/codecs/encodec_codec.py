from __future__ import annotations

import torch

from sonora.codecs.base import CodecTokens


class EncodecCodec:
    """Wrapper around Meta's 24 kHz EnCodec model."""

    name = "encodec_24khz"
    sample_rate = 24_000
    frame_rate = 75
    codebook_size = 1024

    def __init__(self, bandwidth: float = 6.0, device: str | torch.device | None = None) -> None:
        from encodec import EncodecModel

        self.bandwidth = bandwidth
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = EncodecModel.encodec_model_24khz()
        self.model.set_target_bandwidth(bandwidth)
        self.model.to(self.device)
        self.model.eval()
        self.num_codebooks = self._num_codebooks_for_bandwidth(bandwidth)

    @staticmethod
    def _num_codebooks_for_bandwidth(bandwidth: float) -> int:
        # For the 24 kHz EnCodec model, each codebook contributes about 0.75 kbps.
        return int(round(bandwidth / 0.75))

    @torch.inference_mode()
    def encode(self, waveform: torch.Tensor) -> CodecTokens:
        if waveform.ndim != 2:
            raise ValueError(f"Expected waveform shape [channels, samples], got {tuple(waveform.shape)}")

        waveform = waveform.unsqueeze(0).to(self.device)
        encoded_frames = self.model.encode(waveform)
        codes = torch.cat([frame_codes for frame_codes, _scale in encoded_frames], dim=-1)
        codes = codes.squeeze(0).cpu().contiguous().to(torch.long)

        return CodecTokens(
            codes=codes,
            sample_rate=self.sample_rate,
            frame_rate=self.frame_rate,
            num_codebooks=codes.shape[0],
            codebook_size=self.codebook_size,
            bandwidth=self.bandwidth,
            codec_name=self.name,
        )

    @torch.inference_mode()
    def decode(self, tokens: CodecTokens) -> torch.Tensor:
        if tokens.codes.ndim != 2:
            raise ValueError(f"Expected token shape [codebooks, frames], got {tuple(tokens.codes.shape)}")

        codes = tokens.codes.unsqueeze(0).to(self.device)
        waveform = self.model.decode([(codes, None)])
        return waveform.squeeze(0).cpu().contiguous()
