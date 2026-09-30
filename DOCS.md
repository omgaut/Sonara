# Sonora Documentation

This document tracks the pieces we build for Sonora as the project grows.

Sonora is a text-free generative speech modeling project. Instead of turning
speech into text, the pipeline turns speech into audio tokens and trains a
generative model directly on those tokens.

## Current Status

The project currently implements Step 1: the audio data pipeline.

It can:

- find audio files recursively
- load audio into PyTorch tensors
- convert stereo or multi-channel audio to mono
- resample audio to a target sample rate
- peak-normalize each source file
- split audio into fixed-length clips
- save processed clips to disk
- write metadata for the generated clips

This step does not need a GPU. It is mostly CPU work and disk I/O.

## Project Layout

```text
Sonara/
  README.md
  DOCS.md
  pyproject.toml
  requirements.txt

  scripts/
    prepare_audio_clips.py
    sanity_check_audio_pipeline.py
    tokenize_audio.py

  src/
    sonora/
      __init__.py
      codecs/
        base.py
        encodec_codec.py
      data/
        __init__.py
        preprocessing.py
      tokenization/
        storage.py
        tokenize_dataset.py
```

## Environment

The project uses a local virtual environment at `.venv`.

Activate it from the repo root:

```bash
source .venv/bin/activate
```

Install the project in editable mode:

```bash
python -m pip install -e .
```

Editable mode means changes under `src/sonora/` are picked up without
reinstalling the package every time.

## Dependencies

The current dependencies are:

- `torch`: tensor operations
- `torchaudio`: audio resampling
- `soundfile`: audio file loading and saving
- `numpy`: array support used by audio tooling
- `tqdm`: progress bars

We use `soundfile` for file I/O because local `torchaudio.save` may require a
working TorchCodec/FFmpeg setup depending on the installed versions.

## Step 1: Audio Data Pipeline

### Input

The pipeline expects a folder containing audio files. For LibriSpeech, a common
layout is:

```text
data/raw/LibriSpeech/
  train-clean-100/
    19/
      198/
        19-198-0000.flac
        19-198-0001.flac
```

Supported file extensions:

```text
.flac, .wav, .mp3, .ogg, .m4a
```

For LibriSpeech, `.flac` is the important one.

### Output

The pipeline writes fixed-length `.wav` clips and a `metadata.csv` file:

```text
data/clips/train-clean-100_24khz_5s/
  00000000.wav
  00000001.wav
  00000002.wav
  metadata.csv
```

Each metadata row contains:

- `clip_path`: processed clip path
- `source_path`: original audio file path
- `source_clip_index`: clip index within the source file
- `sample_rate`: output sample rate
- `duration_seconds`: requested clip duration
- `num_samples`: number of waveform samples in the clip

For `--sample-rate 24000 --clip-seconds 5`, each full clip has:

```text
24000 * 5 = 120000 samples
```

## Running The Pipeline

Start with a tiny subset first:

```bash
python scripts/prepare_audio_clips.py \
  --input-dir data/raw/LibriSpeech/train-clean-100/19 \
  --output-dir data/clips/debug_24khz_5s \
  --sample-rate 24000 \
  --clip-seconds 5
```

Then run a larger subset:

```bash
python scripts/prepare_audio_clips.py \
  --input-dir data/raw/LibriSpeech/train-clean-100 \
  --output-dir data/clips/train-clean-100_24khz_5s \
  --sample-rate 24000 \
  --clip-seconds 5
```

### Optional Arguments

Use overlapping clips by setting `--hop-seconds` smaller than
`--clip-seconds`:

```bash
python scripts/prepare_audio_clips.py \
  --input-dir data/raw/LibriSpeech/train-clean-100 \
  --output-dir data/clips/train-clean-100_24khz_5s_hop25 \
  --sample-rate 24000 \
  --clip-seconds 5 \
  --hop-seconds 2.5
```

Keep the final short segment by padding it:

```bash
python scripts/prepare_audio_clips.py \
  --input-dir data/raw/LibriSpeech/train-clean-100 \
  --output-dir data/clips/train-clean-100_24khz_5s_keep_last \
  --sample-rate 24000 \
  --clip-seconds 5 \
  --keep-last
```

By default, final clips shorter than `clip_seconds` are dropped.

## Step 1 Sanity Check

Before moving to EnCodec tokenization, run the sanity checker on a small slice
of `train-clean-100`:

```bash
python scripts/sanity_check_audio_pipeline.py \
  --input-dir data/raw/LibriSpeech/train-clean-100 \
  --output-dir data/sanity/step1_train_clean_100 \
  --target-sample-rate 24000 \
  --clip-seconds 5 \
  --batch-size 8 \
  --max-source-files 200 \
  --max-examples 256 \
  --save-examples 5 \
  --random-seed 1337
```

Expected output should look like:

```text
Dataset: train-clean-100
Original sample rates: [16000]
Original channels:     [1]
Target sample rate:    24000 Hz
Segment duration:      5.0 s

Single waveform:
shape = [1, 120000]
dtype = torch.float32
NaN = False
Inf = False

Batch:
shape = [8, 1, 120000]

Transcript exposed downstream: False
Saved sanity WAVs: 5 in data/sanity/step1_train_clean_100
```

Then listen to the saved files:

```text
data/sanity/step1_train_clean_100/sanity_check_00.wav
data/sanity/step1_train_clean_100/sanity_check_01.wav
...
```

They should sound like normal speech. If they sound sped up, slowed down,
distorted, silent, or static-like, stop and debug Step 1 before tokenizing.

## Step 2: EnCodec Tokenization

Step 2 converts processed waveforms into discrete codec tokens.

The important design rule is:

```text
The rest of Sonora talks to an AudioCodec interface, not directly to EnCodec.
```

That keeps EnCodec replaceable when we later add a custom neural codec.

### Small Local Test

Run EnCodec tokenization on a small number of processed clips:

```bash
python scripts/tokenize_audio.py \
  --input-dir data/clips/train-clean-100_24khz_5s \
  --output-dir data/tokens/encodec/train-clean-100_24khz_5s_debug \
  --codec encodec_24khz \
  --bandwidth 6.0 \
  --max-files 50 \
  --torch-cache-dir .cache/torch \
  --decode-output-dir data/sanity/step2_encodec_reconstructions \
  --decode-examples 5
```

Expected output:

```text
data/tokens/encodec/train-clean-100_24khz_5s_debug/
  00000000.pt
  00000001.pt
  ...
  manifest.csv

data/sanity/step2_encodec_reconstructions/
  00000000_reconstruction.wav
  ...
```

Each `.pt` file stores:

- `codes`: tensor shaped `[num_codebooks, num_frames]`
- `sample_rate`: codec sample rate
- `frame_rate`: codec frame rate
- `num_codebooks`: number of active RVQ codebooks
- `codebook_size`: size of each codebook
- `bandwidth`: EnCodec bandwidth
- `codec_name`: codec backend name
- `source_clip`: original processed clip path

For 5-second clips with 24 kHz EnCodec, expect:

```text
codes shape = [8, 375]
```

when using `--bandwidth 6.0`, because the 24 kHz EnCodec model uses about 75
frames per second and 6 kbps uses 8 codebooks.

### GPU Notes

You can run a tiny tokenization test on CPU. For the full 100 hours, use a GPU.

Recommended flow:

1. Tokenize 1-50 clips locally.
2. Decode a few reconstructions.
3. Listen to the reconstructions.
4. Only then rent a GPU for the full `train-clean-100` tokenization run.

## Module Reference

### `src/sonora/data/preprocessing.py`

This module contains reusable preprocessing functions. Later stages should
import these functions instead of duplicating audio logic.

#### `AUDIO_EXTENSIONS`

Set of supported audio extensions:

```python
{".flac", ".wav", ".mp3", ".ogg", ".m4a"}
```

#### `AudioClip`

Small dataclass returned by `load_audio`.

Fields:

- `waveform`: PyTorch tensor shaped `[channels, samples]`
- `sample_rate`: sample rate after optional resampling
- `source_path`: original file path

#### `discover_audio_files(input_dir)`

Recursively finds supported audio files under `input_dir`.

Returns:

```python
list[Path]
```

#### `load_audio(path, target_sample_rate, mono=True)`

Loads an audio file, optionally converts it to mono, resamples it, and
normalizes it.

Returns:

```python
AudioClip
```

Important shape convention:

```text
[channels, samples]
```

This is the shape expected by PyTorch audio tooling.

#### `to_mono(waveform)`

Converts a waveform shaped `[channels, samples]` into one channel by averaging
channels.

If the input is already mono, it is returned unchanged.

#### `normalize_peak(waveform, target_peak=0.95)`

Scales the waveform so its absolute peak is at most `0.95`.

This prevents clipping while keeping quiet files from being needlessly changed
when their peak is already below the target.

#### `split_fixed_length(...)`

Splits one waveform into fixed-length chunks.

Key arguments:

- `clip_seconds`: length of each output clip
- `hop_seconds`: distance between clip starts
- `drop_last`: whether to skip the final short segment

Examples:

- `clip_seconds=5, hop_seconds=None`: non-overlapping 5-second clips
- `clip_seconds=5, hop_seconds=2.5`: overlapping clips every 2.5 seconds

Returns:

```python
list[torch.Tensor]
```

Each tensor is shaped `[channels, clip_samples]`.

#### `save_waveform(path, waveform, sample_rate)`

Saves a waveform tensor to disk.

The function accepts `[channels, samples]` tensors and internally converts them
to the `[samples, channels]` shape expected by `soundfile`.

#### `iter_audio_clips(...)`

Convenience generator that combines:

1. loading
2. resampling
3. normalization
4. splitting

Yields:

```python
(source_path, clip_index, waveform)
```

This is what the CLI uses.

### `scripts/prepare_audio_clips.py`

Command-line entry point for Step 1.

It:

1. parses CLI arguments
2. discovers audio files
3. iterates through processed clips
4. writes clips as `.wav`
5. writes `metadata.csv`

Use this script for preprocessing real datasets.

### `scripts/sanity_check_audio_pipeline.py`

Command-line sanity checker for Step 1.

It verifies:

- LibriSpeech audio files are discoverable
- `.trans.txt` files are not included in downstream audio files
- original sample rates and channels are as expected
- processed waveforms are mono
- resampling changes the waveform length to the target sample rate
- 5-second clips at 24 kHz have exactly 120,000 samples
- tensors are `torch.float32`
- batches have consistent shape
- values are finite
- files are sampled with a fixed random seed for broader speaker coverage
- sanity WAVs can be written for manual listening

### `src/sonora/codecs/base.py`

Defines the small codec interface that the rest of Sonora should depend on.

`CodecTokens` stores discrete codec outputs and codec metadata.

`AudioCodec` defines:

- `encode(waveform)`
- `decode(tokens)`

Future custom codecs should implement this same interface.

### `src/sonora/codecs/encodec_codec.py`

EnCodec backend for the shared codec interface.

It uses Meta's 24 kHz EnCodec model:

```python
EncodecCodec(bandwidth=6.0)
```

Input waveform shape:

```text
[channels, samples]
```

Output token shape:

```text
[num_codebooks, num_frames]
```

### `src/sonora/tokenization/storage.py`

Save/load helpers for codec token files.

### `src/sonora/tokenization/tokenize_dataset.py`

Dataset-level tokenization helpers used by the CLI.

### `scripts/tokenize_audio.py`

Command-line entry point for Step 2.

It tokenizes processed clips and can optionally decode a few examples back to
WAV so you can listen for codec reconstruction quality.

## GPU Notes

Step 1 does not need a GPU.

Expected GPU needs later:

- Step 2, tokenizing lots of audio with EnCodec: GPU recommended
- Step 3/4, Transformer training: GPU strongly recommended
- Training a custom neural codec: GPU required for meaningful experiments
- Inference optimization: GPU needed for realistic speed measurements

## Next Planned Modules

The next project pieces should stay modular so EnCodec can later be swapped out
for a custom codec.

Planned structure:

```text
src/sonora/
  codecs/
    base.py
    encodec_codec.py

  tokenization/
    tokenize_dataset.py
    storage.py

  models/
    transformer.py

  training/
    train.py

  generation/
    generate.py
```

The key future design rule:

```text
The Transformer should not care whether tokens came from EnCodec or a custom codec.
```
