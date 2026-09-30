# Sonora

Text-free generative speech modeling experiments.

See `DOCS.md` for the living technical documentation.

## Step 1: Audio Data Pipeline

Install the project in editable mode from inside the repo:

```bash
source .venv/bin/activate
python -m pip install -e .
```

Create fixed-length speech clips from a folder of audio files:

```bash
source .venv/bin/activate
python scripts/prepare_audio_clips.py \
  --input-dir data/raw/LibriSpeech \
  --output-dir data/clips \
  --sample-rate 24000 \
  --clip-seconds 5
```

The script recursively finds audio files, converts them to mono, resamples them,
normalizes peak volume, slices them into fixed-length clips, and writes a
`metadata.csv` file next to the generated clips.
