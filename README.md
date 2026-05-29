# transcrib-audio

Transcribes audio files to `.srt` subtitle files using [OpenAI Whisper](https://github.com/openai/whisper).

## Requirements

- [uv](https://docs.astral.sh/uv/) — Python package manager
- [ffmpeg](https://ffmpeg.org/) — audio decoding

```bash
brew install ffmpeg
```

## Usage

```bash
uv run transcribe.py /path/to/audio.wav
```

The `.srt` file is saved next to the audio file with the same name:

```
audio.wav  →  audio.srt
```

## Model

The default model is `medium` — a good balance of speed and accuracy for accented speech.
Override via environment variable:

```bash
WHISPER_MODEL=large-v3 uv run transcribe.py /path/to/audio.wav
```

| Model | Speed | Quality |
|-------|-------|---------|
| `tiny` | fastest | lowest |
| `base` | fast | low |
| `small` | moderate | good |
| `medium` | moderate | **better** (default) |
| `large-v3` | slow | best |

## Supported formats

Any format supported by ffmpeg: `.wav`, `.mp3`, `.mp4`, `.m4a`, `.mkv`, etc.
