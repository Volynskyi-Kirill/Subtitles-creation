# transcrib-audio

Transcribes audio files to plain text (`.txt`) or subtitle files (`.srt`) using [OpenAI Whisper](https://github.com/openai/whisper).

## Requirements

- [uv](https://docs.astral.sh/uv/) — Python package manager
- [ffmpeg](https://ffmpeg.org/) — audio decoding

```bash
brew install ffmpeg
```

## Usage

### Via Makefile

```bash
# Plain text (.txt)
make txt FILE=/path/to/audio.wav

# Subtitles (.srt)
make srt FILE=/path/to/audio.wav

# Both (.txt + .srt)
make both FILE=/path/to/audio.wav

# Optional model and language overrides
make txt FILE=/path/to/audio.wav MODEL=large-v3 LANGUAGE=ru
```

### Via CLI

Run via `main.py` (or `transcribe.py`):

```bash
# Plain text transcription (.txt)
uv run main.py /path/to/audio.wav --txt
# or
uv run main.py /path/to/audio.wav -f txt

# Subtitles transcription (.srt - default)
uv run main.py /path/to/audio.wav
# or
uv run main.py /path/to/audio.wav --srt

# Both plain text and subtitles (.txt + .srt)
uv run main.py /path/to/audio.wav --both
# or
uv run main.py /path/to/audio.wav -f both
```

The output file is saved next to the audio file with the same name:

```
audio.wav  →  audio.txt
audio.wav  →  audio.srt
```

### CLI Flags

- `--txt`: Shortcut to transcribe to plain text (`.txt`).
- `--srt`: Shortcut to transcribe to subtitles (`.srt`).
- `--both`: Generate both `.srt` and `.txt`.
- `-f`, `--format {srt,txt,both}`: Specify transcription format (default: `srt`).
- `-m`, `--model`: Specify Whisper model (e.g., `tiny`, `base`, `small`, `medium`, `large-v3`).
- `-l`, `--language`: Specify audio language (e.g. `ru`, `uk`, `en`). If omitted, Whisper automatically detects the spoken language.

## Model

The default model is `medium` — a good balance of speed and accuracy for accented speech.
Override via flag or environment variable:

```bash
uv run main.py /path/to/audio.wav --model large-v3
# or
WHISPER_MODEL=large-v3 uv run main.py /path/to/audio.wav
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
