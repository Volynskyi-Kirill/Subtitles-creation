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

# Faster large-v3 transcription
make txt FILE=/path/to/audio.wav MODEL=large-v3-turbo
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
- `-m`, `--model`: Specify Whisper model (e.g., `tiny`, `base`, `small`, `medium`, `large-v3`, `large-v3-turbo`). The alias `turbo` also selects `large-v3-turbo`.
- `-l`, `--language`: Specify audio language (e.g. `ru`, `uk`, `en`). If omitted, Whisper automatically detects the spoken language.
- `--device {auto,cpu,cuda,mps}`: Select the compute device (default: `auto`).

### Apple Silicon GPU

On M1 and other Apple Silicon Macs, the app automatically uses the GPU through
PyTorch MPS when available. Startup prints the selected device, such as
`Loading model: large-v3-turbo (device: mps)`.

```bash
make txt DEVICE=mps
# or
uv run main.py "media/Tech interview.m4a" --txt --model large-v3-turbo --device mps
```

MPS uses FP32 for compatibility; GPU execution does not require FP16. The app
keeps Whisper's sparse alignment metadata on CPU while moving the encoder and
decoder to the GPU. CUDA uses FP16. If no supported GPU is available, automatic
selection uses CPU. To explicitly use CPU, run `make txt DEVICE=cpu`.

Transcription shows a progress bar with percentage, elapsed time, processing
speed, and estimated time remaining. It updates after each audio window (up to
30 seconds of audio), so the first estimate appears after the first window
finishes. Audio preparation and language detection happen before the bar starts.
The estimate may change as different sections take different amounts of work.

On macOS, the app prevents automatic idle sleep while loading the model and
transcribing; the display can still turn off. Keep the lid open: closing it or
manually putting the Mac to sleep can still suspend processing.

## Model

The default model is `medium` — a good balance of speed and accuracy for accented speech.
Override via flag or environment variable:

```bash
uv run main.py /path/to/audio.wav --model large-v3
# or
WHISPER_MODEL=large-v3 uv run main.py /path/to/audio.wav

# Large-v3-turbo (also available as --model turbo)
uv run main.py /path/to/audio.wav --model large-v3-turbo
# or
WHISPER_MODEL=large-v3-turbo uv run main.py /path/to/audio.wav
```

`large-v3-turbo` is an optimized version of `large-v3` with faster transcription
and a small reduction in accuracy. It is supported by the locked Whisper dependency;
the first use downloads the model automatically.

| Model | Speed | Quality |
|-------|-------|---------|
| `tiny` | fastest | lowest |
| `base` | fast | low |
| `small` | moderate | good |
| `medium` | moderate | **better** (default) |
| `large-v3` | slow | best |
| `large-v3-turbo` / `turbo` | faster than `large-v3` | near `large-v3` |

## Supported formats

Any format supported by ffmpeg: `.wav`, `.mp3`, `.mp4`, `.m4a`, `.mkv`, etc.
