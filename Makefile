# Path to the audio file (override via command line: make txt FILE=path/to/audio.wav)
FILE ?= media/audio.ogg
MODEL ?= medium
LANGUAGE ?=

LANG_FLAG = $(if $(LANGUAGE),--language "$(LANGUAGE)",)

.PHONY: help txt srt both test install

help:
	@echo "Available commands:"
	@echo "  make txt   FILE=path/to/audio.wav   Transcribe to plain text (.txt)"
	@echo "  make srt   FILE=path/to/audio.wav   Transcribe to subtitles (.srt)"
	@echo "  make both  FILE=path/to/audio.wav   Transcribe to both (.txt and .srt)"
	@echo "  make test                           Run test suite"
	@echo "  make install                        Install/sync dependencies with uv"
	@echo ""
	@echo "Optional parameters:"
	@echo "  FILE      Path to audio/video file (current: $(FILE))"
	@echo "  MODEL     Whisper model: tiny, base, small, medium, large-v3 (current: $(MODEL))"
	@echo "  LANGUAGE  Audio language code: ru, uk, en, etc. (default: auto-detect)"

install:
	uv sync

test:
	uv run python -m unittest test_transcribe.py

txt:
	uv run main.py "$(FILE)" --txt --model "$(MODEL)" $(LANG_FLAG)

srt:
	uv run main.py "$(FILE)" --srt --model "$(MODEL)" $(LANG_FLAG)

both:
	uv run main.py "$(FILE)" --both --model "$(MODEL)" $(LANG_FLAG)
