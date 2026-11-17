import os
import subprocess
import sys
from contextlib import contextmanager
import warnings

import torch
import whisper


@contextmanager
def prevent_idle_sleep():
    """Keep macOS awake while working; allow the display to turn off."""
    process = None
    if sys.platform == "darwin":
        try:
            process = subprocess.Popen(
                ["caffeinate", "-i", "-w", str(os.getpid())],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except OSError as error:
            warnings.warn(f"Could not prevent idle sleep: {error}")
    try:
        yield
    finally:
        if process is not None:
            process.terminate()
            process.wait()


def select_device(device: str = "auto") -> str:
    if device not in {"auto", "cpu", "cuda", "mps"}:
        raise ValueError(f"Unsupported device: {device}")
    if device == "auto":
        if torch.cuda.is_available():
            return "cuda"
        return "mps" if torch.backends.mps.is_available() else "cpu"
    if device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("Apple GPU (MPS) is not available in this Python environment")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is not available in this Python environment")
    return device


def load_model(model_name: str, device: str):
    if device == "mps":
        model = whisper.load_model(model_name, device="cpu")
        # MPS cannot hold Whisper's sparse alignment_heads buffer. Keep it on
        # CPU and move the encoder and decoder (all inference weights) to GPU.
        model.encoder.to(device)
        model.decoder.to(device)
        return model
    return whisper.load_model(model_name, device=device)


def format_timestamp(seconds: float) -> str:
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def run_transcription(
    audio_path: str,
    model_name: str | None = None,
    language: str | None = None,
    device: str = "auto",
) -> dict:
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    if model_name is None:
        model_name = os.environ.get("WHISPER_MODEL", "medium")

    device = select_device(device)
    # Whisper enables its progress bar (elapsed time, speed, ETA) only when
    # verbose is explicitly False; its default None hides all progress.
    kwargs = {"fp16": device == "cuda", "verbose": False}
    if language:
        kwargs["language"] = language

    with prevent_idle_sleep():
        print(f"Loading model: {model_name} (device: {device})", flush=True)
        model = load_model(model_name, device)
        print(f"Transcribing: {audio_path}", flush=True)
        print(
            "Preparing audio and detecting language. Progress and estimated time "
            "remaining update after each audio window (up to 30 seconds of audio).",
            flush=True,
        )
        result = model.transcribe(audio_path, **kwargs)
    detected_lang = result.get("language")
    if detected_lang:
        print(f"Detected language: {detected_lang}")
    return result


def transcribe_to_srt(
    audio_path: str,
    result: dict | None = None,
    output_path: str | None = None,
    model_name: str | None = None,
    language: str | None = None,
    device: str = "auto",
) -> str:
    if result is None:
        result = run_transcription(audio_path, model_name=model_name, language=language, device=device)

    if output_path is None:
        base = os.path.splitext(audio_path)[0]
        output_path = base + ".srt"

    with open(output_path, "w", encoding="utf-8") as f:
        for i, segment in enumerate(result.get("segments", []), start=1):
            start = format_timestamp(segment["start"])
            end = format_timestamp(segment["end"])
            text = segment["text"].strip()
            f.write(f"{i}\n{start} --> {end}\n{text}\n\n")

    print(f"Saved: {output_path}")
    return output_path


def transcribe_to_txt(
    audio_path: str,
    result: dict | None = None,
    output_path: str | None = None,
    model_name: str | None = None,
    language: str | None = None,
    device: str = "auto",
) -> str:
    if result is None:
        result = run_transcription(audio_path, model_name=model_name, language=language, device=device)

    if output_path is None:
        base = os.path.splitext(audio_path)[0]
        output_path = base + ".txt"

    text = result.get("text", "").strip()
    if not text and "segments" in result:
        text = "\n".join(seg.get("text", "").strip() for seg in result["segments"])

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(text + "\n")

    print(f"Saved: {output_path}")
    return output_path


def transcribe_to_both(
    audio_path: str,
    model_name: str | None = None,
    language: str | None = None,
    device: str = "auto",
) -> tuple[str, str]:
    result = run_transcription(audio_path, model_name=model_name, language=language, device=device)
    srt_path = transcribe_to_srt(audio_path, result=result)
    txt_path = transcribe_to_txt(audio_path, result=result)
    return srt_path, txt_path


if __name__ == "__main__":
    from main import main
    sys.exit(main())
