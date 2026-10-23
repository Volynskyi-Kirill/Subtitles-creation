import os
import sys
import whisper


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
) -> dict:
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    if model_name is None:
        model_name = os.environ.get("WHISPER_MODEL", "medium")

    print(f"Loading model: {model_name}")
    model = whisper.load_model(model_name)

    print(f"Transcribing: {audio_path}")
    kwargs = {}
    if language:
        kwargs["language"] = language

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
) -> str:
    if result is None:
        result = run_transcription(audio_path, model_name=model_name, language=language)

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
) -> str:
    if result is None:
        result = run_transcription(audio_path, model_name=model_name, language=language)

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
) -> tuple[str, str]:
    result = run_transcription(audio_path, model_name=model_name, language=language)
    srt_path = transcribe_to_srt(audio_path, result=result)
    txt_path = transcribe_to_txt(audio_path, result=result)
    return srt_path, txt_path


if __name__ == "__main__":
    from main import main
    sys.exit(main())
