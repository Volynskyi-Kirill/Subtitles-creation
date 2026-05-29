import sys
import os
import whisper

def format_timestamp(seconds: float) -> str:
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"

def transcribe_to_srt(audio_path: str) -> str:
    model_name = os.environ.get("WHISPER_MODEL", "medium")
    print(f"Loading model: {model_name}")
    model = whisper.load_model(model_name)

    print(f"Transcribing: {audio_path}")
    result = model.transcribe(audio_path, language="en")

    base = os.path.splitext(audio_path)[0]
    srt_path = base + ".srt"

    with open(srt_path, "w", encoding="utf-8") as f:
        for i, segment in enumerate(result["segments"], start=1):
            start = format_timestamp(segment["start"])
            end = format_timestamp(segment["end"])
            text = segment["text"].strip()
            f.write(f"{i}\n{start} --> {end}\n{text}\n\n")

    print(f"Saved: {srt_path}")
    return srt_path

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python transcribe.py /path/to/audio.wav")
        sys.exit(1)
    transcribe_to_srt(sys.argv[1])
