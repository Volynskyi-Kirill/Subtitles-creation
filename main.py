import argparse
import sys
from transcribe import (
    transcribe_to_both,
    transcribe_to_srt,
    transcribe_to_txt,
)


def run_transcribe(
    audio_path: str,
    output_format: str = "srt",
    model_name: str | None = None,
    language: str | None = None,
) -> str | tuple[str, str]:
    """Dispatch transcription using switch-case (match-case) based on the format flag."""
    match output_format.lower():
        case "srt":
            return transcribe_to_srt(audio_path, model_name=model_name, language=language)
        case "txt" | "text":
            return transcribe_to_txt(audio_path, model_name=model_name, language=language)
        case "both" | "all":
            return transcribe_to_both(audio_path, model_name=model_name, language=language)
        case _:
            raise ValueError(
                f"Unsupported format '{output_format}'. Supported formats: 'srt', 'txt', 'both'"
            )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Transcribe audio files to text (.txt) or subtitles (.srt) using OpenAI Whisper."
    )
    parser.add_argument(
        "audio_path",
        help="Path to the audio/video file to transcribe",
    )

    format_group = parser.add_argument_group("Output format options")
    format_group.add_argument(
        "-f",
        "--format",
        choices=["srt", "txt", "both"],
        default="srt",
        help="Transcription format: 'srt', 'txt', or 'both' (default: srt)",
    )
    format_group.add_argument(
        "--txt",
        action="store_true",
        help="Transcribe to plain text (.txt). Shortcut for --format txt",
    )
    format_group.add_argument(
        "--srt",
        action="store_true",
        help="Transcribe to subtitles (.srt). Shortcut for --format srt",
    )
    format_group.add_argument(
        "--both",
        action="store_true",
        help="Transcribe to both .srt and .txt. Shortcut for --format both",
    )

    parser.add_argument(
        "-m",
        "--model",
        default=None,
        help="Whisper model to use (overrides WHISPER_MODEL env var, default: medium)",
    )
    parser.add_argument(
        "-l",
        "--language",
        default=None,
        help="Audio language (e.g. ru, uk, en). If omitted, automatically detects language.",
    )

    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    # Resolve format flag
    if args.both:
        selected_format = "both"
    elif args.txt:
        selected_format = "txt"
    elif args.srt:
        selected_format = "srt"
    else:
        selected_format = args.format

    try:
        result = run_transcribe(
            audio_path=args.audio_path,
            output_format=selected_format,
            model_name=args.model,
            language=args.language,
        )
        return 0
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
