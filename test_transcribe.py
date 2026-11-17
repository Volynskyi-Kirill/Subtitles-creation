import os
import tempfile
import unittest
from unittest.mock import patch

from main import main, parse_args, run_transcribe
from transcribe import (
    format_timestamp,
    load_model,
    run_transcription,
    select_device,
    transcribe_to_both,
    transcribe_to_srt,
    transcribe_to_txt,
)


class TestTranscribe(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.audio_path = os.path.join(self.temp_dir.name, "test_audio.wav")
        with open(self.audio_path, "w") as f:
            f.write("fake audio content")

        self.mock_result = {
            "text": " Hello world! This is a test. ",
            "segments": [
                {
                    "start": 0.0,
                    "end": 2.5,
                    "text": " Hello world!",
                },
                {
                    "start": 2.5,
                    "end": 5.123,
                    "text": " This is a test.",
                },
            ],
        }

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_format_timestamp(self):
        self.assertEqual(format_timestamp(0.0), "00:00:00,000")
        self.assertEqual(format_timestamp(65.5), "00:01:05,500")
        self.assertEqual(format_timestamp(3661.123), "01:01:01,123")

    def test_transcribe_to_txt(self):
        txt_path = transcribe_to_txt(self.audio_path, result=self.mock_result)
        expected_txt = os.path.join(self.temp_dir.name, "test_audio.txt")
        self.assertEqual(txt_path, expected_txt)
        self.assertTrue(os.path.exists(txt_path))

        with open(txt_path, "r", encoding="utf-8") as f:
            content = f.read()
        self.assertEqual(content, "Hello world! This is a test.\n")

    def test_transcribe_to_srt(self):
        srt_path = transcribe_to_srt(self.audio_path, result=self.mock_result)
        expected_srt = os.path.join(self.temp_dir.name, "test_audio.srt")
        self.assertEqual(srt_path, expected_srt)
        self.assertTrue(os.path.exists(srt_path))

        with open(srt_path, "r", encoding="utf-8") as f:
            content = f.read()
        self.assertIn("00:00:00,000 --> 00:00:02,500\nHello world!", content)
        self.assertIn("00:00:02,500 --> 00:00:05,123\nThis is a test.", content)

    def test_transcribe_to_both(self):
        with patch("transcribe.run_transcription", return_value=self.mock_result):
            srt_path, txt_path = transcribe_to_both(self.audio_path)
            self.assertTrue(os.path.exists(srt_path))
            self.assertTrue(os.path.exists(txt_path))

    def test_run_transcribe_switch_case(self):
        with patch("main.transcribe_to_srt", return_value="out.srt") as mock_srt, \
             patch("main.transcribe_to_txt", return_value="out.txt") as mock_txt, \
             patch("main.transcribe_to_both", return_value=("out.srt", "out.txt")) as mock_both:

            res_srt = run_transcribe(self.audio_path, output_format="srt")
            self.assertEqual(res_srt, "out.srt")
            mock_srt.assert_called_once()

            res_txt = run_transcribe(self.audio_path, output_format="txt")
            self.assertEqual(res_txt, "out.txt")
            mock_txt.assert_called_once()

            res_both = run_transcribe(self.audio_path, output_format="both")
            self.assertEqual(res_both, ("out.srt", "out.txt"))
            mock_both.assert_called_once()

            with self.assertRaises(ValueError):
                run_transcribe(self.audio_path, output_format="unsupported_format")

    def test_argparse_flags(self):
        args = parse_args(["audio.wav"])
        self.assertEqual(args.format, "srt")
        self.assertFalse(args.txt)

        args_txt = parse_args(["audio.wav", "--txt"])
        self.assertTrue(args_txt.txt)

        args_both = parse_args(["audio.wav", "--both"])
        self.assertTrue(args_both.both)

        args_format_txt = parse_args(["audio.wav", "-f", "txt"])
        self.assertEqual(args_format_txt.format, "txt")

    def test_main_execution(self):
        with patch("main.run_transcribe", return_value="test.txt") as mock_rt:
            code = main([self.audio_path, "--txt"])
            self.assertEqual(code, 0)
            mock_rt.assert_called_once_with(
                audio_path=self.audio_path,
                output_format="txt",
                model_name=None,
                language=None,
                device="auto",
            )

    def test_run_transcription_language(self):
        from transcribe import run_transcription
        mock_model = unittest.mock.MagicMock()
        mock_model.transcribe.return_value = {
            "text": "Пример текста",
            "language": "ru",
            "segments": [],
        }

        with patch("whisper.load_model", return_value=mock_model), \
             patch("transcribe.select_device", return_value="cpu"):
            # Test default: no language passed, Whisper auto-detects
            res = run_transcription(self.audio_path)
            self.assertEqual(res["language"], "ru")
            mock_model.transcribe.assert_called_with(self.audio_path, fp16=False)

            # Test explicit language
            run_transcription(self.audio_path, language="uk")
            mock_model.transcribe.assert_called_with(self.audio_path, fp16=False, language="uk")

    def test_device_selection(self):
        with patch("torch.cuda.is_available", return_value=False), \
             patch("torch.backends.mps.is_available", return_value=True):
            self.assertEqual(select_device(), "mps")
            self.assertEqual(select_device("cpu"), "cpu")
        with patch("torch.cuda.is_available", return_value=True):
            self.assertEqual(select_device(), "cuda")
        with patch("torch.cuda.is_available", return_value=False), \
             patch("torch.backends.mps.is_available", return_value=False):
            self.assertEqual(select_device(), "cpu")
            with self.assertRaisesRegex(RuntimeError, "MPS"):
                select_device("mps")
            with self.assertRaisesRegex(RuntimeError, "CUDA"):
                select_device("cuda")

    def test_mps_load_keeps_sparse_buffer_on_cpu(self):
        model = unittest.mock.MagicMock()
        with patch("whisper.load_model", return_value=model) as loader:
            self.assertIs(load_model("large-v3-turbo", "mps"), model)
            loader.assert_called_once_with("large-v3-turbo", device="cpu")
        model.encoder.to.assert_called_once_with("mps")
        model.decoder.to.assert_called_once_with("mps")
        model.to.assert_not_called()
        model.alignment_heads.to.assert_not_called()

    def test_device_forwarded_from_cli(self):
        with patch("main.transcribe_to_txt", return_value="test.txt") as transcriber:
            self.assertEqual(main([self.audio_path, "--txt", "--device", "mps"]), 0)
            self.assertEqual(transcriber.call_args.kwargs["device"], "mps")

    def test_gpu_precision(self):
        for device, fp16 in [("mps", False), ("cuda", True), ("cpu", False)]:
            with self.subTest(device=device), \
                 patch("transcribe.select_device", return_value=device), \
                 patch("transcribe.load_model") as loader:
                loader.return_value.transcribe.return_value = self.mock_result
                run_transcription(self.audio_path, device=device)
                loader.return_value.transcribe.assert_called_once_with(self.audio_path, fp16=fp16)


if __name__ == "__main__":
    unittest.main()
