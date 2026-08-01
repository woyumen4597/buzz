import os
from tempfile import mkdtemp

import pytest
from pytestqt.qtbot import QtBot
from PyQt6.QtCore import QCommandLineParser

from buzz.cli import (
    CommandLineError,
    parse_command_line,
    _add_command_options,
    _process_add_output_formats,
    _resolve_translation_settings,
)
from buzz.model_loader import TranscriptionModel
from buzz.transcriber.transcriber import OutputFormat
from tests.audio import test_audio_path


class TestCLI:
    # Must run first: it provides the session qapp's argv, which every
    # parse_command_line() call reads. It also tears down the window/workers/
    # database on completion, so it must not run before DB-using tests.
    @pytest.mark.parametrize(
        "qapp_args",
        [
            pytest.param(
                [
                    "main.py",
                    "add",
                    "--task",
                    "transcribe",
                    "--model-size",
                    "tiny",
                    "--output-directory",
                    mkdtemp(),
                    "--txt",
                    test_audio_path,
                ],
            )
        ],
        indirect=True,
    )
    def test_cli(self, qapp, qapp_args, qtbot: QtBot):
        output_directory = qapp_args[7]

        parse_command_line(qapp)

        def output_exists_at_output_directory():
            assert any(file.endswith(".txt") for file in os.listdir(output_directory))

        qtbot.wait_until(output_exists_at_output_directory, timeout=5 * 60 * 1000)

    def test_output_formats_default_to_srt(self, qapp):
        # A CLI run must never succeed without producing any artifact, so a
        # missing --srt/--vtt/--txt defaults to SRT.
        parser = QCommandLineParser()
        opts = _add_command_options(parser)
        parser.process(["main.py", "add", "--model-size", "tiny", "file.mp3"])

        assert _process_add_output_formats(parser, opts) == {OutputFormat.SRT}

    def test_translate_flag_is_recognized(self, qapp):
        parser = QCommandLineParser()
        opts = _add_command_options(parser)
        parser.process(["main.py", "add", "--translate", "file.mp3"])

        assert parser.isSet(opts["translate"])

    def test_translate_requires_api_key(self, qapp, monkeypatch, mocker):
        monkeypatch.delenv("BUZZ_TRANSLATION_API_KEY", raising=False)
        mocker.patch("buzz.cli.get_password", return_value="")

        with pytest.raises(CommandLineError, match="--translate requires an API key"):
            _resolve_translation_settings()

    def test_translate_requires_model(self, qapp, monkeypatch):
        monkeypatch.setenv("BUZZ_TRANSLATION_API_KEY", "key")
        monkeypatch.delenv("BUZZ_TRANSLATION_API_MODEL", raising=False)

        with pytest.raises(
            CommandLineError, match="--translate requires a translation model"
        ):
            _resolve_translation_settings()

    def test_translate_requires_prompt(self, qapp, monkeypatch):
        monkeypatch.setenv("BUZZ_TRANSLATION_API_KEY", "key")
        monkeypatch.setenv("BUZZ_TRANSLATION_API_MODEL", "model-x")
        monkeypatch.delenv("BUZZ_TRANSLATION_API_PROMPT", raising=False)

        with pytest.raises(
            CommandLineError, match="--translate requires a translation prompt"
        ):
            _resolve_translation_settings()

    def test_resolve_translation_settings_from_env(self, qapp, monkeypatch):
        monkeypatch.setenv("BUZZ_TRANSLATION_API_KEY", "key")
        monkeypatch.setenv("BUZZ_TRANSLATION_API_MODEL", "model-x")
        monkeypatch.setenv("BUZZ_TRANSLATION_API_PROMPT", "Translate this:")

        api_key, model, prompt = _resolve_translation_settings()

        assert (api_key, model, prompt) == ("key", "model-x", "Translate this:")
