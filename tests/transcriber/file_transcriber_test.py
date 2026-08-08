import pytest

from buzz.transcriber.file_transcriber import FileTranscriber
from buzz.transcriber.transcriber import (
    FileTranscriptionTask,
    FileTranscriptionOptions,
    OutputFormat,
    Segment,
    Task,
    TranscriptionOptions,
)
from tests.audio import test_audio_path


class StubFileTranscriber(FileTranscriber):
    def transcribe(self):
        return [
            Segment(start=0, end=1000, text="hello"),
            Segment(start=1500, end=2500, text="world"),
        ]

    def stop(self):
        pass


def _task(output_directory, output_formats, translate=False):
    return FileTranscriptionTask(
        file_path=str(test_audio_path),
        source=FileTranscriptionTask.Source.FILE_IMPORT,
        transcription_options=TranscriptionOptions(task=Task.TRANSCRIBE),
        file_transcription_options=FileTranscriptionOptions(
            output_formats=output_formats, translate=translate
        ),
        model_path="mock_path",
        output_directory=str(output_directory),
    )


class TestOnDownloadProgress:
    """yt-dlp calls this from inside its progress hook; a raise here aborts the
    download, so missing/None byte fields must be tolerated."""

    def _emitted(self, tmp_path, data):
        transcriber = StubFileTranscriber(_task(tmp_path, {OutputFormat.SRT}))
        received = []
        transcriber.download_progress.connect(received.append)
        transcriber.on_download_progress(data)
        return received

    def test_emits_fraction_from_total_bytes(self, tmp_path, qtbot):
        assert self._emitted(
            tmp_path,
            {"status": "downloading", "downloaded_bytes": 50, "total_bytes": 200},
        ) == [0.25]

    def test_falls_back_to_total_bytes_estimate(self, tmp_path, qtbot):
        # DASH/HLS sources (bilibili among them) report only an estimate.
        assert self._emitted(
            tmp_path,
            {
                "status": "downloading",
                "downloaded_bytes": 30,
                "total_bytes_estimate": 120,
            },
        ) == [0.25]

    def test_clamps_overshoot_past_estimate(self, tmp_path, qtbot):
        assert self._emitted(
            tmp_path,
            {
                "status": "downloading",
                "downloaded_bytes": 150,
                "total_bytes_estimate": 100,
            },
        ) == [1.0]

    @pytest.mark.parametrize(
        "data",
        [
            {"status": "downloading", "downloaded_bytes": 10},
            {"status": "downloading", "downloaded_bytes": 10, "total_bytes": None},
            {"status": "downloading", "downloaded_bytes": 10, "total_bytes": 0},
            {"status": "downloading", "total_bytes": 100},
            {"status": "finished", "downloaded_bytes": 100, "total_bytes": 100},
            {},
        ],
    )
    def test_stays_quiet_on_unusable_payloads(self, tmp_path, qtbot, data):
        assert self._emitted(tmp_path, data) == []


class TestFileTranscriberTranslationExport:
    def test_writes_text_outputs_by_default(self, tmp_path, qtbot):
        task = _task(tmp_path, {OutputFormat.SRT})
        transcriber = StubFileTranscriber(task)
        completed = []
        transcriber.completed.connect(completed.append)

        transcriber.run()

        files = list(tmp_path.glob("*.srt"))
        assert len(files) == 1
        assert ".translated." not in files[0].name
        assert "hello" in files[0].read_text(encoding="utf-8")
        assert completed and completed[0][0].text == "hello"

    def test_writes_translated_outputs_when_translate_enabled(
        self, tmp_path, qtbot, mocker
    ):
        task = _task(tmp_path, {OutputFormat.SRT}, translate=True)
        fake_translator = mocker.Mock()
        fake_translator.translate_items_sync.return_value = [
            ("bonjour", 0),
            ("monde", 1),
        ]
        mocker.patch(
            "buzz.transcriber.file_transcriber.Translator",
            return_value=fake_translator,
        )
        transcriber = StubFileTranscriber(task)
        completed = []
        transcriber.completed.connect(completed.append)

        transcriber.run()

        files = list(tmp_path.glob("*.translated.srt"))
        assert len(files) == 1
        content = files[0].read_text(encoding="utf-8")
        assert "bonjour" in content
        assert "monde" in content
        assert "hello" not in content
        fake_translator.translate_items_sync.assert_called_once_with(
            [("hello", 0), ("world", 1)]
        )
        assert completed and completed[0][1].translation == "monde"

    def test_translation_failure_still_writes_successful_segments(
        self, tmp_path, qtbot, mocker
    ):
        task = _task(tmp_path, {OutputFormat.SRT}, translate=True)
        fake_translator = mocker.Mock()
        fake_translator.translate_items_sync.return_value = [
            ("bonjour", 0),
            ("", 1),
        ]
        mocker.patch(
            "buzz.transcriber.file_transcriber.Translator",
            return_value=fake_translator,
        )
        transcriber = StubFileTranscriber(task)

        transcriber.run()

        content = list(tmp_path.glob("*.translated.srt"))[0].read_text(
            encoding="utf-8"
        )
        assert "bonjour" in content
        assert "monde" not in content
