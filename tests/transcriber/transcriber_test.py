import pathlib
from types import SimpleNamespace

import pytest

from buzz.transcriber.file_transcriber import FileTranscriber, to_timestamp, write_output
from buzz.transcriber.transcriber import (
    FileTranscriptionTask,
    OutputFormat,
    Segment,
)


class TestToTimestamp:
    def test_to_timestamp(self):
        assert to_timestamp(0) == "00:00:00.000"
        assert to_timestamp(123456789) == "34:17:36.789"


@pytest.mark.parametrize(
    "output_format,output_text",
    [
        (OutputFormat.TXT, "Bien venue dans "),
        (
            OutputFormat.SRT,
            "1\n00:00:00,040 --> 00:00:00,299\nBien\n\n2\n00:00:00,299 --> 00:00:00,329\nvenue dans\n\n",
        ),
        (
            OutputFormat.VTT,
            "WEBVTT\n\n00:00:00.040 --> 00:00:00.299\nBien\n\n00:00:00.299 --> 00:00:00.329\nvenue dans\n\n",
        ),
    ],
)
def test_write_output(
    tmp_path: pathlib.Path, output_format: OutputFormat, output_text: str
):
    output_file_path = tmp_path / "whisper.txt"
    segments = [Segment(40, 299, "Bien"), Segment(299, 329, "venue dans")]

    write_output(
        path=str(output_file_path), segments=segments, output_format=output_format
    )

    with open(output_file_path, encoding="utf-8") as output_file:
        assert output_text == output_file.read()
    assert not pathlib.Path(f"{output_file_path}.part").exists()


def test_write_output_removes_part_on_failure(tmp_path: pathlib.Path, monkeypatch):
    output_file_path = tmp_path / "whisper.txt"
    output_file_path.write_text("previous output", encoding="utf-8")

    def fail_replace(*args):
        raise OSError("replace failed")

    monkeypatch.setattr("buzz.transcriber.file_transcriber.os.replace", fail_replace)

    with pytest.raises(OSError, match="replace failed"):
        write_output(
            path=str(output_file_path),
            segments=[Segment(40, 299, "Bien")],
            output_format=OutputFormat.TXT,
        )

    assert output_file_path.read_text(encoding="utf-8") == "previous output"
    assert not pathlib.Path(f"{output_file_path}.part").exists()


class _TestFileTranscriber(FileTranscriber):
    def transcribe(self):
        return [Segment(40, 299, "Bien")]

    def stop(self):
        pass


def test_completed_emits_after_output_and_folder_watch(
    tmp_path: pathlib.Path, monkeypatch
):
    events = []
    output_file_path = tmp_path / "whisper.txt"
    task = SimpleNamespace(
        source=FileTranscriptionTask.Source.FOLDER_WATCH,
        file_transcription_options=SimpleNamespace(output_formats=[OutputFormat.TXT]),
        transcription_options=SimpleNamespace(model=None, language=None, task=None),
        file_path=str(tmp_path / "input.wav"),
        output_directory=str(tmp_path),
    )
    transcriber = _TestFileTranscriber(task)

    monkeypatch.setattr(
        "buzz.transcriber.file_transcriber.get_output_file_path",
        lambda **kwargs: str(output_file_path),
    )
    monkeypatch.setattr(
        "buzz.transcriber.file_transcriber.write_output",
        lambda **kwargs: events.append("output"),
    )
    monkeypatch.setattr(
        transcriber, "_handle_folder_watch", lambda: events.append("folder-watch")
    )
    transcriber.completed.connect(lambda segments: events.append("completed"))

    transcriber.run()

    assert events == ["output", "folder-watch", "completed"]


def test_output_failure_emits_error_without_completed(
    tmp_path: pathlib.Path, monkeypatch
):
    task = SimpleNamespace(
        source=FileTranscriptionTask.Source.FILE_IMPORT,
        file_transcription_options=SimpleNamespace(output_formats=[OutputFormat.TXT]),
        transcription_options=SimpleNamespace(model=None, language=None, task=None),
        file_path=str(tmp_path / "input.wav"),
        output_directory=str(tmp_path),
    )
    transcriber = _TestFileTranscriber(task)
    monkeypatch.setattr(
        "buzz.transcriber.file_transcriber.get_output_file_path",
        lambda **kwargs: str(tmp_path / "whisper.txt"),
    )
    def fail_write(**kwargs):
        raise OSError("disk full")

    monkeypatch.setattr("buzz.transcriber.file_transcriber.write_output", fail_write)
    errors = []
    completed = []
    transcriber.error.connect(errors.append)
    transcriber.completed.connect(lambda segments: completed.append(segments))

    transcriber.run()

    assert errors == ["disk full"]
    assert completed == []
