import glob
import logging
import os
import sys
import subprocess
import shutil
import time
from abc import abstractmethod
from typing import Optional, List
from pathlib import Path

from PyQt6.QtCore import QObject, pyqtSignal, pyqtSlot
from yt_dlp import YoutubeDL

from buzz import whisper_audio
from buzz.assets import APP_BASE_DIR
from buzz.transcriber.transcriber import (
    FileTranscriptionTask,
    get_output_file_path,
    Segment,
    OutputFormat,
)
from buzz.translator import Translator

app_env = os.environ.copy()
app_env['PATH'] = os.pathsep.join([os.path.join(APP_BASE_DIR, "_internal")] + [app_env['PATH']])


class FileTranscriber(QObject):
    transcription_task: FileTranscriptionTask
    progress = pyqtSignal(tuple)  # (current, total)
    download_progress = pyqtSignal(float)
    checkpoint = pyqtSignal(list)  # durable partial List[Segment]
    completed = pyqtSignal(list)  # List[Segment]
    error = pyqtSignal(str)

    def __init__(self, task: FileTranscriptionTask, parent: Optional["QObject"] = None):
        super().__init__(parent)
        self.transcription_task = task

    @pyqtSlot()
    def run(self):
        if self.transcription_task.source == FileTranscriptionTask.Source.URL_IMPORT:
            if not self._download_from_url():
                return

        try:
            segments = self.transcribe()
        except Exception as exc:
            logging.exception("")
            self.error.emit(str(exc))
            return

        for segment in segments:
            segment.text = segment.text.strip()

        # Keep a durable checkpoint before translation/export.  A process that
        # dies after inference can be restarted with its last complete result.
        self.transcription_task.segments = segments
        self.checkpoint.emit(segments)

        try:
            segment_key = "text"
            if self.transcription_task.file_transcription_options.translate:
                translator = Translator(
                    self.transcription_task.transcription_options
                )
                results = translator.translate_items_sync(
                    [(segment.text, i) for i, segment in enumerate(segments)]
                )
                for index, (translation, _tid) in enumerate(results):
                    segments[index].translation = translation
                segment_key = "translation"

            for (
                output_format
            ) in self.transcription_task.file_transcription_options.output_formats:
                default_path = get_output_file_path(
                    file_path=self.transcription_task.file_path,
                    output_format=output_format,
                    language=self.transcription_task.transcription_options.language,
                    output_directory=self.transcription_task.output_directory,
                    model=self.transcription_task.transcription_options.model,
                    task=self.transcription_task.transcription_options.task,
                    variant="translated"
                    if segment_key == "translation"
                    else "",
                )

                write_output(
                    path=default_path,
                    segments=segments,
                    output_format=output_format,
                    segment_key=segment_key,
                )

            if self.transcription_task.source == FileTranscriptionTask.Source.FOLDER_WATCH:
                self._handle_folder_watch()
        except Exception as exc:
            logging.exception("")
            self.error.emit(str(exc))
            return

        self.completed.emit(segments)

    def _download_from_url(self) -> bool:
        cookiefile = os.getenv("BUZZ_DOWNLOAD_COOKIEFILE")

        extract_options = {
            "logger": logging.getLogger(),
        }
        if cookiefile:
            extract_options["cookiefile"] = cookiefile

        try:
            with YoutubeDL(extract_options) as ydl_info:
                info = ydl_info.extract_info(self.transcription_task.url, download=False)
                video_title = info.get("title", "audio")
        except Exception as exc:
            logging.debug(f"Error extracting video info: {exc}")
            video_title = "audio"

        video_title = YoutubeDL.sanitize_info({"title": video_title})["title"]
        for char in ['/', '\\', ':', '*', '?', '"', '<', '>', '|']:
            video_title = video_title.replace(char, '_')

        removed = cleanup_download_cache()
        if removed:
            logging.info(
                "Removed %s expired url-download cache entr%s",
                removed, "y" if removed == 1 else "ies",
            )

        # Downloads live in an app-managed cache dir (per-task subfolder) so the
        # original video can be reused later for MP4 subtitle export and stale
        # files can be cleaned up.
        cache_dir = _download_cache_dir()
        task_dir = os.path.join(cache_dir, str(self.transcription_task.uid))
        os.makedirs(task_dir, exist_ok=True)
        temp_output_path = os.path.join(task_dir, video_title)
        wav_file = os.path.join(task_dir, video_title + ".wav")
        wav_file = str(Path(wav_file).resolve())

        options = {
            "format": "bestaudio/best",
            "progress_hooks": [self.on_download_progress],
            "outtmpl": temp_output_path,
            "logger": logging.getLogger(),
        }

        if cookiefile:
            options["cookiefile"] = cookiefile

        try:
            logging.debug(f"Downloading audio file from URL: {self.transcription_task.url}")
            with YoutubeDL(options) as ydl:
                ydl.download([self.transcription_task.url])
        except Exception as exc:
            logging.debug(f"Error downloading audio: {exc}")
            self.error.emit(str(exc))
            return False

        downloaded = _find_downloaded_file(temp_output_path)
        if downloaded is None:
            message = (
                f"Error downloading audio: no file was downloaded from "
                f"{self.transcription_task.url}"
            )
            logging.debug(message)
            self.error.emit(message)
            return False

        cmd = [
            "ffmpeg",
            "-nostdin",
            "-threads", "0",
            "-i", downloaded,
            "-ac", "1",
            "-ar", str(whisper_audio.SAMPLE_RATE),
            "-acodec", "pcm_s16le",
            "-loglevel", "panic",
            wav_file
        ]

        if sys.platform == "win32":
            si = subprocess.STARTUPINFO()
            si.dwFlags |= subprocess.STARTF_USESHOWWINDOW
            si.wShowWindow = subprocess.SW_HIDE
            result = subprocess.run(
                cmd,
                capture_output=True,
                startupinfo=si,
                env=app_env,
                creationflags=subprocess.CREATE_NO_WINDOW
            )
        else:
            result = subprocess.run(cmd, capture_output=True)

        if result.returncode != 0 or not os.path.isfile(wav_file):
            stderr = (
                result.stderr.decode(errors="replace")[:2000]
                if result.stderr
                else ""
            )
            logging.warning(
                "Error processing downloaded audio (returncode %s): %s",
                result.returncode, stderr,
            )
            raise Exception(
                f"Error processing downloaded audio: {stderr or 'unknown ffmpeg error'}"
            )

        self.transcription_task.file_path = wav_file
        logging.debug(f"Downloaded audio to file: {self.transcription_task.file_path}")
        return True

    def _handle_folder_watch(self):
        source_path = (
            self.transcription_task.original_file_path
            or self.transcription_task.file_path
        )
        if source_path and os.path.exists(source_path):
            if self.transcription_task.delete_source_file:
                os.remove(source_path)
            else:
                shutil.move(
                    source_path,
                    os.path.join(
                        self.transcription_task.output_directory,
                        os.path.basename(source_path),
                    ),
                )

    def on_download_progress(self, data: dict):
        if data["status"] == "downloading":
            self.download_progress.emit(data["downloaded_bytes"] / data["total_bytes"])

    @abstractmethod
    def transcribe(self) -> List[Segment]:
        ...

    @abstractmethod
    def stop(self):
        ...


def sanitize_segments(
    segments: List[Segment], segment_key: str = "text"
) -> List[Segment]:
    """Sort by start time and drop segments that would produce invalid
    subtitle output: negative timestamps, zero-length ranges or empty text."""
    valid: List[Segment] = []
    for segment in segments:
        start = segment.start
        end = segment.end
        if start is None or end is None or start < 0 or end <= start:
            logging.warning(
                "Skipping segment with invalid timestamps: %s -> %s", start, end
            )
            continue
        if not (getattr(segment, segment_key) or "").strip():
            logging.warning(
                "Skipping segment with empty %s text", segment_key
            )
            continue
        valid.append(segment)
    valid.sort(key=lambda segment: (segment.start, segment.end))
    return valid


def write_output(
    path: str,
    segments: List[Segment],
    output_format: OutputFormat,
    segment_key: str = 'text'
):
    logging.debug(
        "Writing transcription output, path = %s, output format = %s, number of segments = %s",
        path,
        output_format,
        len(segments),
    )

    segments = sanitize_segments(segments, segment_key)

    temp_path = f"{path}.part"
    try:
        with open(os.fsencode(temp_path), "w", encoding="utf-8") as file:
            if output_format == OutputFormat.TXT:
                combined_text = ""
                previous_end_time = None

                paragraph_split_time = int(os.getenv("BUZZ_PARAGRAPH_SPLIT_TIME", "2000"))

                for segment in segments:
                    if previous_end_time is not None and (segment.start - previous_end_time) >= paragraph_split_time:
                        combined_text += "\n\n"
                    combined_text += getattr(segment, segment_key).strip() + " "
                    previous_end_time = segment.end

                file.write(combined_text)

            elif output_format == OutputFormat.VTT:
                file.write("WEBVTT\n\n")
                for segment in segments:
                    file.write(
                        f"{to_timestamp(segment.start)} --> {to_timestamp(segment.end)}\n"
                    )
                    file.write(f"{getattr(segment, segment_key)}\n\n")

            elif output_format == OutputFormat.SRT:
                for i, segment in enumerate(segments):
                    file.write(f"{i + 1}\n")
                    file.write(
                        f'{to_timestamp(segment.start, ms_separator=",")} --> {to_timestamp(segment.end, ms_separator=",")}\n'
                    )
                    file.write(f"{getattr(segment, segment_key)}\n\n")

        os.replace(temp_path, path)
    except Exception:
        try:
            os.remove(temp_path)
        except OSError:
            pass
        raise

    logging.debug("Written transcription output")


def to_timestamp(ms: float, ms_separator=".") -> str:
    hr = int(ms / (1000 * 60 * 60))
    ms -= hr * (1000 * 60 * 60)
    min = int(ms / (1000 * 60))
    ms -= min * (1000 * 60)
    sec = int(ms / 1000)
    ms = int(ms - sec * 1000)
    return f"{hr:02d}:{min:02d}:{sec:02d}{ms_separator}{ms:03d}"

# To detect when transcription source is a video
VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".avi", ".m4v", ".webm", ".ogm", ".wmv"}

def is_video_file(path: str) -> bool:
    return Path(path).suffix.lower() in VIDEO_EXTENSIONS


# URL imports keep their downloads (audio + on-demand video) in an app-managed
# cache directory so the original video is still available for MP4 subtitle
# export; entries expire after this many days.
URL_CACHE_MAX_AGE_DAYS = 30


def _download_cache_dir() -> str:
    if os.name == "nt":
        base = os.environ.get("LOCALAPPDATA") or os.path.expanduser("~")
        return os.path.join(base, "Buzz", "Cache", "url-downloads")
    return os.path.join(os.path.expanduser("~"), ".cache", "buzz", "url-downloads")


def cleanup_download_cache(max_age_days: int = URL_CACHE_MAX_AGE_DAYS) -> int:
    """Delete url-download cache entries older than max_age_days; returns the
    number of entries removed."""
    cache_dir = _download_cache_dir()
    if not os.path.isdir(cache_dir):
        return 0
    cutoff = time.time() - max_age_days * 24 * 60 * 60
    removed = 0
    for entry in os.listdir(cache_dir):
        path = os.path.join(cache_dir, entry)
        try:
            if os.path.getmtime(path) >= cutoff:
                continue
            if os.path.isdir(path) and not os.path.islink(path):
                shutil.rmtree(path)
            else:
                os.remove(path)
            removed += 1
        except OSError:
            logging.debug("Failed to clean download cache entry: %s", path, exc_info=True)
    return removed


def _find_downloaded_file(outtmpl: str) -> Optional[str]:
    if os.path.isfile(outtmpl):
        return outtmpl
    for path in sorted(glob.glob(outtmpl + ".*")):
        if os.path.isfile(path):
            return path
    return None


def find_cached_video(audio_wav_path: str) -> Optional[str]:
    """Locate the original video kept next to the URL-downloaded audio file."""
    title = os.path.splitext(os.path.basename(audio_wav_path))[0]
    directory = os.path.dirname(audio_wav_path)
    for ext in sorted(VIDEO_EXTENSIONS, key=len, reverse=True):
        candidate = os.path.join(directory, f"{title}{ext}")
        if os.path.isfile(candidate):
            return candidate
    return None


def download_video_to_cache(url: str, audio_wav_path: str) -> str:
    """Download the original video for a URL transcription into the same cache
    directory as its audio file; returns the video file path."""
    cached = find_cached_video(audio_wav_path)
    if cached:
        return cached

    directory = os.path.dirname(audio_wav_path)
    os.makedirs(directory, exist_ok=True)
    title = os.path.splitext(os.path.basename(audio_wav_path))[0]
    options = {
        "format": "bestvideo+bestaudio/best",
        "merge_output_format": "mp4",
        "outtmpl": os.path.join(directory, title),
        "logger": logging.getLogger(),
    }
    cookiefile = os.getenv("BUZZ_DOWNLOAD_COOKIEFILE")
    if cookiefile:
        options["cookiefile"] = cookiefile

    try:
        with YoutubeDL(options) as ydl:
            ydl.download([url])
    except Exception as exc:
        raise Exception(f"Error downloading video: {exc}") from exc

    cached = find_cached_video(audio_wav_path)
    if cached is None:
        raise Exception("Video download produced no playable file")
    return cached
