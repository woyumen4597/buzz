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
from yt_dlp.utils import DownloadCancelled

from buzz import whisper_audio
from buzz.assets import APP_BASE_DIR
from buzz.transcriber.download_cookies import (
    apply_cookie_options,
    parse_cookies_from_browser,
)
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
        # Set by stop(). The URL download path polls this so Cancel interrupts a
        # download or an ffmpeg transcode instead of running to completion in the
        # background.
        self.stopped = False

    @pyqtSlot()
    def run(self):
        if self.transcription_task.source == FileTranscriptionTask.Source.URL_IMPORT:
            if not self._download_from_url():
                return

        if self.stopped:
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
        extract_options = apply_cookie_options({
            "logger": logging.getLogger(),
            # This probe only needs a title. Without these a season/playlist URL
            # (e.g. bilibili .../play/ss33378) makes yt-dlp recursively resolve
            # every episode -- over a thousand HTTP round trips before a single
            # byte of audio is fetched, which looks like a task frozen at 0%.
            "extract_flat": "in_playlist",
            "playlist_items": "1",
            "noplaylist": True,
        })

        info = None
        try:
            with YoutubeDL(extract_options) as ydl_info:
                info = ydl_info.extract_info(self.transcription_task.url, download=False)
                video_title = info.get("title", "audio")
        except Exception as exc:
            logging.debug(f"Error extracting video info: {exc}")
            video_title = "audio"

        if self.stopped:
            return False

        # A playlist/season URL has no single audio track to transcribe. Say so
        # instead of silently transcribing whichever episode happens to be first.
        if info is not None and info.get("_type") == "playlist":
            message = (
                f"{self.transcription_task.url} is a playlist, not a single video. "
                f"Add the URL of one video (for bilibili, an .../play/ep... link)."
            )
            logging.debug(message)
            self.error.emit(message)
            return False

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
            # Never fan out to a whole playlist from a single task.
            "noplaylist": True,
            # Called once per candidate video during extraction, where progress
            # hooks do not fire yet. Raising DownloadCancelled is yt-dlp's
            # documented way to abort from here.
            "match_filter": self._abort_download_if_stopped,
        }

        apply_cookie_options(options)

        try:
            logging.debug(f"Downloading audio file from URL: {self.transcription_task.url}")
            with YoutubeDL(options) as ydl:
                ydl.download([self.transcription_task.url])
        except DownloadCancelled:
            logging.debug("Download canceled by user")
            return False
        except Exception as exc:
            logging.debug(f"Error downloading audio: {exc}")
            # stop() aborts the download by raising out of the progress hook;
            # that is a cancellation, not a failure worth surfacing.
            if self.stopped:
                return False
            self.error.emit(str(exc))
            return False

        if self.stopped:
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

        popen_kwargs = {"stdout": subprocess.PIPE, "stderr": subprocess.PIPE}
        if sys.platform == "win32":
            si = subprocess.STARTUPINFO()
            si.dwFlags |= subprocess.STARTF_USESHOWWINDOW
            si.wShowWindow = subprocess.SW_HIDE
            popen_kwargs.update(
                startupinfo=si,
                env=app_env,
                creationflags=subprocess.CREATE_NO_WINDOW,
            )

        # Poll instead of subprocess.run so Cancel during a long transcode kills
        # ffmpeg rather than leaving it running to completion in the background.
        process = subprocess.Popen(cmd, **popen_kwargs)
        while True:
            try:
                stdout, stderr_bytes = process.communicate(timeout=0.5)
                break
            except subprocess.TimeoutExpired:
                if self.stopped:
                    process.kill()
                    process.communicate()
                    logging.debug("Canceled audio transcode")
                    return False

        if process.returncode != 0 or not os.path.isfile(wav_file):
            if self.stopped:
                return False
            stderr = (
                stderr_bytes.decode(errors="replace")[:2000]
                if stderr_bytes
                else ""
            )
            logging.warning(
                "Error processing downloaded audio (returncode %s): %s",
                process.returncode, stderr,
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

    def _abort_download_if_stopped(self, info_dict, incomplete=False):
        """yt-dlp match_filter hook. Returning None means "keep going"; raising
        DownloadCancelled aborts the whole download. This is the only callback
        yt-dlp runs during extraction, so it is what makes Cancel effective
        before any bytes start moving."""
        if self.stopped:
            raise DownloadCancelled("Canceled by user")
        return None

    def on_download_progress(self, data: dict):
        # Cancel arrives on another thread while ydl.download() blocks here.
        # Raising out of the progress hook is how yt-dlp is told to give up.
        if self.stopped:
            raise DownloadCancelled("Canceled by user")

        if data.get("status") != "downloading":
            return

        # DASH/HLS sources (bilibili among them) often report only an estimate,
        # and either field can be missing or None on the first callbacks. A raise
        # here would propagate out of yt-dlp's progress hook and kill the
        # download, so bail out quietly instead.
        total = data.get("total_bytes") or data.get("total_bytes_estimate")
        downloaded = data.get("downloaded_bytes")
        if not total or downloaded is None:
            return

        self.download_progress.emit(min(1.0, downloaded / total))

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
    apply_cookie_options(options)

    try:
        with YoutubeDL(options) as ydl:
            ydl.download([url])
    except Exception as exc:
        raise Exception(f"Error downloading video: {exc}") from exc

    cached = find_cached_video(audio_wav_path)
    if cached is None:
        raise Exception("Video download produced no playable file")
    return cached
