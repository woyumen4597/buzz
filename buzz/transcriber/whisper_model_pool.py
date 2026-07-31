"""Persistent Whisper model workers shared across transcription tasks.

Each queue worker owns a ``WhisperModelPool``. The pool keeps one child
process per model key (model type + model path + settings); the child loads
the model once and then serves tasks over a duplex pipe, so consecutive
files transcribed with the same model never reload it.

GPU concurrency is capped process-wide: a module-level semaphore (default 1,
override with ``BUZZ_MAX_GPU_MODELS``) limits how many GPU models can be
resident at once across all pools. When a queue worker has no more tasks it
releases its idle GPU workers so other workers are not starved of GPU slots.
"""

import logging
import multiprocessing
import os
import sys
import threading
from dataclasses import dataclass
from multiprocessing import util as mp_util
from typing import Callable, Dict, List, Optional

from buzz.conn import pipe_stderr
from buzz.model_loader import ModelType
from buzz.transcriber.transcriber import FileTranscriptionTask, Segment
from buzz.transcriber.whisper_file_transcriber import (
    PROGRESS_REGEX,
    WhisperFileTranscriber,
    backend_install_hint,
)

# How long a queue worker waits for the next task before releasing idle GPU
# workers (kept small so the GPU slot is freed quickly when the queue drains).
IDLE_POLL_SECONDS = 2.0

_BACKEND_EXTRAS = {
    ModelType.WHISPER: ("openai-whisper", "whisper"),
    ModelType.FASTER_WHISPER: ("Faster Whisper", "whisper"),
    ModelType.HUGGING_FACE: ("Transformers", "whisper"),
}


@dataclass(frozen=True)
class ModelKey:
    model_type: ModelType
    model_path: str
    word_level_timings: bool
    # Heuristic device/compute-type (resolved without importing torch in the
    # parent). Only needs to be consistent between tasks, not exact.
    device: str
    compute_type: str

    @property
    def is_gpu(self) -> bool:
        return self.device == "gpu"


def _is_gpu_host() -> bool:
    """Parent-side heuristic: on macOS torch is CPU-only (MPS not used here),
    elsewhere assume CUDA unless the user forced CPU."""
    if sys.platform == "darwin":
        return False
    return os.getenv("BUZZ_FORCE_CPU", "false").lower() != "true"


def _gpu_model_limit() -> int:
    try:
        return max(1, int(os.getenv("BUZZ_MAX_GPU_MODELS", "1")))
    except ValueError:
        return 1


_gpu_semaphore = threading.BoundedSemaphore(_gpu_model_limit())


def model_key(task: FileTranscriptionTask) -> ModelKey:
    options = task.transcription_options
    device = "gpu" if _is_gpu_host() else "cpu"
    compute_type = "default"
    if os.getenv("BUZZ_REDUCE_GPU_MEMORY", "false").lower() != "false":
        compute_type = "int8_float16" if device == "gpu" else "int8"
    return ModelKey(
        model_type=options.model.model_type,
        model_path=task.model_path or "",
        word_level_timings=options.word_level_timings,
        device=device,
        compute_type=compute_type,
    )


def _load_model_for_key(key: ModelKey):
    try:
        if key.model_type == ModelType.FASTER_WHISPER:
            import faster_whisper
            from platformdirs import user_cache_dir

            device, compute_type = WhisperFileTranscriber.faster_whisper_settings()
            model_root_dir = os.path.join(user_cache_dir("Buzz"), "models")
            model_root_dir = os.getenv("BUZZ_MODEL_ROOT", model_root_dir)
            return faster_whisper.WhisperModel(
                model_size_or_path=key.model_path,
                download_root=model_root_dir,
                device=device,
                compute_type=compute_type,
                cpu_threads=(os.cpu_count() or 8) // 2,
            )
        if key.model_type == ModelType.WHISPER:
            return WhisperFileTranscriber.load_openai_whisper_model_for_path(
                key.model_path, key.word_level_timings
            )
        if key.model_type == ModelType.HUGGING_FACE:
            from buzz.transformers_whisper import TransformersTranscriber

            return TransformersTranscriber(key.model_path)
        raise ValueError(f"Model type {key.model_type} cannot be pooled")
    except ImportError as exc:
        backend_name, extra = _BACKEND_EXTRAS.get(
            key.model_type, (str(key.model_type), "whisper")
        )
        raise ImportError(backend_install_hint(backend_name, extra)) from exc


def _transcribe_with_model(model, key: ModelKey, task, conn) -> List[Segment]:
    if key.model_type == ModelType.FASTER_WHISPER:
        return WhisperFileTranscriber.transcribe_faster_whisper_with_model(
            model,
            task,
            progress_callback=lambda pct: conn.send(("progress", int(pct))),
        )
    if key.model_type == ModelType.WHISPER:
        return WhisperFileTranscriber.transcribe_openai_whisper_with_model(model, task)
    if key.model_type == ModelType.HUGGING_FACE:
        return WhisperFileTranscriber.transcribe_hugging_face_with_model(model, task)
    raise ValueError(f"Model type {key.model_type} cannot be pooled")


def _persistent_model_worker(conn, key: ModelKey) -> None:
    """Child-process entry point: load the model once, then serve tasks.

    Progress lines written by the backends to stderr (tqdm) are forwarded
    verbatim over the pipe; structured messages carry progress/segments/errors.
    """
    try:
        model = _load_model_for_key(key)
        conn.send(("ready", None))
    except Exception as exc:
        logging.exception("Failed to load model in persistent worker")
        try:
            conn.send(("error", str(exc)))
        except Exception:
            pass
        try:
            conn.close()
        except Exception:
            pass
        return

    while True:
        try:
            task = conn.recv()
        except (EOFError, OSError):
            break
        if task is None:
            break
        try:
            with pipe_stderr(conn):
                conn.send(("progress", 0))
                segments = _transcribe_with_model(model, key, task, conn)
                conn.send(("progress", 100))
                conn.send(("segments", segments))
        except Exception as exc:
            logging.exception("Error in persistent worker task")
            try:
                conn.send(("error", str(exc)))
            except Exception:
                break
        try:
            conn.send(("task_done", None))
        except Exception:
            break
    try:
        conn.close()
    except Exception:
        pass


class _ModelWorker:
    def __init__(self, key: ModelKey, worker_target: Optional[Callable] = None):
        self.key = key
        self._gpu_slot = False
        if key.is_gpu:
            _gpu_semaphore.acquire()
            self._gpu_slot = True
        parent_conn, child_conn = multiprocessing.Pipe(duplex=True)
        self.conn = parent_conn
        target = worker_target or _persistent_model_worker
        self.process = multiprocessing.Process(target=target, args=(child_conn, key))
        self.process.start()
        child_conn.close()
        try:
            msg = self.conn.recv()
        except (EOFError, OSError):
            msg = ("error", "model worker exited before becoming ready")
        if msg[0] == "error":
            error = msg[1]
            self._terminate()
            raise Exception(error)

    def is_alive(self) -> bool:
        return self.process.is_alive()

    def run_task(
        self, task: FileTranscriptionTask, progress_callback: Callable[[int], None]
    ) -> List[Segment]:
        try:
            self.conn.send(task)
        except (BrokenPipeError, OSError):
            raise Exception("Transcription was canceled")

        error = None
        segments: List[Segment] = []
        while True:
            try:
                msg = self.conn.recv()
            except (EOFError, OSError):
                # Child died or was terminated (cancel); treat as cancellation.
                raise Exception("Transcription was canceled")
            if isinstance(msg, str):
                match = PROGRESS_REGEX.search(msg)
                if match is not None:
                    progress_callback(int(match.group().strip("%")))
                continue
            kind = msg[0]
            if kind == "progress":
                progress_callback(msg[1])
            elif kind == "segments":
                segments = msg[1]
            elif kind == "error":
                error = msg[1]
            elif kind == "task_done":
                break
        if error:
            raise Exception(error)
        return segments

    def _terminate(self) -> None:
        if self.process.is_alive():
            self.process.terminate()
            self.process.join(timeout=5)
            if self.process.is_alive():
                self.process.kill()
                self.process.join(timeout=5)
        try:
            self.conn.close()
        except OSError:
            pass
        if self._gpu_slot:
            _gpu_semaphore.release()
            self._gpu_slot = False


class WhisperModelPool:
    """Keeps model worker processes alive across tasks, keyed by model settings."""

    def __init__(self, worker_target: Optional[Callable] = None):
        self._worker_target = worker_target
        self._workers: Dict[ModelKey, _ModelWorker] = {}
        self._lock = threading.Lock()
        # Kill leftover workers if the pool is garbage-collected or the
        # interpreter exits without abort(): multiprocessing only terminates
        # daemonic children itself and then blocks joining live non-daemon
        # ones, which would hang shutdown. exitpriority=0 runs this before
        # that join loop.
        self._exit_cleanup = mp_util.Finalize(self, self.abort, exitpriority=0)

    @staticmethod
    def model_key(task: FileTranscriptionTask) -> ModelKey:
        return model_key(task)

    def transcribe(
        self, task: FileTranscriptionTask, progress_callback: Callable[[int], None]
    ) -> List[Segment]:
        key = model_key(task)
        with self._lock:
            worker = self._workers.get(key)
            if worker is None or not worker.is_alive():
                if worker is not None:
                    worker._terminate()
                worker = _ModelWorker(key, self._worker_target)
                self._workers[key] = worker
        return worker.run_task(task, progress_callback)

    def abort_key(self, key: ModelKey) -> None:
        """Terminate the worker for ``key`` (used on cancel)."""
        with self._lock:
            worker = self._workers.pop(key, None)
        if worker is not None:
            worker._terminate()

    def release_idle(self) -> None:
        """Terminate GPU workers when the queue is idle so other workers are
        not starved of GPU slots. CPU workers are kept hot."""
        with self._lock:
            idle = [w for k, w in self._workers.items() if w.key.is_gpu]
            for k, w in list(self._workers.items()):
                if w.key.is_gpu:
                    self._workers.pop(k)
        for worker in idle:
            worker._terminate()

    def abort(self) -> None:
        """Terminate every worker (app shutdown)."""
        with self._lock:
            workers = list(self._workers.values())
            self._workers.clear()
        for worker in workers:
            worker._terminate()
