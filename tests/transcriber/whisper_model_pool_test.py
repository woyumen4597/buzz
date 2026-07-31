import threading
import time
from unittest.mock import MagicMock

import pytest

from buzz.model_loader import ModelType, TranscriptionModel
from buzz.transcriber.transcriber import FileTranscriptionTask, Segment, Task, TranscriptionOptions
from buzz.transcriber.whisper_model_pool import ModelKey, WhisperModelPool


def make_task(model_path="model-a", word_level_timings=False) -> FileTranscriptionTask:
    return FileTranscriptionTask(
        transcription_options=TranscriptionOptions(
            language="en",
            task=Task.TRANSCRIBE,
            model=TranscriptionModel(model_type=ModelType.FASTER_WHISPER),
            word_level_timings=word_level_timings,
        ),
        file_transcription_options=None,
        file_path="/tmp/audio.mp3",
        model_path=model_path,
    )


def _echo_worker(conn, key):
    """Fake worker: ready once per process, echoes a fixed segment per task."""
    conn.send(("ready", None))
    while True:
        try:
            task = conn.recv()
        except (EOFError, OSError):
            break
        if task is None:
            break
        try:
            conn.send(("progress", 0))
            conn.send(("progress", 50))
            conn.send(("segments", [Segment(start=0, end=1000, text="hello")]))
        except Exception:
            break
        conn.send(("task_done", None))
    conn.close()


def _error_worker(conn, key):
    conn.send(("ready", None))
    while True:
        try:
            task = conn.recv()
        except (EOFError, OSError):
            break
        if task is None:
            break
        try:
            conn.send(("error", "boom"))
        except Exception:
            break
        conn.send(("task_done", None))
    conn.close()


def _hang_worker(conn, key):
    """Worker that never answers a task; used to test cancellation."""
    conn.send(("ready", None))
    while True:
        try:
            task = conn.recv()
        except (EOFError, OSError):
            break
        if task is None:
            break
        time.sleep(60)


def test_pool_reuses_worker_process_for_same_key():
    pool = WhisperModelPool(worker_target=_echo_worker)
    progress = []
    task = make_task()

    segments = pool.transcribe(task, lambda pct: progress.append(pct))
    assert segments[0].text == "hello"
    assert progress == [0, 50]

    first_pid = list(pool._workers.values())[0].process.pid
    # Second task with the same key must reuse the same process (model loaded once)
    pool.transcribe(task, lambda pct: None)
    assert len(pool._workers) == 1
    assert list(pool._workers.values())[0].process.pid == first_pid

    pool.abort()


def test_pool_spawns_new_worker_for_different_key():
    pool = WhisperModelPool(worker_target=_echo_worker)

    pool.transcribe(make_task(model_path="model-a"), lambda pct: None)
    pool.transcribe(make_task(model_path="model-b"), lambda pct: None)

    assert len(pool._workers) == 2

    pool.abort()


def test_pool_reports_task_error():
    pool = WhisperModelPool(worker_target=_error_worker)

    with pytest.raises(Exception, match="boom"):
        pool.transcribe(make_task(), lambda pct: None)

    pool.abort()


def test_cancel_terminates_worker_and_raises_cancellation():
    pool = WhisperModelPool(worker_target=_hang_worker)
    task = make_task()

    errors = []

    def run_in_thread():
        try:
            pool.transcribe(task, lambda pct: None)
        except Exception as exc:
            errors.append(exc)

    thread = threading.Thread(target=run_in_thread, daemon=True)
    thread.start()
    # Wait until the worker exists and is alive (it never answers the task).
    deadline = time.time() + 10
    while True:
        with pool._lock:
            workers = list(pool._workers.values())
        if workers and workers[0].is_alive():
            break
        if time.time() > deadline:
            pytest.fail("worker did not start")
        time.sleep(0.05)
    # Give the transcribe thread time to hand the task to the child.
    time.sleep(0.2)
    worker = workers[0]
    pool.abort_key(worker.key)
    thread.join(timeout=10)

    assert not thread.is_alive()
    assert len(errors) == 1
    assert "canceled" in str(errors[0]).lower()

    assert len(pool._workers) == 0


def test_release_idle_releases_only_gpu_workers():
    pool = WhisperModelPool(worker_target=_echo_worker)
    gpu_key = ModelKey(
        model_type=ModelType.FASTER_WHISPER,
        model_path="gpu-model",
        word_level_timings=False,
        device="gpu",
        compute_type="default",
    )
    cpu_key = ModelKey(
        model_type=ModelType.FASTER_WHISPER,
        model_path="cpu-model",
        word_level_timings=False,
        device="cpu",
        compute_type="default",
    )

    # Seed two live workers; the host heuristic decides their real device, so
    # re-key them explicitly to exercise release_idle's GPU-only release on
    # any host.
    pool.transcribe(make_task(model_path="gpu-model"), lambda pct: None)
    pool.transcribe(make_task(model_path="cpu-model"), lambda pct: None)
    workers = list(pool._workers.values())
    pool._workers.clear()
    workers[0].key = gpu_key
    workers[1].key = cpu_key
    pool._workers[gpu_key] = workers[0]
    pool._workers[cpu_key] = workers[1]

    pool.release_idle()

    assert gpu_key not in pool._workers
    assert cpu_key in pool._workers

    pool.abort()
