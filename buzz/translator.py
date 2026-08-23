import json
import os
import re
import logging
import queue
import threading
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, TimeoutError, as_completed

from typing import Optional, List, Tuple
import httpx
from PyQt6.QtCore import QObject, pyqtSignal

from buzz.settings.settings import (
    DEFAULT_TRANSLATION_BATCH_SIZE,
    DEFAULT_TRANSLATION_CONCURRENCY,
    DEFAULT_TRANSLATION_MAX_OUTPUT_TOKENS,
    DEFAULT_TRANSLATION_READ_TIMEOUT,
    MAX_TRANSLATION_BATCH_SIZE,
    MAX_TRANSLATION_CONCURRENCY,
    MAX_TRANSLATION_MAX_OUTPUT_TOKENS,
    MAX_TRANSLATION_READ_TIMEOUT,
    MIN_TRANSLATION_BATCH_SIZE,
    MIN_TRANSLATION_CONCURRENCY,
    MIN_TRANSLATION_MAX_OUTPUT_TOKENS,
    MIN_TRANSLATION_READ_TIMEOUT,
    Settings,
)
from buzz.store.keyring_store import get_password, Key
from buzz.transcriber.transcriber import TranscriptionOptions
from buzz.widgets.transcriber.advanced_settings_dialog import AdvancedSettingsDialog


# Count, character, and token caps keep each provider request bounded. The
# count cap is user-configurable in Preferences; this is only the fallback for
# callers that do not pass an explicit batch size.
BATCH_SIZE = DEFAULT_TRANSLATION_BATCH_SIZE
MAX_BATCH_CHARS = 3000
# A character limit is not a token limit.  This conservative estimate keeps
# multilingual input below the provider context limit without a tokenizer.
MAX_BATCH_TOKENS = 2048
# Fallback only; the effective value is per-instance and user-configurable via
# Preferences or BUZZ_TRANSLATION_CONCURRENCY.
MAX_CONCURRENT_REQUESTS = DEFAULT_TRANSLATION_CONCURRENCY
# After the first item arrives, wait briefly for more items so a burst of
# enqueues becomes a few deterministic batch requests instead of racing
# get_nowait() into per-item requests.
BATCH_WINDOW_SECONDS = 0.1
MAX_ATTEMPTS = 4
DEFAULT_REQUESTS_PER_MINUTE = 60.0
TRANSLATION_CACHE_SIZE = 1024

DEFAULT_OPENAI_BASE_URL = "https://api.openai.com/v1"
# Must stay below the viewer's graceful shutdown wait so closing the window
# never has to terminate the worker thread mid-request. Read timeout stays
# 30s (within the 35s shutdown budget); the connect timeout is shorter so a
# dead/hung proxy fails fast and the retry can reach a healthy path sooner
# instead of burning the full budget each attempt.
REQUEST_TIMEOUT = httpx.Timeout(connect=10.0, read=30.0, write=30.0, pool=10.0)
CHAT_COMPLETIONS_PROTOCOL = "chat_completions"
RESPONSES_PROTOCOL = "responses"


def _chat_completions_url(base_url: str) -> str:
    base = base_url.rstrip("/")
    if base.endswith("/chat/completions"):
        return base
    if base.endswith("/v1"):
        return base + "/chat/completions"
    return base + "/v1/chat/completions"


def _responses_url(base_url: str) -> str:
    base = base_url.rstrip("/")
    if base.endswith("/responses"):
        return base
    if base.endswith("/v1"):
        return base + "/responses"
    return base + "/v1/responses"


def _translation_api_protocol(configured: str = "") -> str:
    configured = (
        os.getenv("BUZZ_TRANSLATION_API_PROTOCOL", "").strip().lower()
        or str(configured or "").strip().lower()
    )
    if configured in {CHAT_COMPLETIONS_PROTOCOL, RESPONSES_PROTOCOL}:
        return configured
    return CHAT_COMPLETIONS_PROTOCOL


# Queue marker that cancels the current run without stopping the worker:
# pending items are drained and the loop keeps running for the next run.
_CANCEL = object()


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


class Translator(QObject):
    # Only non-empty results are emitted on `translation`; failures (including
    # empty responses) are emitted on `translation_failed` so the UI can show
    # progress and keep the segment pending for retry.
    translation = pyqtSignal(str, int)
    translation_failed = pyqtSignal(int)
    batch_completed = pyqtSignal()
    finished = pyqtSignal()

    def __init__(
        self,
        transcription_options: TranscriptionOptions,
        advanced_settings_dialog: Optional[AdvancedSettingsDialog] = None,
        parent: Optional[QObject] = None,
    ) -> None:
        super().__init__(parent)

        logging.debug(f"Translator init: {transcription_options}")

        self.transcription_options = transcription_options
        self.advanced_settings_dialog = advanced_settings_dialog
        if advanced_settings_dialog is not None:
            advanced_settings_dialog.transcription_options_changed.connect(
                self.on_transcription_options_changed
            )

        self.queue = queue.Queue()
        self._client_instance: Optional[httpx.Client] = None
        self._client_lock = threading.Lock()
        self._state_lock = threading.RLock()
        self._future_lock = threading.Lock()
        self._active_futures = set()
        self._stopping = False
        self._cancelled = False
        self._generation = 0
        self._cancel_event = threading.Event()
        self._cancel_marker_expected = False
        self._thread_context = threading.local()

        self._cache_lock = threading.RLock()
        self._translation_cache = OrderedDict()
        self._cache_size = max(
            1, _env_int("BUZZ_TRANSLATION_CACHE_SIZE", TRANSLATION_CACHE_SIZE)
        )
        self.max_batch_tokens = max(
            1, _env_int("BUZZ_TRANSLATION_MAX_BATCH_TOKENS", MAX_BATCH_TOKENS)
        )
        requests_per_minute = _env_float(
            "BUZZ_TRANSLATION_REQUESTS_PER_MINUTE", DEFAULT_REQUESTS_PER_MINUTE
        )
        self._rate_per_second = max(0.0, requests_per_minute / 60.0)
        # One token prevents a whole minute's requests from bursting at start.
        self._rate_capacity = 1.0
        self._rate_tokens = self._rate_capacity
        self._rate_updated = time.monotonic()
        self._rate_lock = threading.Lock()

        settings = Settings()
        self.api_key = os.getenv("BUZZ_TRANSLATION_API_KEY") or get_password(
            Key.OPENAI_API_KEY
        )
        configured_base_url = os.getenv(
            "BUZZ_TRANSLATION_API_BASE_URL",
            os.getenv(
                "BUZZ_TRANSLATION_API_BASE_URl",
                settings.value(Settings.Key.CUSTOM_OPENAI_BASE_URL, ""),
            ),
        )
        self.base_url = configured_base_url or DEFAULT_OPENAI_BASE_URL
        configured_proxy = os.getenv("BUZZ_TRANSLATION_PROXY")
        if configured_proxy is None:
            configured_proxy = settings.value(Settings.Key.TRANSLATION_PROXY, "")
        self.proxy = str(configured_proxy or "").strip() or None
        # Env var wins over the stored preference so a desktop launch can be
        # configured without touching saved settings.
        self.batch_size = min(
            MAX_TRANSLATION_BATCH_SIZE,
            max(
                MIN_TRANSLATION_BATCH_SIZE,
                _env_int(
                    "BUZZ_TRANSLATION_BATCH_SIZE",
                    settings.value(
                        Settings.Key.TRANSLATION_BATCH_SIZE,
                        DEFAULT_TRANSLATION_BATCH_SIZE,
                    ),
                ),
            ),
        )
        # Same precedence as batch_size: env var wins over the stored
        # preference, and both are clamped to the supported range.
        self.max_concurrent_requests = min(
            MAX_TRANSLATION_CONCURRENCY,
            max(
                MIN_TRANSLATION_CONCURRENCY,
                _env_int(
                    "BUZZ_TRANSLATION_CONCURRENCY",
                    settings.value(
                        Settings.Key.TRANSLATION_CONCURRENCY,
                        DEFAULT_TRANSLATION_CONCURRENCY,
                    ),
                ),
            ),
        )
        self.api_protocol = _translation_api_protocol(
            settings.value(
                Settings.Key.TRANSLATION_API_PROTOCOL,
                CHAT_COMPLETIONS_PROTOCOL,
            )
        )
        # Same precedence as batch_size / concurrency.
        self.read_timeout = min(
            MAX_TRANSLATION_READ_TIMEOUT,
            max(
                MIN_TRANSLATION_READ_TIMEOUT,
                _env_int(
                    "BUZZ_TRANSLATION_READ_TIMEOUT",
                    settings.value(
                        Settings.Key.TRANSLATION_READ_TIMEOUT,
                        DEFAULT_TRANSLATION_READ_TIMEOUT,
                    ),
                ),
            ),
        )

        # Cap on generated tokens per request. Providers like DeepSeek count
        # reasoning tokens against this budget, so a hardcoded 4096 lets a
        # thinking-heavy batch exhaust the cap before any visible output is
        # emitted and the stream ends with response.incomplete (the
        # "empty stream" failure). Same env-over-settings precedence as above.
        self.max_output_tokens = min(
            MAX_TRANSLATION_MAX_OUTPUT_TOKENS,
            max(
                MIN_TRANSLATION_MAX_OUTPUT_TOKENS,
                _env_int(
                    "BUZZ_TRANSLATION_MAX_OUTPUT_TOKENS",
                    settings.value(
                        Settings.Key.TRANSLATION_MAX_OUTPUT_TOKENS,
                        DEFAULT_TRANSLATION_MAX_OUTPUT_TOKENS,
                    ),
                ),
            ),
        )

        # ponytail: per-task llm_model wins, fall back to global preferences model.
        self.llm_model = self.transcription_options.llm_model or os.getenv(
            "BUZZ_TRANSLATION_API_MODEL"
        ) or settings.value(Settings.Key.OPENAI_API_MODEL, "")

        # Reasoning effort for OpenAI-compatible reasoning models. DeepSeek
        # enables thinking by default and counts reasoning tokens against
        # max_output_tokens, so a thinking-heavy batch can exhaust the whole
        # output budget before any visible text is produced. DSH configures
        # gpt-5.6 models with high reasoning effort; match that default here.
        # BUZZ_TRANSLATION_REASONING_EFFORT overrides the model default.
        self.reasoning_effort = os.getenv(
            "BUZZ_TRANSLATION_REASONING_EFFORT", ""
        ).strip().lower()
        if not self.reasoning_effort:
            base_url_lower = (self.base_url or "").lower()
            model_lower = (self.llm_model or "").lower()
            if (
                "deepseek" in base_url_lower
                and "reasoner" not in model_lower
                and "r1" not in model_lower
            ):
                # deepseek-chat and friends default thinking ON upstream;
                # deepseek-reasoner forces thinking and ignores this knob.
                self.reasoning_effort = "none"
            elif model_lower.startswith("gpt-5.6-"):
                self.reasoning_effort = "high"

        # Log the resolved throughput knobs so a run's effective settings can be
        # confirmed from the log. No credentials or base URL here.
        logging.debug(
            "Translation config resolved: batch_size=%s, concurrency=%s, "
            "requests_per_minute=%s, max_batch_tokens=%s, protocol=%s, "
            "read_timeout=%ss, max_output_tokens=%s, reasoning=%s, model=%s",
            self.batch_size,
            self.max_concurrent_requests,
            requests_per_minute,
            self.max_batch_tokens,
            self.api_protocol,
            self.read_timeout,
            self.max_output_tokens,
            self.reasoning_effort or "default",
            self.llm_model or "<empty>",
        )

    def _client(self) -> httpx.Client:
        with self._client_lock:
            if self._client_instance is None:
                client_kwargs = {
                    "timeout": httpx.Timeout(
                        connect=REQUEST_TIMEOUT.connect,
                        read=self.read_timeout,
                        write=REQUEST_TIMEOUT.write,
                        pool=REQUEST_TIMEOUT.pool,
                    )
                }
                if self.proxy:
                    client_kwargs["proxy"] = self.proxy
                self._client_instance = httpx.Client(**client_kwargs)
            return self._client_instance

    def _close_client(self) -> None:
        with self._client_lock:
            client = self._client_instance
            self._client_instance = None
        if client is not None:
            try:
                client.close()
            except Exception:
                logging.debug("Translation client close failed", exc_info=True)

    def _run_context(self) -> Tuple[Optional[int], Optional[threading.Event]]:
        return (
            getattr(self._thread_context, "generation", None),
            getattr(self._thread_context, "cancel_event", None),
        )

    def _capture_generation(self) -> Tuple[int, threading.Event]:
        with self._state_lock:
            return self._generation, self._cancel_event

    def _is_cancelled(
        self,
        generation: Optional[int] = None,
        cancel_event: Optional[threading.Event] = None,
    ) -> bool:
        with self._state_lock:
            if self._stopping:
                return True
            if generation is not None and generation != self._generation:
                return True
            if generation is None and self._cancelled:
                return True
        return cancel_event.is_set() if cancel_event is not None else False

    def _set_run_context(
        self, generation: Optional[int], cancel_event: Optional[threading.Event]
    ) -> None:
        if generation is None:
            self._thread_context.__dict__.pop("generation", None)
            self._thread_context.__dict__.pop("cancel_event", None)
        else:
            self._thread_context.generation = generation
            self._thread_context.cancel_event = cancel_event

    def _last_request_status(self) -> Optional[int]:
        return getattr(self._thread_context, "last_request_status", None)

    def _set_last_request_status(self, status: Optional[int]) -> None:
        self._thread_context.last_request_status = status

    def _run_batch_group(
        self,
        items: List[Tuple[str, int]],
        generation: int,
        cancel_event: threading.Event,
    ) -> List[Tuple[str, int]]:
        self._set_run_context(generation, cancel_event)
        try:
            return self._translate_batch_group(items)
        finally:
            self._set_run_context(None, None)

    def _run_batch(
        self,
        items: List[Tuple[str, int]],
        generation: int,
        cancel_event: threading.Event,
    ) -> List[Tuple[str, int]]:
        self._set_run_context(generation, cancel_event)
        try:
            return self._translate_batch(items)
        finally:
            self._set_run_context(None, None)

    def _track_future(self, future) -> None:
        with self._future_lock:
            self._active_futures.add(future)
        future.add_done_callback(self._untrack_future)

    def _untrack_future(self, future) -> None:
        with self._future_lock:
            self._active_futures.discard(future)

    def _cancel_active_futures(self) -> None:
        with self._future_lock:
            futures = tuple(self._active_futures)
        for future in futures:
            future.cancel()

    def _invalidate_generation(self) -> None:
        with self._state_lock:
            self._cancelled = True
            self._generation += 1
            event = self._cancel_event
            self._cancel_event = threading.Event()
            self._cancel_marker_expected = True
        event.set()
        self._cancel_active_futures()
        self._close_client()

    def _cache_key(self, source: str) -> Tuple[str, str, str]:
        return (
            str(self.llm_model or ""),
            str(self.transcription_options.llm_prompt or ""),
            source,
        )

    def _cache_get(self, source: str) -> Optional[str]:
        key = self._cache_key(source)
        with self._cache_lock:
            value = self._translation_cache.get(key)
            if value is not None:
                self._translation_cache.move_to_end(key)
            return value

    def _cache_put(self, source: str, translation: str) -> None:
        if not translation:
            return
        key = self._cache_key(source)
        with self._cache_lock:
            self._translation_cache[key] = translation
            self._translation_cache.move_to_end(key)
            while len(self._translation_cache) > self._cache_size:
                self._translation_cache.popitem(last=False)

    def _acquire_rate_limit(
        self,
        generation: Optional[int] = None,
        cancel_event: Optional[threading.Event] = None,
    ) -> bool:
        if self._rate_per_second <= 0:
            return not self._is_cancelled(generation, cancel_event)
        while True:
            with self._rate_lock:
                now = time.monotonic()
                elapsed = max(0.0, now - self._rate_updated)
                self._rate_tokens = min(
                    self._rate_capacity,
                    self._rate_tokens + elapsed * self._rate_per_second,
                )
                self._rate_updated = now
                if self._rate_tokens >= 1.0:
                    self._rate_tokens -= 1.0
                    return not self._is_cancelled(generation, cancel_event)
                delay = (1.0 - self._rate_tokens) / self._rate_per_second
            if self._is_cancelled(generation, cancel_event):
                return False
            delay = min(delay, 0.5)
            if cancel_event is None:
                with self._state_lock:
                    cancel_event = self._cancel_event
            if self._is_cancelled(generation, cancel_event):
                return False
            cancel_event.wait(delay)

    def _build_request(
        self, system: str, user_content: str, json_mode: bool
    ) -> Tuple[str, dict, dict]:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        if self.api_protocol == RESPONSES_PROTOCOL:
            url = _responses_url(self.base_url)
            body = {
                "model": self.llm_model,
                "max_output_tokens": self.max_output_tokens,
                # Ask the upstream to stream. Without this the gateway buffers
                # the whole completion before sending a byte, so a large batch
                # blows the read timeout on time-to-first-byte alone.
                "stream": True,
                "store": False,
                "input": [
                    {
                        "role": "system",
                        "content": [{"type": "input_text", "text": system}],
                    },
                    {
                        "role": "user",
                        "content": [{"type": "input_text", "text": user_content}],
                    },
                ],
            }
            if self.reasoning_effort:
                # DeepSeek defaults thinking ON and counts reasoning tokens
                # against max_output_tokens; "none" keeps the whole budget for
                # the visible translation output.
                reasoning = {"effort": self.reasoning_effort}
                if (self.llm_model or "").lower().startswith("gpt-5.6-"):
                    reasoning["summary"] = "auto"
                    body["include"] = ["reasoning.encrypted_content"]
                body["reasoning"] = reasoning
            if json_mode:
                body["text"] = {"format": {"type": "json_object"}}
        else:
            url = _chat_completions_url(self.base_url)
            body = {
                "model": self.llm_model,
                "max_tokens": self.max_output_tokens,
                # See the note in the responses branch: streaming is what makes
                # the 30s read timeout a per-chunk gap instead of a hard cap on
                # the entire generation.
                "stream": True,
                "stream_options": {"include_usage": True},
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user_content},
                ],
            }
            if json_mode:
                body["response_format"] = {"type": "json_object"}
            if (
                self.reasoning_effort
                and (self.llm_model or "").lower().startswith("gpt-5.6-")
            ):
                body["reasoning_effort"] = self.reasoning_effort
        return url, headers, body

    def _extract_text(self, data: dict) -> Optional[str]:
        if self.api_protocol == RESPONSES_PROTOCOL:
            output_text = data.get("output_text")
            if isinstance(output_text, str):
                text = output_text
            elif isinstance(output_text, list):
                text = "".join(part for part in output_text if isinstance(part, str))
            else:
                text = ""

            if not text:
                text = "".join(
                    block.get("text", "")
                    for item in data.get("output", [])
                    if isinstance(item, dict)
                    for block in item.get("content", [])
                    if isinstance(block, dict)
                    and block.get("type") == "output_text"
                    and isinstance(block.get("text"), str)
                )
        else:
            choices = data.get("choices", [])
            message = choices[0].get("message", {}) if choices else {}
            text = message.get("content") if isinstance(message, dict) else None

        if isinstance(text, str) and text:
            # Never put potentially sensitive translated text in default logs.
            logging.debug("Received translation response (%d chars)", len(text))
            return text
        logging.error("Translation error! Unexpected response shape")
        return None

    def _read_stream(self, resp: httpx.Response) -> Tuple[Optional[str], Optional[str]]:
        """Accumulate an SSE response body into the full text.

        Falls back to whole-body JSON when the gateway ignores `stream: true`
        and answers with a plain completion object instead of an event stream.
        Returns (text, incomplete_reason): the second element is the upstream
        `incomplete_details.reason` when the stream ended with a
        `response.incomplete` event (e.g. "max_output_tokens" on DeepSeek when
        the token budget is exhausted), else None.
        """
        chunks: List[str] = []
        raw_lines: List[str] = []
        event_types = []
        saw_events = False
        incomplete_reason = None

        for line in resp.iter_lines():
            if not line:
                continue
            if not line.startswith("data:"):
                # Not SSE — remember it in case this is a buffered JSON body.
                raw_lines.append(line)
                continue
            saw_events = True
            payload = line[len("data:") :].strip()
            if not payload or payload == "[DONE]":
                continue
            try:
                event = json.loads(payload)
            except (ValueError, TypeError):
                continue
            if not isinstance(event, dict):
                continue
            event_type = event.get("type")
            if isinstance(event_type, str) and event_type not in event_types:
                event_types.append(event_type)
            if event_type == "response.incomplete":
                details = event.get("incomplete_details")
                if isinstance(details, dict):
                    reason = details.get("reason")
                    if isinstance(reason, str):
                        incomplete_reason = reason
            chunks.append(self._stream_delta(event))

        if saw_events:
            text = "".join(chunks)
            if text:
                logging.debug("Received streamed translation (%d chars)", len(text))
                if incomplete_reason:
                    # Text arrived but the stream ended before completion
                    # (e.g. JSON cut off mid-object on token exhaustion). The
                    # caller may still parse it, but truncated JSON is likely.
                    logging.warning(
                        "Streamed translation ended incomplete "
                        "(incomplete_reason=%s, %d chars received)",
                        incomplete_reason,
                        len(text),
                    )
                return text, incomplete_reason
            if incomplete_reason:
                logging.error(
                    "Translation error! Empty event stream (events=%s, "
                    "incomplete_reason=%s)",
                    ",".join(event_types) or "untyped",
                    incomplete_reason,
                )
            else:
                logging.error(
                    "Translation error! Empty event stream (events=%s)",
                    ",".join(event_types) or "untyped",
                )
            return None, incomplete_reason

        # No SSE framing at all: treat the body as one JSON document.
        try:
            return self._extract_text(json.loads("".join(raw_lines))), None
        except (ValueError, TypeError):
            logging.error("Translation error! Unparsable response body")
            return None, None

    def _stream_delta(self, event: dict) -> str:
        """Pull the incremental text out of one SSE event."""
        if self.api_protocol == RESPONSES_PROTOCOL:
            if event.get("type") == "response.output_text.delta":
                delta = event.get("delta")
                return delta if isinstance(delta, str) else ""
            return ""

        choices = event.get("choices")
        if not isinstance(choices, list) or not choices:
            return ""
        first = choices[0]
        if not isinstance(first, dict):
            return ""
        delta = first.get("delta")
        if isinstance(delta, dict):
            content = delta.get("content")
            if isinstance(content, str):
                return content
        # Some gateways replay non-streaming `message` objects inside events.
        message = first.get("message")
        if isinstance(message, dict):
            content = message.get("content")
            if isinstance(content, str):
                return content
        return ""

    @staticmethod
    def _is_retryable_status(status: int) -> bool:
        # 4xx errors are permanent (bad key, bad request, quota...); only
        # retry 408/429/5xx and network failures.
        return status == 408 or status == 429 or status >= 500

    @staticmethod
    def _retry_after_seconds(response: Optional[httpx.Response]) -> Optional[float]:
        if response is None:
            return None
        try:
            value = response.headers.get("Retry-After")
        except Exception:
            return None
        if not value:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def _sleep(
        self,
        seconds: float,
        generation: Optional[int] = None,
        cancel_event: Optional[threading.Event] = None,
    ) -> None:
        # Sleep in small slices so stop() can interrupt a backoff wait.
        while seconds > 0 and not self._is_cancelled(generation, cancel_event):
            slice_ = min(seconds, 0.5)
            if cancel_event is not None:
                cancel_event.wait(slice_)
            else:
                time.sleep(slice_)
            seconds -= slice_

    def _backoff(
        self,
        attempt: int,
        response: Optional[httpx.Response] = None,
        generation: Optional[int] = None,
        cancel_event: Optional[threading.Event] = None,
    ) -> None:
        delay = min(2 ** attempt, 8.0)
        retry_after = self._retry_after_seconds(response)
        if retry_after is not None:
            delay = max(delay, min(retry_after, 30.0))
        self._sleep(delay, generation, cancel_event)

    def _messages(
        self,
        system: str,
        user_content: str,
        json_mode: bool = False,
        generation: Optional[int] = None,
        cancel_event: Optional[threading.Event] = None,
    ) -> Optional[str]:
        """Call the configured translation protocol and return its text response."""
        if generation is None and cancel_event is None:
            generation, cancel_event = self._run_context()
        url, headers, body = self._build_request(system, user_content, json_mode)
        self._set_last_request_status(None)

        # Failure diagnostics: emitted ONLY when a request fails, so the success
        # path stays silent. Body sizes plus time-to-first-byte are what tell a
        # slow-generation stall apart from a genuine network fault.
        t_start = time.monotonic()
        user_bytes = len(user_content.encode("utf-8")) if user_content else 0
        system_bytes = len(system.encode("utf-8")) if system else 0
        n_segments = user_content.count("[") if user_content else 0  # batch marker count
        attempt_start = attempt_sent = attempt_headers = None
        attempt_mode = "unknown"
        attempt_read_timeout = self.read_timeout

        def _diag(exc_type: str, status=None, attempts: int = 0,
                  incomplete_reason: Optional[str] = None):
            now = time.monotonic()
            parts = [
                f"{exc_type} status={status} attempts={attempts}",
                f"elapsed={now - t_start:.1f}s",
                f"user={user_bytes}B system={system_bytes}B seg_markers={n_segments}",
                f"attempt_elapsed={(now - attempt_start):.1f}s" if attempt_start else "",
                f"mode={attempt_mode} read_timeout={attempt_read_timeout}s",
            ]
            parts = [part for part in parts if part]
            if incomplete_reason:
                parts.append(f"incomplete_reason={incomplete_reason}")
            if attempt_sent is not None:
                parts.append(
                    f"to_first_byte={(attempt_headers or now) - attempt_sent:.1f}s"
                )
            logging.error("Translation diag: " + " | ".join(parts))

        for attempt in range(MAX_ATTEMPTS):
            if self._is_cancelled(generation, cancel_event):
                return None
            if not self._acquire_rate_limit(generation, cancel_event):
                return None
            try:
                # Stream so the read timeout is per-chunk (30s between chunks)
                # rather than one deadline for the whole body — slow-first-byte
                # gateways and long responses no longer share the same tight cap.
                attempt_start = time.monotonic()
                attempt_sent = time.monotonic()
                attempt_headers = None
                request_body = dict(body)
                attempt_mode = "stream" if request_body.get("stream") else "buffered"
                attempt_read_timeout = (
                    self.read_timeout
                    if request_body.get("stream")
                    else min(self.read_timeout, REQUEST_TIMEOUT.read)
                )
                stream_options = {
                    "headers": headers,
                    "json": request_body,
                }
                if request_body.get("stream") is False:
                    # Some OpenAI-compatible relays accept the minimal string
                    # input but queue or proxy a structured Responses request
                    # indefinitely. Keep the compatibility fallback bounded.
                    stream_options["timeout"] = httpx.Timeout(
                        connect=REQUEST_TIMEOUT.connect,
                        read=min(self.read_timeout, REQUEST_TIMEOUT.read),
                        write=REQUEST_TIMEOUT.write,
                        pool=REQUEST_TIMEOUT.pool,
                    )
                with self._client().stream("POST", url, **stream_options) as resp:
                    attempt_headers = time.monotonic()
                    resp.raise_for_status()
                    attempt_headers = time.monotonic()
                    text, incomplete_reason = self._read_stream(resp)
                    if text is not None:
                        self._set_last_request_status(None)
                        return text
                    # An HTTP 200 with an empty/malformed SSE body is a
                    # transient provider/proxy failure, not a successful
                    # translation. Keep it inside the retry loop instead of
                    # returning None immediately and losing items.
                    _diag(
                        "EmptyTranslationStream",
                        status=resp.status_code,
                        attempts=attempt + 1,
                        incomplete_reason=incomplete_reason,
                    )
                    if incomplete_reason == "max_output_tokens":
                        # Distinct cause: the upstream burned the whole output
                        # budget (typically on reasoning tokens) before any
                        # visible text. Retrying usually won't help unless the
                        # budget grows or reasoning is disabled.
                        logging.warning(
                            "Translation output token budget exhausted "
                            "(max_output_tokens=%s, incomplete_reason=%s), "
                            "attempt %d/%d",
                            self.max_output_tokens,
                            incomplete_reason,
                            attempt + 1,
                            MAX_ATTEMPTS,
                        )
                    else:
                        logging.warning(
                            "Translation returned an empty stream, attempt %d/%d",
                            attempt + 1, MAX_ATTEMPTS,
                        )
                    if (
                        self._is_cancelled(generation, cancel_event)
                        or attempt == MAX_ATTEMPTS - 1
                    ):
                        return None
                    self._backoff(
                        attempt,
                        generation=generation,
                        cancel_event=cancel_event,
                    )
            except httpx.HTTPStatusError as e:
                status = getattr(e.response, "status_code", "unknown")
                self._set_last_request_status(status if isinstance(status, int) else None)
                _diag("HTTPStatusError", status=status, attempts=attempt + 1)
                logging.error("Translation error! HTTP %s", status)
                retryable_status = (
                    isinstance(status, int) and self._is_retryable_status(status)
                )
                if (
                    self._is_cancelled(generation, cancel_event)
                    or attempt == MAX_ATTEMPTS - 1
                    or not retryable_status
                ):
                    return None
                if retryable_status and body.get("stream") is True:
                    body["stream"] = False
                    if self.api_protocol == RESPONSES_PROTOCOL:
                        body["input"] = f"{system}\n\n{user_content}"
                        body.pop("text", None)
                    logging.warning(
                        "Translation streaming request returned HTTP %s; "
                        "retrying with a buffered compatibility payload",
                        status,
                    )
                self._backoff(attempt, e.response, generation, cancel_event)
            except httpx.TransportError as e:
                # Transient network failures are retried; only the final attempt
                # is an error, earlier ones are just progress noise.
                _diag(type(e).__name__, status=None, attempts=attempt + 1)
                log = logging.error if attempt == MAX_ATTEMPTS - 1 else logging.warning
                log("Translation network failure (%s), attempt %d/%d",
                    type(e).__name__, attempt + 1, MAX_ATTEMPTS)
                if (
                    self._is_cancelled(generation, cancel_event)
                    or attempt == MAX_ATTEMPTS - 1
                ):
                    return None
                self._backoff(attempt, generation=generation, cancel_event=cancel_event)
            except httpx.StreamError as e:
                # StreamError (e.g. ResponseNotRead) subclasses RuntimeError, not
                # TransportError, so it used to fall through to the catch-all below
                # and kill the whole batch without a single retry. It means the
                # body was cut short mid-flight, which is exactly as transient as
                # a ReadTimeout — retry it the same way.
                _diag(type(e).__name__, status=None, attempts=attempt + 1)
                log = logging.error if attempt == MAX_ATTEMPTS - 1 else logging.warning
                log("Translation stream failure (%s), attempt %d/%d",
                    type(e).__name__, attempt + 1, MAX_ATTEMPTS)
                if (
                    self._is_cancelled(generation, cancel_event)
                    or attempt == MAX_ATTEMPTS - 1
                ):
                    return None
                self._backoff(attempt, generation=generation, cancel_event=cancel_event)
            except Exception as e:
                _diag(type(e).__name__, status=None, attempts=attempt + 1)
                logging.error("Translation error! %s", type(e).__name__)
                return None
        return None

    def _translate_single(self, transcript: str, transcript_id: int) -> Tuple[str, int]:
        """Translate a single transcript via the API. Returns (translation, transcript_id)."""
        cached = self._cache_get(transcript)
        if cached is not None:
            return cached, transcript_id
        generation, cancel_event = self._run_context()
        message_args = {
            "system": self.transcription_options.llm_prompt,
            "user_content": transcript,
        }
        if generation is not None or cancel_event is not None:
            message_args.update(generation=generation, cancel_event=cancel_event)
        translation = self._messages(
            **message_args,
        )
        if translation:
            self._cache_put(transcript, translation)
        return translation or "", transcript_id

    def _translate_batch(self, items: List[Tuple[str, int]]) -> List[Tuple[str, int]]:
        """Translate multiple transcripts in a single API call.
        Returns list of (translation, transcript_id) in the same order as input."""
        if not items:
            return []

        by_index = {}
        missing = []
        for index, (transcript, tid) in enumerate(items):
            cached = self._cache_get(transcript)
            if cached is None:
                missing.append((index, transcript, tid))
            else:
                by_index[index] = cached
        if not missing:
            return [(by_index.get(index, ""), tid) for index, (_, tid) in enumerate(items)]

        numbered_parts = []
        for i, (_, transcript, _) in enumerate(missing, 1):
            numbered_parts.append(f"[{i}] {transcript}")
        combined = "\n".join(numbered_parts)

        batch_prompt = (
            f"{self.transcription_options.llm_prompt}\n\n"
            f"You will receive {len(missing)} numbered texts. "
            f"Process each one separately according to the instruction above "
            f"and return each processed text on its own line using the exact "
            f"format [number] translated text. Preserve the input numbering, "
            f"do not add comments or notes, and respond with only those numbered "
            f"lines."
        )

        generation, cancel_event = self._run_context()
        message_args = {
            "system": batch_prompt,
            "user_content": combined,
        }
        if generation is not None or cancel_event is not None:
            message_args.update(generation=generation, cancel_event=cancel_event)
        response_text = self._messages(
            **message_args,
        )
        if not response_text:
            failed_status = self._last_request_status()
            if (
                isinstance(failed_status, int)
                and failed_status >= 500
                and len(missing) > 1
            ):
                midpoint = max(1, len(missing) // 2)
                logging.warning(
                    "Translation batch failed with HTTP %s; retrying %d items "
                    "as smaller batches",
                    failed_status,
                    len(missing),
                )
                recovered = []
                for chunk in (missing[:midpoint], missing[midpoint:]):
                    recovered.extend(
                        self._translate_batch(
                            [(transcript, tid) for _, transcript, tid in chunk]
                        )
                    )
                recovered_by_id = {tid: text for text, tid in recovered}
                return [
                    (
                        by_index.get(index, recovered_by_id.get(tid, "")),
                        tid,
                    )
                    for index, (_, tid) in enumerate(items)
                ]
            return [
                (by_index.get(index, ""), tid)
                for index, (_, tid) in enumerate(items)
            ]

        translations = self._parse_batch_response(response_text, len(missing))
        retry_missing = []
        for translation, (index, transcript, tid) in zip(translations, missing):
            if translation:
                by_index[index] = translation
                self._cache_put(transcript, translation)
            else:
                retry_missing.append((index, transcript, tid))

        if retry_missing:
            # A malformed/missing entry must not lose the whole batch; re-request
            # only the failed items individually.
            logging.debug(
                f"Batch response incomplete ({len(retry_missing)} missing), "
                "retrying individually"
            )
            for index, transcript, tid in retry_missing:
                by_index[index] = self._translate_single(transcript, tid)[0]

        return [
            (by_index.get(index, ""), tid)
            for index, (_, tid) in enumerate(items)
        ]

    @staticmethod
    def _try_parse_json_mapping(response: str) -> Optional[dict]:
        """Parse a JSON batch response into {int index: text}, or None."""
        text = response.strip()
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text).strip()
        try:
            data = json.loads(text)
        except (json.JSONDecodeError, ValueError):
            return None
        mapping = None
        if isinstance(data, dict):
            candidate = data.get("translations")
            if isinstance(candidate, dict):
                mapping = candidate
            elif data and all(
                isinstance(k, (str, int)) and isinstance(v, str)
                for k, v in data.items()
            ):
                mapping = data
        if not mapping:
            return None
        translations = {}
        for key, value in mapping.items():
            try:
                translations[int(key)] = value
            except (TypeError, ValueError):
                continue
        return translations or None

    @staticmethod
    def _parse_batch_response(response: str, expected_count: int) -> List[str]:
        """Parse a batch response into a list of strings.
        Accepts a JSON object mapping numbers to texts, or numbered '[N] text' lines."""
        mapping = Translator._try_parse_json_mapping(response)
        if mapping is not None:
            return [mapping.get(i, "") for i in range(1, expected_count + 1)]

        # Fallback: split on [N] markers — re.split with a group returns:
        # [before, group1, after1, group2, after2, ...]
        parts = re.split(r'\[(\d+)\]\s*', response)

        translations = {}
        for i in range(1, len(parts) - 1, 2):
            num = int(parts[i])
            text = parts[i + 1].strip()
            translations[num] = text

        return [
            translations.get(i, "")
            for i in range(1, expected_count + 1)
        ]

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        # Three UTF-8 bytes per token is deliberately conservative for both
        # ASCII and CJK text; it avoids adding a tokenizer dependency.
        text = str(text or "")
        return max(1, (len(text.encode("utf-8")) + 2) // 3)

    @staticmethod
    def _split_batches(
        items: List[Tuple[str, int]],
        token_budget: Optional[int] = None,
        prompt_tokens: int = 0,
        max_tokens: Optional[int] = None,
        batch_size: Optional[int] = None,
    ) -> List[List[Tuple[str, int]]]:
        """Split items by count, chars, and a conservative token estimate."""
        if max_tokens is not None:
            token_budget = max_tokens
        batch_size = max(1, batch_size or BATCH_SIZE)
        token_budget = max(1, token_budget or MAX_BATCH_TOKENS)
        token_budget = max(1, token_budget - max(0, prompt_tokens))
        batches = []
        current: List[Tuple[str, int]] = []
        current_chars = 0
        current_tokens = 0
        for item in items:
            chars = len(item[0])
            tokens = Translator._estimate_tokens(item[0])
            if current and (
                len(current) >= batch_size
                or current_chars + chars > MAX_BATCH_CHARS
                or current_tokens + tokens > token_budget
            ):
                batches.append(current)
                current = []
                current_chars = 0
                current_tokens = 0
            current.append(item)
            current_chars += chars
            current_tokens += tokens
        if current:
            batches.append(current)
        return batches

    def translate_items_sync(
        self, items: List[Tuple[str, int]]
    ) -> List[Tuple[str, int]]:
        """Translate items concurrently while returning results in input order."""
        batches = self._split_batches(
            items,
            token_budget=self.max_batch_tokens,
            prompt_tokens=self._estimate_tokens(self.transcription_options.llm_prompt),
            batch_size=self.batch_size,
        )
        completed = {}
        pending = {}
        executor = ThreadPoolExecutor(max_workers=self.max_concurrent_requests)
        generation, cancel_event = self._capture_generation()
        try:
            for index, batch in enumerate(batches):
                future = executor.submit(
                    self._run_batch, batch, generation, cancel_event
                )
                pending[future] = (index, batch)
                self._track_future(future)

            while pending:
                if self._is_cancelled(generation, cancel_event):
                    for future in pending:
                        future.cancel()
                    break
                try:
                    future = next(as_completed(tuple(pending), timeout=0.1))
                except TimeoutError:
                    continue
                index, batch = pending.pop(future)
                try:
                    completed[index] = future.result()
                except Exception:
                    logging.exception("Translation batch failed")
                    completed[index] = [("", tid) for _, tid in batch]
        finally:
            cancelled = self._is_cancelled(generation, cancel_event)
            executor.shutdown(wait=not cancelled, cancel_futures=True)

        results: List[Tuple[str, int]] = []
        for index, batch in enumerate(batches):
            results.extend(
                completed.get(index, [("", tid) for _, tid in batch])
            )
        return results

    def _translate_batch_group(
        self, items: List[Tuple[str, int]]
    ) -> List[Tuple[str, int]]:
        results: List[Tuple[str, int]] = []
        prompt_tokens = self._estimate_tokens(self.transcription_options.llm_prompt)
        for sub_batch in self._split_batches(
            items,
            token_budget=self.max_batch_tokens,
            prompt_tokens=prompt_tokens,
            batch_size=self.batch_size,
        ):
            generation, cancel_event = self._run_context()
            if self._is_cancelled(generation, cancel_event):
                break
            if len(sub_batch) == 1:
                results.append(self._translate_single(*sub_batch[0]))
            else:
                logging.debug(
                    f"Translating batch of {len(sub_batch)} in single request"
                )
                results.extend(self._translate_batch(sub_batch))
        return results

    def _emit_batch_result(
        self,
        results: List[Tuple[str, int]],
        generation: Optional[int] = None,
    ) -> None:
        if not self._is_cancelled(generation):
            for translation, tid in results:
                if translation:
                    self.translation.emit(translation, tid)
                else:
                    self.translation_failed.emit(tid)
        self.batch_completed.emit()

    def start(self):
        logging.debug("Starting translation queue")
        with self._state_lock:
            self._stopping = False
            if self._cancel_event.is_set():
                self._cancel_event = threading.Event()
        executor = ThreadPoolExecutor(max_workers=self.max_concurrent_requests)
        pending = {}
        completed_in_order = {}
        next_sequence = 0
        next_emit = 0
        preserve_order = False
        stop_after_pending = False
        cancel_seen = False

        def consume_cancel_marker() -> bool:
            with self._state_lock:
                expected = self._cancel_marker_expected
                self._cancel_marker_expected = False
            if expected:
                return True
            self._invalidate_generation()
            with self._state_lock:
                self._cancel_marker_expected = False
            return True

        try:
            while True:
                if self._cancelled or self._stopping:
                    cancel_seen = True

                while (
                    len(pending) < self.max_concurrent_requests
                    and not stop_after_pending
                    and not cancel_seen
                ):
                    try:
                        item = self.queue.get(
                            timeout=BATCH_WINDOW_SECONDS if pending else None
                        )
                    except queue.Empty:
                        break

                    if item is None:
                        stop_after_pending = True
                        break

                    if item is _CANCEL:
                        consume_cancel_marker()
                        cancel_seen = True
                        break

                    # Collect a short burst into one provider request.
                    batch = [item]
                    deadline = time.monotonic() + BATCH_WINDOW_SECONDS
                    while len(batch) < self.batch_size:
                        remaining = deadline - time.monotonic()
                        if remaining <= 0:
                            break
                        try:
                            next_item = self.queue.get(timeout=remaining)
                        except queue.Empty:
                            break
                        if next_item is None:
                            stop_after_pending = True
                            break
                        if next_item is _CANCEL:
                            consume_cancel_marker()
                            cancel_seen = True
                            batch = []
                            break
                        batch.append(next_item)

                    if batch:
                        preserve_order = preserve_order or any(
                            transcript_id is None for _, transcript_id in batch
                        )
                        generation, cancel_event = self._capture_generation()
                        future = executor.submit(
                            self._run_batch_group,
                            batch,
                            generation,
                            cancel_event,
                        )
                        pending[future] = (batch, generation, next_sequence)
                        next_sequence += 1
                        self._track_future(future)

                if cancel_seen:
                    for future in pending:
                        future.cancel()
                    pending.clear()
                    if self._stopping:
                        executor.shutdown(wait=False, cancel_futures=True)
                        break
                    executor.shutdown(wait=False, cancel_futures=True)
                    executor = ThreadPoolExecutor(max_workers=self.max_concurrent_requests)
                    with self._state_lock:
                        self._cancelled = False
                    completed_in_order.clear()
                    next_sequence = 0
                    next_emit = 0
                    preserve_order = False
                    cancel_seen = False
                    stop_after_pending = False
                    continue

                if pending:
                    try:
                        future = next(as_completed(tuple(pending), timeout=0.1))
                    except TimeoutError:
                        continue
                    batch, generation, sequence = pending.pop(future)
                    try:
                        results = future.result()
                    except Exception:
                        logging.exception("Translation batch failed")
                        results = [("", tid) for _, tid in batch]
                    if preserve_order:
                        completed_in_order[sequence] = (results, generation)
                        while next_emit in completed_in_order:
                            ready_results, ready_generation = completed_in_order.pop(
                                next_emit
                            )
                            self._emit_batch_result(ready_results, ready_generation)
                            next_emit += 1
                    else:
                        self._emit_batch_result(results, generation)
                    continue

                if stop_after_pending or self._stopping:
                    logging.debug("Translation queue received stop signal")
                    break
        finally:
            cancelled = self._stopping or self._cancelled
            for future in pending:
                future.cancel()
            executor.shutdown(wait=not cancelled, cancel_futures=True)

        logging.debug("Translation queue stopped")
        self.finished.emit()

    def on_transcription_options_changed(
        self, transcription_options: TranscriptionOptions
    ):
        self.transcription_options = transcription_options
        # ponytail: re-resolve model so a per-task change wins over the global fallback.
        self.llm_model = transcription_options.llm_model or os.getenv(
            "BUZZ_TRANSLATION_API_MODEL"
        ) or Settings().value(Settings.Key.OPENAI_API_MODEL, "")

    def enqueue(self, transcript: str, transcript_id: Optional[int] = None):
        self.llm_model = self.transcription_options.llm_model or os.getenv(
            "BUZZ_TRANSLATION_API_MODEL"
        ) or Settings().value(Settings.Key.OPENAI_API_MODEL, "")
        self.queue.put((transcript, transcript_id))

    def stop(self):
        # Stop retries/backoff and close the transport so shutdown is prompt.
        with self._state_lock:
            self._stopping = True
            event = self._cancel_event
        event.set()
        self._cancel_active_futures()
        self._close_client()
        self.queue.put(None)

    def cancel(self):
        """Cancel the current run: drop queued items and discard in-flight
        results. The worker thread stays alive for the next run."""
        self._invalidate_generation()
        while True:
            try:
                self.queue.get_nowait()
            except queue.Empty:
                break
        with self._state_lock:
            self._cancel_marker_expected = True
        self.queue.put(_CANCEL)
