import json
import os
import re
import logging
import queue
import time
from concurrent.futures import ThreadPoolExecutor

from typing import Optional, List, Tuple
import httpx
from PyQt6.QtCore import QObject, pyqtSignal

from buzz.settings.settings import Settings
from buzz.store.keyring_store import get_password, Key
from buzz.transcriber.transcriber import TranscriptionOptions
from buzz.widgets.transcriber.advanced_settings_dialog import AdvancedSettingsDialog


# Max items per batch and max combined chars per batch. The char budget keeps
# the prompt and the expected translation output within max_tokens.
BATCH_SIZE = 20
MAX_BATCH_CHARS = 3000
MAX_CONCURRENT_REQUESTS = 2
# After the first item arrives, wait briefly for more items so a burst of
# enqueues becomes a few deterministic batch requests instead of racing
# get_nowait() into per-item requests.
BATCH_WINDOW_SECONDS = 0.1
MAX_ATTEMPTS = 4

DEFAULT_OPENAI_BASE_URL = "https://api.openai.com/v1"
# Must stay below the viewer's graceful shutdown wait so closing the window
# never has to terminate the worker thread mid-request.
REQUEST_TIMEOUT = 30.0
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
        self._stopping = False
        self._cancelled = False

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
        self.api_protocol = _translation_api_protocol(
            settings.value(
                Settings.Key.TRANSLATION_API_PROTOCOL,
                CHAT_COMPLETIONS_PROTOCOL,
            )
        )

        # ponytail: per-task llm_model wins, fall back to global preferences model.
        self.llm_model = self.transcription_options.llm_model or os.getenv(
            "BUZZ_TRANSLATION_API_MODEL"
        ) or settings.value(Settings.Key.OPENAI_API_MODEL, "")

    def _client(self) -> httpx.Client:
        if self._client_instance is None:
            self._client_instance = httpx.Client(timeout=REQUEST_TIMEOUT)
        return self._client_instance

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
                "max_output_tokens": 4096,
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
            if json_mode:
                body["text"] = {"format": {"type": "json_object"}}
        else:
            url = _chat_completions_url(self.base_url)
            body = {
                "model": self.llm_model,
                "max_tokens": 4096,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user_content},
                ],
            }
            if json_mode:
                body["response_format"] = {"type": "json_object"}
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
            logging.debug(f"Received translation response: {data}")
            return text
        logging.error(f"Translation error! Unexpected response: {data}")
        return None

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

    def _sleep(self, seconds: float) -> None:
        # Sleep in small slices so stop() can interrupt a backoff wait.
        while seconds > 0 and not self._stopping:
            slice_ = min(seconds, 0.5)
            time.sleep(slice_)
            seconds -= slice_

    def _backoff(self, attempt: int, response: Optional[httpx.Response] = None) -> None:
        delay = min(2 ** attempt, 8.0)
        retry_after = self._retry_after_seconds(response)
        if retry_after is not None:
            delay = max(delay, min(retry_after, 30.0))
        self._sleep(delay)

    def _messages(
        self, system: str, user_content: str, json_mode: bool = False
    ) -> Optional[str]:
        """Call the configured translation protocol and return its text response."""
        url, headers, body = self._build_request(system, user_content, json_mode)
        for attempt in range(MAX_ATTEMPTS):
            try:
                resp = self._client().post(url, headers=headers, json=body)
                resp.raise_for_status()
                return self._extract_text(resp.json())
            except httpx.HTTPStatusError as e:
                status = e.response.status_code
                logging.error(
                    f"Translation error! HTTP {status}: {e.response.text[:300]}"
                )
                if (
                    self._stopping
                    or self._cancelled
                    or attempt == MAX_ATTEMPTS - 1
                    or not self._is_retryable_status(status)
                ):
                    return None
                self._backoff(attempt, e.response)
            except httpx.TransportError as e:
                logging.error(f"Translation error! Network failure: {e}")
                if self._stopping or self._cancelled or attempt == MAX_ATTEMPTS - 1:
                    return None
                self._backoff(attempt)
            except Exception as e:
                logging.error(f"Translation error! {e}")
                return None
        return None

    def _translate_single(self, transcript: str, transcript_id: int) -> Tuple[str, int]:
        """Translate a single transcript via the API. Returns (translation, transcript_id)."""
        translation = self._messages(
            system=self.transcription_options.llm_prompt, user_content=transcript
        )
        return translation or "", transcript_id

    def _translate_batch(self, items: List[Tuple[str, int]]) -> List[Tuple[str, int]]:
        """Translate multiple transcripts in a single API call.
        Returns list of (translation, transcript_id) in the same order as input."""
        numbered_parts = []
        for i, (transcript, _) in enumerate(items, 1):
            numbered_parts.append(f"[{i}] {transcript}")
        combined = "\n".join(numbered_parts)

        batch_prompt = (
            f"{self.transcription_options.llm_prompt}\n\n"
            f"You will receive {len(items)} numbered texts. "
            f"Process each one separately according to the instruction above "
            f"and return a JSON object mapping each number to its processed text, "
            f'e.g. {{"translations": {{"1": "processed text 1", "2": "processed text 2"}}}}. '
            f"Respond with only that JSON object and nothing else."
        )

        response_text = self._messages(
            system=batch_prompt, user_content=combined, json_mode=True
        )
        if not response_text:
            return [("", tid) for _, tid in items]

        translations = self._parse_batch_response(response_text, len(items))
        missing = []
        by_id = {}
        for translation, (transcript, tid) in zip(translations, items):
            if translation:
                by_id[tid] = translation
            else:
                missing.append((transcript, tid))

        if missing:
            # A malformed/missing entry must not lose the whole batch; re-request
            # only the failed items individually.
            logging.debug(
                f"Batch response incomplete ({len(missing)} missing), "
                "retrying individually"
            )
            for transcript, tid in missing:
                by_id[tid] = self._translate_single(transcript, tid)[0]

        return [(by_id.get(tid, ""), tid) for _, tid in items]

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
    def _split_batches(items: List[Tuple[str, int]]) -> List[List[Tuple[str, int]]]:
        """Split items into sub-batches that fit the character budget."""
        batches = []
        current: List[Tuple[str, int]] = []
        current_chars = 0
        for item in items:
            chars = len(item[0])
            if current and (
                len(current) >= BATCH_SIZE
                or current_chars + chars > MAX_BATCH_CHARS
            ):
                batches.append(current)
                current = []
                current_chars = 0
            current.append(item)
            current_chars += chars
        if current:
            batches.append(current)
        return batches

    def translate_items_sync(
        self, items: List[Tuple[str, int]]
    ) -> List[Tuple[str, int]]:
        """Translate items concurrently while returning results in input order."""
        results: List[Tuple[str, int]] = []
        batches = self._split_batches(items)
        with ThreadPoolExecutor(max_workers=MAX_CONCURRENT_REQUESTS) as executor:
            futures = [
                executor.submit(self._translate_batch, batch)
                for batch in batches
            ]
            for batch, future in zip(batches, futures):
                try:
                    results.extend(future.result())
                except Exception:
                    logging.exception("Translation batch failed")
                    results.extend(("", tid) for _, tid in batch)
        return results

    def _translate_batch_group(
        self, items: List[Tuple[str, int]]
    ) -> List[Tuple[str, int]]:
        results: List[Tuple[str, int]] = []
        for sub_batch in self._split_batches(items):
            if self._cancelled:
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
    ) -> None:
        if not self._cancelled:
            for translation, tid in results:
                if translation:
                    self.translation.emit(translation, tid)
                else:
                    self.translation_failed.emit(tid)
        self.batch_completed.emit()

    def start(self):
        logging.debug("Starting translation queue")
        self._stopping = False

        pending = []
        stop_after_pending = False
        cancel_seen = False
        with ThreadPoolExecutor(max_workers=MAX_CONCURRENT_REQUESTS) as executor:
            while True:
                while (
                    len(pending) < MAX_CONCURRENT_REQUESTS
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
                        cancel_seen = True
                        break

                    # Collect a batch: start with the first item, then wait
                    # briefly for more instead of racing get_nowait() into
                    # single-item requests.
                    batch = [item]
                    batch_window_expired = False
                    deadline = time.monotonic() + BATCH_WINDOW_SECONDS
                    while len(batch) < BATCH_SIZE:
                        remaining = deadline - time.monotonic()
                        if remaining <= 0:
                            batch_window_expired = True
                            break
                        try:
                            next_item = self.queue.get(timeout=remaining)
                        except queue.Empty:
                            batch_window_expired = True
                            break
                        if next_item is None:
                            stop_after_pending = True
                            break
                        if next_item is _CANCEL:
                            # Cancel drops already-collected items; they stay
                            # untranslated and can be re-queued later.
                            cancel_seen = True
                            batch = []
                            break
                        batch.append(next_item)

                    if batch:
                        if not pending and batch_window_expired:
                            try:
                                results = self._translate_batch_group(batch)
                            except Exception:
                                logging.exception("Translation batch failed")
                                results = [("", tid) for _, tid in batch]
                            self._emit_batch_result(results)
                        else:
                            future = executor.submit(
                                self._translate_batch_group, batch
                            )
                            pending.append((future, batch))

                if pending:
                    # Wait for the oldest submitted batch so live-recording
                    # translations remain in input order while requests run
                    # concurrently underneath.
                    future, batch = pending.pop(0)
                    try:
                        results = future.result()
                    except Exception:
                        logging.exception("Translation batch failed")
                        results = [("", tid) for _, tid in batch]
                    self._emit_batch_result(results)
                    continue

                if cancel_seen:
                    self._cancelled = False
                    cancel_seen = False
                    continue

                if stop_after_pending or self._stopping:
                    logging.debug("Translation queue received stop signal")
                    break

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
        # Flag stops retries/backoff immediately; the sentinel unblocks the
        # worker so it exits after the current batch without draining the queue.
        self._stopping = True
        self.queue.put(None)

    def cancel(self):
        """Cancel the current run: drop queued items and discard in-flight
        results. The worker thread stays alive for the next run."""
        self._cancelled = True
        while True:
            try:
                self.queue.get_nowait()
            except queue.Empty:
                break
        self.queue.put(_CANCEL)
