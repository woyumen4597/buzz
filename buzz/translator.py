import os
import re
import logging
import queue

from typing import Optional, List, Tuple
import httpx
from PyQt6.QtCore import QObject, pyqtSignal

from buzz.settings.settings import Settings
from buzz.store.keyring_store import get_password, Key
from buzz.transcriber.transcriber import TranscriptionOptions
from buzz.widgets.transcriber.advanced_settings_dialog import AdvancedSettingsDialog


BATCH_SIZE = 10

DEFAULT_ANTHROPIC_BASE_URL = "https://api.anthropic.com"
ANTHROPIC_VERSION = "2023-06-01"
REQUEST_TIMEOUT = 180.0


def _messages_url(base_url: str) -> str:
    # base_url may or may not end in /v1; normalize so we hit .../v1/messages.
    base = base_url.rstrip("/")
    if base.endswith("/v1"):
        return base + "/messages"
    return base + "/v1/messages"


class Translator(QObject):
    translation = pyqtSignal(str, int)
    finished = pyqtSignal()

    def __init__(
        self,
        transcription_options: TranscriptionOptions,
        advanced_settings_dialog: AdvancedSettingsDialog,
        parent: Optional[QObject] = None,
    ) -> None:
        super().__init__(parent)

        logging.debug(f"Translator init: {transcription_options}")

        self.transcription_options = transcription_options
        self.advanced_settings_dialog = advanced_settings_dialog
        self.advanced_settings_dialog.transcription_options_changed.connect(
            self.on_transcription_options_changed
        )

        self.queue = queue.Queue()

        settings = Settings()
        self.api_key = os.getenv(
            "BUZZ_TRANSLATION_API_KEY", get_password(Key.OPENAI_API_KEY)
        )
        self.base_url = os.getenv(
            "BUZZ_TRANSLATION_API_BASE_URl",
            settings.value(Settings.Key.CUSTOM_OPENAI_BASE_URL, ""),
        ) or DEFAULT_ANTHROPIC_BASE_URL

        # ponytail: per-task llm_model wins, fall back to global preferences model.
        self.llm_model = self.transcription_options.llm_model or os.getenv(
            "BUZZ_TRANSLATION_API_MODEL",
            settings.value(Settings.Key.OPENAI_API_MODEL, ""),
        )

    def _messages(self, system: str, user_content: str) -> Optional[str]:
        """Call Anthropic Messages API via httpx. Returns top text block or None on error."""
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": ANTHROPIC_VERSION,
            "content-type": "application/json",
        }
        body = {
            "model": self.llm_model,
            "max_tokens": 4096,
            "system": system,
            "messages": [{"role": "user", "content": user_content}],
        }
        try:
            resp = httpx.post(
                _messages_url(self.base_url),
                headers=headers,
                json=body,
                timeout=REQUEST_TIMEOUT,
            )
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            logging.error(f"Translation error! Server response: {e}")
            return None

        content = data.get("content", [])
        if content and isinstance(content, list) and content[0].get("text"):
            logging.debug(f"Received translation response: {data}")
            return content[0]["text"]
        logging.error(f"Translation error! Unexpected response: {data}")
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
            f"and return them in the exact same numbered format, e.g.:\n"
            f"[1] processed text\n[2] processed text"
        )

        response_text = self._messages(system=batch_prompt, user_content=combined)
        if not response_text:
            return [("", tid) for _, tid in items]

        translations = self._parse_batch_response(response_text, len(items))
        return [
            (translations[i], items[i][1]) if i < len(translations) else ("", items[i][1])
            for i in range(len(items))
        ]

    @staticmethod
    def _parse_batch_response(response: str, expected_count: int) -> List[str]:
        """Parse a numbered batch response like '[1] text\\n[2] text' into a list of strings."""
        # Split on [N] markers — re.split with a group returns: [before, group1, after1, group2, after2, ...]
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

    def start(self):
        logging.debug("Starting translation queue")

        while True:
            item = self.queue.get()  # Block until item available

            # Check for sentinel value (None means stop)
            if item is None:
                logging.debug("Translation queue received stop signal")
                break

            # Collect a batch: start with the first item, then drain more
            batch = [item]
            stop_after_batch = False
            while len(batch) < BATCH_SIZE:
                try:
                    next_item = self.queue.get_nowait()
                    if next_item is None:
                        stop_after_batch = True
                        break
                    batch.append(next_item)
                except queue.Empty:
                    break

            if len(batch) == 1:
                transcript, transcript_id = batch[0]
                translation, tid = self._translate_single(transcript, transcript_id)
                self.translation.emit(translation, tid)
            else:
                logging.debug(f"Translating batch of {len(batch)} in single request")
                results = self._translate_batch(batch)
                for translation, tid in results:
                    self.translation.emit(translation, tid)

            if stop_after_batch:
                logging.debug("Translation queue received stop signal")
                break

        logging.debug("Translation queue stopped")
        self.finished.emit()

    def on_transcription_options_changed(
        self, transcription_options: TranscriptionOptions
    ):
        self.transcription_options = transcription_options
        # ponytail: re-resolve model so a per-task change wins over the global fallback.
        self.llm_model = transcription_options.llm_model or Settings().value(
            Settings.Key.OPENAI_API_MODEL, ""
        )

    def enqueue(self, transcript: str, transcript_id: Optional[int] = None):
        self.queue.put((transcript, transcript_id))

    def stop(self):
        # Send sentinel value to unblock and stop the worker thread
        self.queue.put(None)
