import threading
import time
from unittest.mock import Mock, call, patch

import httpx
from PyQt6.QtCore import QThread

from buzz.translator import Translator
from buzz.transcriber.transcriber import TranscriptionOptions
from buzz.widgets.transcriber.advanced_settings_dialog import AdvancedSettingsDialog


def _failing_response(status_code, retry_after=None):
    resp = Mock(status_code=status_code)
    resp.raise_for_status.side_effect = httpx.HTTPStatusError(
        f"HTTP {status_code}", request=Mock(), response=resp
    )
    resp.headers = {"Retry-After": retry_after} if retry_after else {}
    resp.text = f"error {status_code}"
    return resp


def _success_response(content):
    resp = Mock(status_code=200)
    resp.raise_for_status.return_value = None
    resp.json.return_value = content
    return resp


class TestParseBatchResponse:
    def test_simple_batch(self):
        response = "[1] Hello\n[2] World"
        result = Translator._parse_batch_response(response, 2)
        assert len(result) == 2
        assert result[0] == "Hello"
        assert result[1] == "World"

    def test_missing_entries_fallback(self):
        response = "[1] Hello\n[3] World"
        result = Translator._parse_batch_response(response, 3)
        assert len(result) == 3
        assert result[0] == "Hello"
        assert result[1] == ""
        assert result[2] == "World"

    def test_multiline_entries(self):
        response = "[1] This is a long\nmultiline translation\n[2] Short"
        result = Translator._parse_batch_response(response, 2)
        assert len(result) == 2
        assert "multiline" in result[0]
        assert result[1] == "Short"

    def test_single_item_batch(self):
        response = "[1] Single translation"
        result = Translator._parse_batch_response(response, 1)
        assert len(result) == 1
        assert result[0] == "Single translation"

    def test_empty_response(self):
        response = ""
        result = Translator._parse_batch_response(response, 2)
        assert len(result) == 2
        assert result[0] == ""
        assert result[1] == ""

    def test_whitespace_handling(self):
        response = "[1]   Hello with spaces   \n[2]   World   "
        result = Translator._parse_batch_response(response, 2)
        assert result[0] == "Hello with spaces"
        assert result[1] == "World"

    def test_out_of_order_entries(self):
        response = "[2] Second\n[1] First"
        result = Translator._parse_batch_response(response, 2)
        assert result[0] == "First"
        assert result[1] == "Second"

    def test_json_object_response(self):
        response = '{"translations": {"1": "Hello", "2": "World"}}'
        result = Translator._parse_batch_response(response, 2)
        assert result == ["Hello", "World"]

    def test_json_flat_mapping_response(self):
        response = '{"1": "Hello", "2": "World"}'
        result = Translator._parse_batch_response(response, 2)
        assert result == ["Hello", "World"]

    def test_json_response_with_missing_entry(self):
        response = '{"translations": {"1": "Hello"}}'
        result = Translator._parse_batch_response(response, 2)
        assert result == ["Hello", ""]

    def test_json_fenced_response(self):
        response = '```json\n{"translations": {"1": "Hello"}}\n```'
        result = Translator._parse_batch_response(response, 1)
        assert result == ["Hello"]

    def test_json_response_with_numbering_in_text(self):
        # JSON output is immune to '[N]' appearing inside the text itself.
        response = '{"translations": {"1": "see [2] for details", "2": "done"}}'
        result = Translator._parse_batch_response(response, 2)
        assert result == ["see [2] for details", "done"]


class TestSplitBatches:
    def test_splits_on_char_budget(self):
        items = [("a" * 1500, 1), ("b" * 1500, 2), ("c" * 100, 3)]
        batches = Translator._split_batches(items)
        assert batches == [[items[0], items[1]], [items[2]]]

    def test_single_oversized_item_goes_alone(self):
        items = [("a" * 5000, 1)]
        batches = Translator._split_batches(items)
        assert batches == [[items[0]]]


class TestTranslateItemsSync:
    def test_returns_results_in_input_order(self, qtbot):
        options = TranscriptionOptions(llm_model="gpt-4o-mini", llm_prompt="Translate:")
        translator = Translator(options)

        with patch.object(
            translator,
            "_translate_batch",
            side_effect=lambda batch: [(f"t-{tid}", tid) for _, tid in batch],
        ):
            results = translator.translate_items_sync(
                [("a", 1), ("b", 2), ("c", 3)]
            )

        assert results == [("t-1", 1), ("t-2", 2), ("t-3", 3)]

    def test_failures_come_back_empty_for_retry(self, qtbot):
        options = TranscriptionOptions(llm_model="gpt-4o-mini", llm_prompt="Translate:")
        translator = Translator(options)

        with patch.object(
            translator,
            "_translate_batch",
            return_value=[("ok", 1), ("", 2)],
        ):
            results = translator.translate_items_sync([("a", 1), ("b", 2)])

        assert results == [("ok", 1), ("", 2)]

    def test_splits_oversized_items_into_multiple_batches(self, qtbot):
        options = TranscriptionOptions(llm_model="gpt-4o-mini", llm_prompt="Translate:")
        translator = Translator(options)
        batches = []
        with patch.object(
            translator,
            "_translate_batch",
            side_effect=lambda batch: (batches.append(batch) or
                                       [(f"t-{tid}", tid) for _, tid in batch]),
        ):
            translator.translate_items_sync(
                [("a" * 1500, 1), ("b" * 1500, 2), ("c" * 100, 3)]
            )

        assert len(batches) == 2

    def test_runs_two_batches_concurrently_and_preserves_order(self, qtbot):
        options = TranscriptionOptions(llm_model="gpt-4o-mini", llm_prompt="Translate:")
        translator = Translator(options)
        barrier = threading.Barrier(2, timeout=2)
        lock = threading.Lock()
        state = {"active": 0, "max_active": 0, "broken": False, "sizes": []}

        def translate(batch):
            with lock:
                state["active"] += 1
                state["max_active"] = max(state["max_active"], state["active"])
                state["sizes"].append(len(batch))
            try:
                barrier.wait()
            except threading.BrokenBarrierError:
                state["broken"] = True
            finally:
                with lock:
                    state["active"] -= 1
            return [(f"t-{tid}", tid) for _, tid in batch]

        items = [("text", tid) for tid in range(40)]
        with patch.object(translator, "_translate_batch", side_effect=translate):
            results = translator.translate_items_sync(items)

        assert state["broken"] is False
        assert state["max_active"] == 2
        assert sorted(state["sizes"]) == [20, 20]
        assert [tid for _, tid in results] == list(range(40))


class TestBatchRecovery:
    def _make_translator(self):
        options = TranscriptionOptions(llm_model="gpt-4o-mini", llm_prompt="Translate:")
        return Translator(
            options,
            AdvancedSettingsDialog(transcription_options=options, parent=None),
        )

    def test_parse_failure_retries_missing_as_singles(self, qtbot):
        translator = self._make_translator()
        with patch.object(translator, "_messages", return_value="I cannot do that"), \
                patch.object(
                    translator,
                    "_translate_single",
                    side_effect=lambda text, tid: (f"single-{tid}", tid),
                ):
            results = translator._translate_batch([("a", 1), ("b", 2)])
        assert results == [("single-1", 1), ("single-2", 2)]

    def test_partial_batch_keeps_parsed_and_retries_missing(self, qtbot):
        translator = self._make_translator()
        with patch.object(
                translator,
                "_messages",
                return_value='{"translations": {"1": "parsed"}}'), \
                patch.object(
                    translator,
                    "_translate_single",
                    side_effect=lambda text, tid: (f"single-{tid}", tid),
                ):
            results = translator._translate_batch([("a", 1), ("b", 2)])
        assert results == [("parsed", 1), ("single-2", 2)]

    def test_api_failure_leaves_empties_for_retry(self, qtbot):
        translator = self._make_translator()
        with patch.object(translator, "_messages", return_value=None):
            results = translator._translate_batch([("a", 1), ("b", 2)])
        assert results == [("", 1), ("", 2)]


class TestTranslator:
    @patch('buzz.translator.httpx.Client')
    def test_openai_chat_completions(self, mock_client_class, qtbot, monkeypatch):
        monkeypatch.setenv("BUZZ_TRANSLATION_API_KEY", "openai-key")
        monkeypatch.setenv(
            "BUZZ_TRANSLATION_API_BASE_URL", "https://api.openai.com/v1"
        )
        monkeypatch.delenv("BUZZ_TRANSLATION_API_PROTOCOL", raising=False)

        mock_client = Mock()
        mock_client.post.return_value = _success_response(
            {"choices": [{"message": {"content": "AI Translated"}}]}
        )
        mock_client_class.return_value = mock_client

        transcription_options = TranscriptionOptions(
            enable_llm_translation=False,
            llm_model="gpt-4o-mini",
            llm_prompt="Translate this text:",
        )
        translator = Translator(
            transcription_options,
            AdvancedSettingsDialog(
                transcription_options=transcription_options, parent=None
            ),
        )

        assert translator._messages("Translate this text:", "Hello") == "AI Translated"
        mock_client_class.assert_called_once_with(timeout=30.0)
        mock_client.post.assert_called_once_with(
            "https://api.openai.com/v1/chat/completions",
            headers={
                "Authorization": "Bearer openai-key",
                "Content-Type": "application/json",
            },
            json={
                "model": "gpt-4o-mini",
                "max_tokens": 4096,
                "messages": [
                    {"role": "system", "content": "Translate this text:"},
                    {"role": "user", "content": "Hello"},
                ],
            },
        )

    @patch('buzz.translator.httpx.Client')
    def test_batch_request_uses_json_mode(self, mock_client_class, qtbot, monkeypatch):
        monkeypatch.setenv("BUZZ_TRANSLATION_API_KEY", "openai-key")
        monkeypatch.setenv(
            "BUZZ_TRANSLATION_API_BASE_URL", "https://api.openai.com/v1"
        )

        mock_client = Mock()
        mock_client.post.return_value = _success_response(
            {"choices": [{"message": {"content": '{"translations": {}}'}}]}
        )
        mock_client_class.return_value = mock_client

        options = TranscriptionOptions(llm_model="gpt-4o-mini", llm_prompt="Translate:")
        translator = Translator(
            options,
            AdvancedSettingsDialog(transcription_options=options, parent=None),
        )

        translator._messages("Translate:", "Hello", json_mode=True)
        request_body = mock_client.post.call_args.kwargs["json"]
        assert request_body["response_format"] == {"type": "json_object"}

    @patch('buzz.translator.httpx.Client')
    @patch('buzz.translator.queue.Queue', autospec=True)
    def test_start(self, mock_queue, mock_client_class, qtbot, monkeypatch):
        monkeypatch.setenv("BUZZ_TRANSLATION_API_PROTOCOL", "anthropic")

        def side_effect(*args, **kwargs):
            if side_effect.call_count <= 1:
                side_effect.call_count += 1
                return ("Hello, how are you?", 1)

            # Finally return sentinel to stop
            return None

        side_effect.call_count = 0

        mock_queue.get.side_effect = side_effect
        mock_client = Mock()
        mock_client.post.return_value = _success_response(
            {"content": [{"type": "text", "text": "AI Translated: Hello, how are you?"}]}
        )
        mock_client_class.return_value = mock_client

        transcription_options = TranscriptionOptions(
            enable_llm_translation=False,
            llm_model="llama3",
            llm_prompt="Please translate this text:",
        )
        translator = Translator(
            transcription_options,
            AdvancedSettingsDialog(
                transcription_options=transcription_options, parent=None
            )
        )
        translator.queue = mock_queue

        translator.start()

        mock_queue.get.assert_called()
        mock_client.post.assert_called()

        translator.stop()

    @patch('buzz.translator.queue.Queue', autospec=True)
    def test_start_emits_success_only_for_non_empty_and_failure_signal(
        self, mock_queue, qtbot
    ):
        mock_queue.get.side_effect = [("Hello", 1), ("World", 2), None]

        transcription_options = TranscriptionOptions(
            llm_model="gpt-4o-mini", llm_prompt="Translate this text:",
        )
        translator = Translator(
            transcription_options,
            AdvancedSettingsDialog(
                transcription_options=transcription_options, parent=None
            ),
        )
        translator.queue = mock_queue
        successes = []
        failures = []
        batches = []
        translator.translation.connect(
            lambda text, tid: successes.append((text, tid))
        )
        translator.translation_failed.connect(failures.append)
        translator.batch_completed.connect(lambda: batches.append(True))

        with patch.object(
            translator,
            "_translate_batch",
            return_value=[("AI Translated", 1), ("", 2)],
        ):
            translator.start()

        # Empty results are not emitted as translations; they go to the
        # failure signal so the segment stays pending for retry.
        assert successes == [("AI Translated", 1)]
        assert failures == [2]
        assert batches == [True]

    @patch('buzz.translator.queue.Queue', autospec=True)
    def test_cancel_drops_collected_and_queued_items_but_keeps_worker_alive(
        self, mock_queue, qtbot
    ):
        from buzz.translator import _CANCEL

        # Item 1 is collected, then the cancel marker drops it; item 2 starts
        # a fresh run and the stop sentinel ends the worker.
        mock_queue.get.side_effect = [
            ("Hello", 1), _CANCEL, ("World", 2), None,
        ]

        transcription_options = TranscriptionOptions(
            llm_model="gpt-4o-mini", llm_prompt="Translate this text:",
        )
        translator = Translator(
            transcription_options,
            AdvancedSettingsDialog(
                transcription_options=transcription_options, parent=None
            ),
        )
        translator.queue = mock_queue
        successes = []
        translator.translation.connect(
            lambda text, tid: successes.append((text, tid))
        )

        with patch.object(
            translator,
            "_translate_single",
            side_effect=lambda text, tid: ("translated", tid),
        ):
            translator.start()

        # The cancelled item is dropped; the post-cancel run still works.
        assert successes == [("translated", 2)]

    def test_cancel_drains_pending_queue(self, qtbot):
        transcription_options = TranscriptionOptions(
            llm_model="gpt-4o-mini", llm_prompt="Translate this text:",
        )
        translator = Translator(
            transcription_options,
            AdvancedSettingsDialog(
                transcription_options=transcription_options, parent=None
            ),
        )
        translator.enqueue("a", 1)
        translator.enqueue("b", 2)

        translator.cancel()

        from buzz.translator import _CANCEL
        assert translator.queue.qsize() == 1
        assert translator.queue.get() is _CANCEL

    @patch('buzz.translator.httpx.Client')
    def test_translator(self, mock_client_class, qtbot, monkeypatch):
        monkeypatch.setenv("BUZZ_TRANSLATION_API_PROTOCOL", "anthropic")

        self.on_next_translation_called = False

        def on_next_translation(text: str):
            self.on_next_translation_called = True
            assert text.startswith("AI Translated:")

        mock_client = Mock()
        mock_client.post.return_value = _success_response(
            {"content": [{"type": "text", "text": "AI Translated: Hello, how are you?"}]}
        )
        mock_client_class.return_value = mock_client

        self.translation_thread = QThread()
        self.transcription_options = TranscriptionOptions(
            enable_llm_translation=False,
            llm_model="llama3",
            llm_prompt="Please translate this text:",
        )

        self.translator = Translator(
            self.transcription_options,
            AdvancedSettingsDialog(
                transcription_options=self.transcription_options, parent=None
            )
        )

        self.translator.moveToThread(self.translation_thread)

        self.translation_thread.started.connect(self.translator.start)
        self.translation_thread.finished.connect(
            self.translation_thread.deleteLater
        )

        self.translator.finished.connect(self.translation_thread.quit)
        self.translator.finished.connect(self.translator.deleteLater)

        self.translator.translation.connect(on_next_translation)

        self.translation_thread.start()

        time.sleep(1)  # Give thread time to start

        self.translator.enqueue("Hello, how are you?")

        def translation_signal_received():
            assert self.on_next_translation_called

        qtbot.wait_until(translation_signal_received, timeout=60 * 1000)

        if self.translator is not None:
            self.translator.stop()

        if self.translation_thread is not None:
            self.translation_thread.quit()
            # Wait for the thread to actually finish before cleanup
            self.translation_thread.wait()
            # Process pending events to ensure deleteLater() is handled
            from PyQt6.QtCore import QCoreApplication
            QCoreApplication.processEvents()
            time.sleep(0.1)  # Give time for cleanup

        # Note: translator and translation_thread will be automatically deleted
        # via the deleteLater() connections set up earlier

    @patch('buzz.translator.time.sleep')
    @patch('buzz.translator.httpx.Client')
    def test_messages_retries_until_success(
        self, mock_client_class, mock_sleep, qtbot, monkeypatch
    ):
        monkeypatch.setenv("BUZZ_TRANSLATION_API_KEY", "openai-key")
        monkeypatch.setenv("BUZZ_TRANSLATION_API_BASE_URL", "https://api.openai.com/v1")

        mock_client = Mock()
        mock_client.post.side_effect = [_failing_response(500)] * 3 + [
            _success_response({"choices": [{"message": {"content": "AI Translated"}}]})
        ]
        mock_client_class.return_value = mock_client
        options = TranscriptionOptions(llm_model="gpt-4o-mini", llm_prompt="Translate:")
        translator = Translator(
            options,
            AdvancedSettingsDialog(transcription_options=options, parent=None),
        )

        assert translator._messages("Translate:", "Hello") == "AI Translated"
        assert mock_client.post.call_count == 4

    @patch('buzz.translator.time.sleep')
    @patch('buzz.translator.httpx.Client')
    def test_messages_retries_429_with_retry_after(
        self, mock_client_class, mock_sleep, qtbot, monkeypatch
    ):
        monkeypatch.setenv("BUZZ_TRANSLATION_API_KEY", "openai-key")
        monkeypatch.setenv("BUZZ_TRANSLATION_API_BASE_URL", "https://api.openai.com/v1")

        mock_client = Mock()
        mock_client.post.side_effect = [
            _failing_response(429, retry_after="2"),
            _success_response({"choices": [{"message": {"content": "AI Translated"}}]}),
        ]
        mock_client_class.return_value = mock_client
        options = TranscriptionOptions(llm_model="gpt-4o-mini", llm_prompt="Translate:")
        translator = Translator(
            options,
            AdvancedSettingsDialog(transcription_options=options, parent=None),
        )

        assert translator._messages("Translate:", "Hello") == "AI Translated"
        assert mock_client.post.call_count == 2
        # delay = max(2**0, min(2, 30)) = 2.0s, slept in 0.5s slices
        assert mock_sleep.call_count == 4
        mock_sleep.assert_has_calls([call(0.5)] * 4)

    @patch('buzz.translator.time.sleep')
    @patch('buzz.translator.httpx.Client')
    def test_messages_retries_transient_network_errors(
        self, mock_client_class, mock_sleep, qtbot, monkeypatch
    ):
        monkeypatch.setenv("BUZZ_TRANSLATION_API_KEY", "openai-key")
        monkeypatch.setenv("BUZZ_TRANSLATION_API_BASE_URL", "https://api.openai.com/v1")

        mock_client = Mock()
        mock_client.post.side_effect = [
            httpx.ConnectError("connection refused"),
            httpx.ConnectError("connection refused"),
            _success_response({"choices": [{"message": {"content": "AI Translated"}}]}),
        ]
        mock_client_class.return_value = mock_client
        options = TranscriptionOptions(llm_model="gpt-4o-mini", llm_prompt="Translate:")
        translator = Translator(
            options,
            AdvancedSettingsDialog(transcription_options=options, parent=None),
        )

        assert translator._messages("Translate:", "Hello") == "AI Translated"
        assert mock_client.post.call_count == 3

    @patch('buzz.translator.httpx.Client')
    def test_messages_does_not_retry_auth_error(
        self, mock_client_class, qtbot, monkeypatch
    ):
        monkeypatch.setenv("BUZZ_TRANSLATION_API_KEY", "openai-key")
        monkeypatch.setenv("BUZZ_TRANSLATION_API_BASE_URL", "https://api.openai.com/v1")

        mock_client = Mock()
        mock_client.post.side_effect = [_failing_response(401)]
        mock_client_class.return_value = mock_client
        options = TranscriptionOptions(llm_model="gpt-4o-mini", llm_prompt="Translate:")
        translator = Translator(
            options,
            AdvancedSettingsDialog(transcription_options=options, parent=None),
        )

        assert translator._messages("Translate:", "Hello") is None
        assert mock_client.post.call_count == 1

    @patch('buzz.translator.httpx.Client')
    def test_messages_does_not_retry_unexpected_errors(
        self, mock_client_class, qtbot, monkeypatch
    ):
        monkeypatch.setenv("BUZZ_TRANSLATION_API_KEY", "openai-key")
        monkeypatch.setenv("BUZZ_TRANSLATION_API_BASE_URL", "https://api.openai.com/v1")

        mock_client = Mock()
        mock_client.post.side_effect = [Exception("temporary failure")]
        mock_client_class.return_value = mock_client
        options = TranscriptionOptions(llm_model="gpt-4o-mini", llm_prompt="Translate:")
        translator = Translator(
            options,
            AdvancedSettingsDialog(transcription_options=options, parent=None),
        )

        assert translator._messages("Translate:", "Hello") is None
        assert mock_client.post.call_count == 1
