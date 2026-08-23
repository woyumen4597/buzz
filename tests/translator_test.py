import threading
import time
from unittest.mock import Mock, call, patch, MagicMock

import httpx
from PyQt6.QtCore import QThread

from buzz.settings.settings import (
    DEFAULT_TRANSLATION_BATCH_SIZE,
    DEFAULT_TRANSLATION_CONCURRENCY,
    MAX_TRANSLATION_BATCH_SIZE,
    MAX_TRANSLATION_CONCURRENCY,
    MIN_TRANSLATION_BATCH_SIZE,
    MIN_TRANSLATION_CONCURRENCY,
    Settings,
)
from buzz.translator import (
    CHAT_COMPLETIONS_PROTOCOL,
    RESPONSES_PROTOCOL,
    Translator,
)
from buzz.transcriber.transcriber import TranscriptionOptions
from buzz.widgets.transcriber.advanced_settings_dialog import AdvancedSettingsDialog


def _sse_lines(text, protocol=CHAT_COMPLETIONS_PROTOCOL):
    """Convert text into SSE delta lines for streaming response."""
    lines = []
    if protocol == RESPONSES_PROTOCOL:
        for char in text:
            lines.append(f'data: {{"type":"response.output_text.delta","delta":"{char}"}}')
    else:
        for char in text:
            lines.append(f'data: {{"choices":[{{"delta":{{"content":"{char}"}}}}]}}')
    lines.append("data: [DONE]")
    return lines


def _stream_response(text=None, lines=None, protocol=CHAT_COMPLETIONS_PROTOCOL):
    """Create a mock streaming response that supports context manager."""
    if lines is None:
        lines = _sse_lines(text or "", protocol)

    resp = MagicMock()
    resp.__enter__ = MagicMock(return_value=resp)
    resp.__exit__ = MagicMock(return_value=None)
    resp.raise_for_status = MagicMock(return_value=None)
    resp.iter_lines = MagicMock(return_value=iter(lines))
    return resp


def _failing_response(status_code, retry_after=None):
    resp = MagicMock()
    resp.__enter__ = MagicMock(return_value=resp)
    resp.__exit__ = MagicMock(return_value=None)
    resp.status_code = status_code
    resp.raise_for_status.side_effect = httpx.HTTPStatusError(
        f"HTTP {status_code}", request=Mock(), response=resp
    )
    resp.headers = {"Retry-After": retry_after} if retry_after else {}
    resp.text = f"error {status_code}"
    return resp


def _success_response(content):
    resp = MagicMock()
    resp.__enter__ = MagicMock(return_value=resp)
    resp.__exit__ = MagicMock(return_value=None)
    resp.status_code = 200
    resp.raise_for_status.return_value = None
    resp.json.return_value = content
    lines = []
    if "choices" in content and content["choices"]:
        # Chat completions protocol - extract message content
        text = content["choices"][0].get("message", {}).get("content", "")
        lines = _sse_lines(text, CHAT_COMPLETIONS_PROTOCOL)
    elif "output" in content:
        # Responses protocol - extract text from output
        text = ""
        for item in content.get("output", []):
            if item.get("type") == "message":
                for block in item.get("content", []):
                    if block.get("type") == "output_text":
                        text += block.get("text", "")
        lines = _sse_lines(text, RESPONSES_PROTOCOL)
    resp.iter_lines = MagicMock(return_value=iter(lines))
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

    def test_splits_on_conservative_token_budget(self):
        items = [("中" * 6, 1), ("中" * 6, 2)]
        assert Translator._estimate_tokens(items[0][0]) == 6
        assert Translator._split_batches(items, max_tokens=6) == [
            [items[0]],
            [items[1]],
        ]

    def test_honours_explicit_batch_size(self):
        items = [("a", i) for i in range(1, 6)]
        assert Translator._split_batches(items, batch_size=2) == [
            [items[0], items[1]],
            [items[2], items[3]],
            [items[4]],
        ]

    def test_falls_back_to_default_batch_size(self):
        items = [("a", i) for i in range(1, DEFAULT_TRANSLATION_BATCH_SIZE + 2)]
        batches = Translator._split_batches(items)
        assert len(batches[0]) == DEFAULT_TRANSLATION_BATCH_SIZE
        assert len(batches[1]) == 1


class TestTranslationBatchSizeConfig:
    def _make_translator(self):
        options = TranscriptionOptions(llm_model="gpt-4o-mini", llm_prompt="Translate:")
        return Translator(options)

    def test_defaults_to_configured_default(self, monkeypatch):
        monkeypatch.delenv("BUZZ_TRANSLATION_BATCH_SIZE", raising=False)
        with patch(
            "buzz.translator.Settings.value",
            side_effect=lambda key, default_value=None, *a, **kw: default_value,
        ):
            translator = self._make_translator()
        assert translator.batch_size == DEFAULT_TRANSLATION_BATCH_SIZE

    def test_reads_stored_preference(self, monkeypatch):
        monkeypatch.delenv("BUZZ_TRANSLATION_BATCH_SIZE", raising=False)

        def fake_value(key, default_value=None, *args, **kwargs):
            if key == Settings.Key.TRANSLATION_BATCH_SIZE:
                return 7
            return default_value

        with patch("buzz.translator.Settings.value", side_effect=fake_value):
            translator = self._make_translator()
        assert translator.batch_size == 7

    def test_env_var_overrides_preference(self, monkeypatch):
        monkeypatch.setenv("BUZZ_TRANSLATION_BATCH_SIZE", "3")

        def fake_value(key, default_value=None, *args, **kwargs):
            if key == Settings.Key.TRANSLATION_BATCH_SIZE:
                return 50
            return default_value

        with patch("buzz.translator.Settings.value", side_effect=fake_value):
            translator = self._make_translator()
        assert translator.batch_size == 3

    def test_clamps_out_of_range_values(self, monkeypatch):
        monkeypatch.setenv("BUZZ_TRANSLATION_BATCH_SIZE", "0")
        with patch(
            "buzz.translator.Settings.value",
            side_effect=lambda key, default_value=None, *a, **kw: default_value,
        ):
            assert self._make_translator().batch_size == MIN_TRANSLATION_BATCH_SIZE

        monkeypatch.setenv("BUZZ_TRANSLATION_BATCH_SIZE", "99999")
        with patch(
            "buzz.translator.Settings.value",
            side_effect=lambda key, default_value=None, *a, **kw: default_value,
        ):
            assert self._make_translator().batch_size == MAX_TRANSLATION_BATCH_SIZE


class TestTranslationConcurrencyConfig:
    def _make_translator(self):
        options = TranscriptionOptions(llm_model="gpt-4o-mini", llm_prompt="Translate:")
        return Translator(options)

    def test_defaults_to_configured_default(self, monkeypatch):
        monkeypatch.delenv("BUZZ_TRANSLATION_CONCURRENCY", raising=False)
        with patch(
            "buzz.translator.Settings.value",
            side_effect=lambda key, default_value=None, *a, **kw: default_value,
        ):
            translator = self._make_translator()
        assert translator.max_concurrent_requests == DEFAULT_TRANSLATION_CONCURRENCY

    def test_reads_stored_preference(self, monkeypatch):
        monkeypatch.delenv("BUZZ_TRANSLATION_CONCURRENCY", raising=False)

        def fake_value(key, default_value=None, *args, **kwargs):
            if key == Settings.Key.TRANSLATION_CONCURRENCY:
                return 6
            return default_value

        with patch("buzz.translator.Settings.value", side_effect=fake_value):
            translator = self._make_translator()
        assert translator.max_concurrent_requests == 6

    def test_env_var_overrides_preference(self, monkeypatch):
        monkeypatch.setenv("BUZZ_TRANSLATION_CONCURRENCY", "5")

        def fake_value(key, default_value=None, *args, **kwargs):
            if key == Settings.Key.TRANSLATION_CONCURRENCY:
                return 2
            return default_value

        with patch("buzz.translator.Settings.value", side_effect=fake_value):
            translator = self._make_translator()
        assert translator.max_concurrent_requests == 5

    def test_clamps_out_of_range_values(self, monkeypatch):
        monkeypatch.setenv("BUZZ_TRANSLATION_CONCURRENCY", "0")
        with patch(
            "buzz.translator.Settings.value",
            side_effect=lambda key, default_value=None, *a, **kw: default_value,
        ):
            assert (
                self._make_translator().max_concurrent_requests
                == MIN_TRANSLATION_CONCURRENCY
            )

        monkeypatch.setenv("BUZZ_TRANSLATION_CONCURRENCY", "99999")
        with patch(
            "buzz.translator.Settings.value",
            side_effect=lambda key, default_value=None, *a, **kw: default_value,
        ):
            assert (
                self._make_translator().max_concurrent_requests
                == MAX_TRANSLATION_CONCURRENCY
            )


class TestTranslationProxyConfig:
    def _make_translator(self):
        options = TranscriptionOptions(llm_model="gpt-4o-mini", llm_prompt="Translate:")
        return Translator(options)

    def test_reads_proxy_from_environment(self, monkeypatch):
        monkeypatch.setenv("BUZZ_TRANSLATION_PROXY", "http://127.0.0.1:10808")
        with patch(
            "buzz.translator.Settings.value",
            side_effect=lambda key, default_value=None, *a, **kw: default_value,
        ):
            translator = self._make_translator()
        assert translator.proxy == "http://127.0.0.1:10808"

    def test_environment_proxy_overrides_stored_preference(self, monkeypatch):
        monkeypatch.setenv("BUZZ_TRANSLATION_PROXY", "http://env-proxy:8080")

        def fake_value(key, default_value=None, *args, **kwargs):
            if key == Settings.Key.TRANSLATION_PROXY:
                return "http://stored-proxy:8080"
            return default_value

        with patch("buzz.translator.Settings.value", side_effect=fake_value):
            translator = self._make_translator()
        assert translator.proxy == "http://env-proxy:8080"

    def test_passes_proxy_to_httpx_client(self, monkeypatch):
        monkeypatch.setenv("BUZZ_TRANSLATION_PROXY", "http://127.0.0.1:10808")
        with patch(
            "buzz.translator.Settings.value",
            side_effect=lambda key, default_value=None, *a, **kw: default_value,
        ), patch("buzz.translator.httpx.Client") as client_class:
            translator = self._make_translator()
            translator._client()
        assert client_class.call_args.kwargs["proxy"] == "http://127.0.0.1:10808"


class TestTranslationRuntimeControls:
    def _make_translator(self):
        options = TranscriptionOptions(llm_model="gpt-4o-mini", llm_prompt="Translate:")
        return Translator(options)

    def test_cache_keyed_by_model_prompt_and_source(self, qtbot):
        translator = self._make_translator()
        with patch.object(translator, "_messages", return_value="translated") as messages:
            assert translator._translate_single("hello", 1) == ("translated", 1)
            assert translator._translate_single("hello", 2) == ("translated", 2)
            translator.llm_model = "other-model"
            assert translator._translate_single("hello", 3) == ("translated", 3)
        assert messages.call_count == 2

    def test_partial_batch_result_is_reused_after_cancel(self, qtbot):
        translator = self._make_translator()
        with patch.object(
            translator,
            "_messages",
            side_effect=['{"translations": {"1": "first"}}', None],
        ) as messages:
            assert translator._translate_batch([("a", 1), ("b", 2)]) == [
                ("first", 1),
                ("", 2),
            ]
            translator.cancel()
            assert translator._translate_single("a", 99) == ("first", 99)
        assert messages.call_count == 2

    def test_queue_emits_completed_batch_without_head_of_line_blocking(self, qtbot):
        translator = self._make_translator()
        for tid in range(40):
            translator.queue.put((f"text-{tid}", tid))
        translator.queue.put(None)
        first_started = threading.Event()
        second_finished = threading.Event()
        second_emitted = threading.Event()
        release_first = threading.Event()
        emitted = []

        def translate(batch):
            if batch[0][1] == 0:
                first_started.set()
                release_first.wait(2)
            else:
                second_finished.set()
            return [(f"t-{tid}", tid) for _, tid in batch]

        def emit(results, generation=None):
            emitted.append(results)
            if results and results[0][1] == 20:
                second_emitted.set()

        with patch.object(translator, "_translate_batch_group", side_effect=translate), patch.object(
            translator, "_emit_batch_result", side_effect=emit
        ):
            worker = threading.Thread(target=translator.start)
            worker.start()
            assert first_started.wait(2)
            assert second_finished.wait(2)
            assert second_emitted.wait(2)
            assert emitted and emitted[0][0][1] == 20
            release_first.set()
            worker.join(2)
        assert not worker.is_alive()

    def test_cancel_closes_in_flight_client_and_interrupts_wait(self, qtbot):
        translator = self._make_translator()
        started = threading.Event()
        closed = threading.Event()
        client = Mock()

        def stream(*args, **kwargs):
            started.set()
            closed.wait(2)
            raise httpx.ConnectError("closed")

        client.stream.side_effect = stream
        client.close.side_effect = closed.set
        translator._client_instance = client
        result = []
        worker = threading.Thread(
            target=lambda: result.append(translator._messages("Translate:", "hello"))
        )
        worker.start()
        assert started.wait(2)
        translator.cancel()
        worker.join(1)
        assert not worker.is_alive()
        assert client.close.called
        assert result == [None]

    def test_rate_limiter_wait_is_cancelable(self, monkeypatch, qtbot):
        monkeypatch.setenv("BUZZ_TRANSLATION_REQUESTS_PER_MINUTE", "1")
        translator = self._make_translator()
        translator._rate_tokens = 0
        generation, cancel_event = translator._capture_generation()
        result = []
        worker = threading.Thread(
            target=lambda: result.append(
                translator._acquire_rate_limit(generation, cancel_event)
            )
        )
        worker.start()
        time.sleep(0.05)
        translator.cancel()
        worker.join(1)
        assert not worker.is_alive()
        assert result == [False]

    def test_default_logs_do_not_contain_response_text(self, caplog, qtbot):
        translator = self._make_translator()
        translator.api_protocol = CHAT_COMPLETIONS_PROTOCOL
        caplog.set_level("DEBUG")
        secret = "private translated response"
        assert translator._extract_text(
            {"choices": [{"message": {"content": secret}}]}
        ) == secret
        assert secret not in caplog.text


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
        # Concurrency and batch size are user-configurable, so pin both here;
        # this test is about ordering, not about the configured defaults.
        translator.max_concurrent_requests = 2
        translator.batch_size = 20
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

    def test_configured_concurrency_raises_batches_in_flight(self, monkeypatch, qtbot):
        monkeypatch.setenv("BUZZ_TRANSLATION_CONCURRENCY", "4")
        monkeypatch.delenv("BUZZ_TRANSLATION_BATCH_SIZE", raising=False)
        options = TranscriptionOptions(llm_model="gpt-4o-mini", llm_prompt="Translate:")
        with patch(
            "buzz.translator.Settings.value",
            side_effect=lambda key, default_value=None, *a, **kw: default_value,
        ):
            translator = Translator(options)
        assert translator.max_concurrent_requests == 4

        # Only releases once all four batches are in flight at the same time,
        # so it cannot pass while concurrency is capped at 2.
        barrier = threading.Barrier(4, timeout=5)
        lock = threading.Lock()
        state = {"active": 0, "max_active": 0, "broken": False}

        def translate(batch):
            with lock:
                state["active"] += 1
                state["max_active"] = max(state["max_active"], state["active"])
            try:
                barrier.wait()
            except threading.BrokenBarrierError:
                state["broken"] = True
            finally:
                with lock:
                    state["active"] -= 1
            return [(f"t-{tid}", tid) for _, tid in batch]

        items = [("text", tid) for tid in range(80)]
        with patch.object(translator, "_translate_batch", side_effect=translate):
            results = translator.translate_items_sync(items)

        assert state["broken"] is False
        assert state["max_active"] == 4
        assert [tid for _, tid in results] == list(range(80))


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
        monkeypatch.setenv("BUZZ_TRANSLATION_API_PROTOCOL", CHAT_COMPLETIONS_PROTOCOL)

        mock_client = Mock()
        mock_client.stream.return_value = _stream_response(
            "AI Translated", protocol=CHAT_COMPLETIONS_PROTOCOL
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
        mock_client_class.assert_called_once_with(timeout=httpx.Timeout(connect=10.0, read=30.0, write=30.0, pool=10.0))
        mock_client.stream.assert_called_once_with(
            "POST",
            "https://api.openai.com/v1/chat/completions",
            headers={
                "Authorization": "Bearer openai-key",
                "Content-Type": "application/json",
            },
            json={
                "model": "gpt-4o-mini",
                "max_tokens": 8192,
                "stream": True,
                "messages": [
                    {"role": "system", "content": "Translate this text:"},
                    {"role": "user", "content": "Hello"},
                ],
            },
        )

    @patch('buzz.translator.httpx.Client')
    def test_openai_responses(self, mock_client_class, qtbot, monkeypatch):
        monkeypatch.setenv("BUZZ_TRANSLATION_API_KEY", "openai-key")
        monkeypatch.setenv("BUZZ_TRANSLATION_API_BASE_URL", "https://api.openai.com/v1")
        monkeypatch.setenv("BUZZ_TRANSLATION_API_PROTOCOL", RESPONSES_PROTOCOL)

        mock_client = Mock()
        mock_client.stream.return_value = _stream_response(
            "AI Translated", protocol=RESPONSES_PROTOCOL
        )
        mock_client_class.return_value = mock_client

        options = TranscriptionOptions(
            llm_model="gpt-4o-mini", llm_prompt="Translate this text:"
        )
        translator = Translator(
            options,
            AdvancedSettingsDialog(transcription_options=options, parent=None),
        )

        assert (
            translator._messages("Translate this text:", "Hello", json_mode=True)
            == "AI Translated"
        )
        mock_client.stream.assert_called_once_with(
            "POST",
            "https://api.openai.com/v1/responses",
            headers={
                "Authorization": "Bearer openai-key",
                "Content-Type": "application/json",
            },
            json={
                "model": "gpt-4o-mini",
                "max_output_tokens": 8192,
                "stream": True,
                "input": [
                    {
                        "role": "system",
                        "content": [
                            {"type": "input_text", "text": "Translate this text:"}
                        ],
                    },
                    {
                        "role": "user",
                        "content": [{"type": "input_text", "text": "Hello"}],
                    },
                ],
                "text": {"format": {"type": "json_object"}},
            },
        )

    @patch('buzz.translator.httpx.Client')
    def test_batch_request_uses_json_mode(self, mock_client_class, qtbot, monkeypatch):
        monkeypatch.setenv("BUZZ_TRANSLATION_API_KEY", "openai-key")
        monkeypatch.setenv(
            "BUZZ_TRANSLATION_API_BASE_URL", "https://api.openai.com/v1"
        )
        monkeypatch.setenv("BUZZ_TRANSLATION_API_PROTOCOL", CHAT_COMPLETIONS_PROTOCOL)

        mock_client = Mock()
        mock_client.stream.return_value = _stream_response(
            '{"translations": {}}'
        )
        mock_client_class.return_value = mock_client

        options = TranscriptionOptions(llm_model="gpt-4o-mini", llm_prompt="Translate:")
        translator = Translator(
            options,
            AdvancedSettingsDialog(transcription_options=options, parent=None),
        )

        translator._messages("Translate:", "Hello", json_mode=True)
        request_body = mock_client.stream.call_args.kwargs["json"]
        assert request_body["response_format"] == {"type": "json_object"}

    @patch('buzz.translator.httpx.Client')
    def test_deepseek_responses_defaults_to_no_reasoning(
        self, mock_client_class, qtbot, monkeypatch
    ):
        """DeepSeek defaults thinking ON and counts it against
        max_output_tokens; the responses request must opt out so the token
        budget goes to visible translation output."""
        monkeypatch.setenv("BUZZ_TRANSLATION_API_KEY", "deepseek-key")
        monkeypatch.setenv(
            "BUZZ_TRANSLATION_API_BASE_URL", "https://api.deepseek.com/v1"
        )
        monkeypatch.setenv("BUZZ_TRANSLATION_API_PROTOCOL", RESPONSES_PROTOCOL)

        mock_client = Mock()
        mock_client.stream.return_value = _stream_response(
            "翻译", protocol=RESPONSES_PROTOCOL
        )
        mock_client_class.return_value = mock_client

        options = TranscriptionOptions(
            llm_model="deepseek-chat", llm_prompt="Translate this text:"
        )
        translator = Translator(
            options,
            AdvancedSettingsDialog(transcription_options=options, parent=None),
        )

        translator._messages("Translate this text:", "Hello")
        request_body = mock_client.stream.call_args.kwargs["json"]
        assert request_body["reasoning"] == {"effort": "none"}
        assert request_body["max_output_tokens"] == 8192

    @patch('buzz.translator.httpx.Client')
    def test_deepseek_reasoner_does_not_force_reasoning_off(
        self, mock_client_class, qtbot, monkeypatch
    ):
        """deepseek-reasoner always reasons; sending effort=none would be
        ignored or rejected, so leave the reasoning knob out entirely."""
        monkeypatch.setenv("BUZZ_TRANSLATION_API_KEY", "deepseek-key")
        monkeypatch.setenv(
            "BUZZ_TRANSLATION_API_BASE_URL", "https://api.deepseek.com/v1"
        )
        monkeypatch.setenv("BUZZ_TRANSLATION_API_PROTOCOL", RESPONSES_PROTOCOL)

        mock_client = Mock()
        mock_client.stream.return_value = _stream_response(
            "翻译", protocol=RESPONSES_PROTOCOL
        )
        mock_client_class.return_value = mock_client

        options = TranscriptionOptions(
            llm_model="deepseek-reasoner", llm_prompt="Translate this text:"
        )
        translator = Translator(
            options,
            AdvancedSettingsDialog(transcription_options=options, parent=None),
        )

        translator._messages("Translate this text:", "Hello")
        request_body = mock_client.stream.call_args.kwargs["json"]
        assert "reasoning" not in request_body

    @patch('buzz.translator.httpx.Client')
    def test_reasoning_effort_env_override(
        self, mock_client_class, qtbot, monkeypatch
    ):
        """BUZZ_TRANSLATION_REASONING_EFFORT wins over the DeepSeek default."""
        monkeypatch.setenv("BUZZ_TRANSLATION_API_KEY", "deepseek-key")
        monkeypatch.setenv(
            "BUZZ_TRANSLATION_API_BASE_URL", "https://api.deepseek.com/v1"
        )
        monkeypatch.setenv("BUZZ_TRANSLATION_API_PROTOCOL", RESPONSES_PROTOCOL)
        monkeypatch.setenv("BUZZ_TRANSLATION_REASONING_EFFORT", "medium")

        mock_client = Mock()
        mock_client.stream.return_value = _stream_response(
            "翻译", protocol=RESPONSES_PROTOCOL
        )
        mock_client_class.return_value = mock_client

        options = TranscriptionOptions(
            llm_model="deepseek-chat", llm_prompt="Translate this text:"
        )
        translator = Translator(
            options,
            AdvancedSettingsDialog(transcription_options=options, parent=None),
        )

        translator._messages("Translate this text:", "Hello")
        request_body = mock_client.stream.call_args.kwargs["json"]
        assert request_body["reasoning"] == {"effort": "medium"}

    @patch('buzz.translator.httpx.Client')
    def test_max_output_tokens_env_override(
        self, mock_client_class, qtbot, monkeypatch
    ):
        monkeypatch.setenv("BUZZ_TRANSLATION_API_KEY", "openai-key")
        monkeypatch.setenv(
            "BUZZ_TRANSLATION_API_BASE_URL", "https://api.openai.com/v1"
        )
        monkeypatch.setenv("BUZZ_TRANSLATION_API_PROTOCOL", RESPONSES_PROTOCOL)
        monkeypatch.setenv("BUZZ_TRANSLATION_MAX_OUTPUT_TOKENS", "16384")

        mock_client = Mock()
        mock_client.stream.return_value = _stream_response(
            "AI Translated", protocol=RESPONSES_PROTOCOL
        )
        mock_client_class.return_value = mock_client

        options = TranscriptionOptions(
            llm_model="gpt-4o-mini", llm_prompt="Translate this text:"
        )
        translator = Translator(
            options,
            AdvancedSettingsDialog(transcription_options=options, parent=None),
        )

        translator._messages("Translate this text:", "Hello")
        request_body = mock_client.stream.call_args.kwargs["json"]
        assert request_body["max_output_tokens"] == 16384
        # Non-DeepSeek endpoints keep the legacy behavior: no reasoning knob.
        assert "reasoning" not in request_body

    def test_read_stream_reports_incomplete_reason(self, qtbot, monkeypatch):
        """A response.incomplete event surfaces its reason so the caller can
        tell "max_output_tokens" apart from a transient empty stream."""
        monkeypatch.setenv("BUZZ_TRANSLATION_API_PROTOCOL", RESPONSES_PROTOCOL)

        options = TranscriptionOptions(
            llm_model="deepseek-chat", llm_prompt="Translate:"
        )
        translator = Translator(
            options,
            AdvancedSettingsDialog(transcription_options=options, parent=None),
        )

        lines = [
            'data: {"type":"response.created"}',
            'data: {"type":"response.reasoning_text.delta","delta":"thinking..."}',
            'data: {"type":"response.reasoning_text.done"}',
            'data: {"type":"response.incomplete",'
            '"incomplete_details":{"reason":"max_output_tokens"}}',
        ]
        resp = _stream_response(lines=lines, protocol=RESPONSES_PROTOCOL)

        text, incomplete_reason = translator._read_stream(resp)
        assert text is None
        assert incomplete_reason == "max_output_tokens"

    def test_read_stream_incomplete_with_partial_text(self, qtbot, monkeypatch):
        """Truncated output still carries the incomplete reason so batch JSON
        parse failures can be traced back to a token-budget cutoff."""
        monkeypatch.setenv("BUZZ_TRANSLATION_API_PROTOCOL", RESPONSES_PROTOCOL)

        options = TranscriptionOptions(
            llm_model="deepseek-chat", llm_prompt="Translate:"
        )
        translator = Translator(
            options,
            AdvancedSettingsDialog(transcription_options=options, parent=None),
        )

        lines = [
            'data: {"type":"response.output_text.delta","delta":"{\\"translations\\": {"}',
            'data: {"type":"response.incomplete",'
            '"incomplete_details":{"reason":"max_output_tokens"}}',
        ]
        resp = _stream_response(lines=lines, protocol=RESPONSES_PROTOCOL)

        text, incomplete_reason = translator._read_stream(resp)
        assert text == '{"translations": {'
        assert incomplete_reason == "max_output_tokens"

    @patch('buzz.translator.time.sleep')
    @patch('buzz.translator.httpx.Client')
    def test_messages_reports_max_output_tokens_diagnostic(
        self, mock_client_class, mock_sleep, qtbot, monkeypatch, caplog
    ):
        """Empty stream caused by max_output_tokens logs a distinct warning
        instead of the generic empty-stream message."""
        monkeypatch.setenv("BUZZ_TRANSLATION_API_KEY", "deepseek-key")
        monkeypatch.setenv(
            "BUZZ_TRANSLATION_API_BASE_URL", "https://api.deepseek.com/v1"
        )
        monkeypatch.setenv("BUZZ_TRANSLATION_API_PROTOCOL", RESPONSES_PROTOCOL)

        incomplete_lines = [
            'data: {"type":"response.created"}',
            'data: {"type":"response.reasoning_text.delta","delta":"thinking..."}',
            'data: {"type":"response.reasoning_text.done"}',
            'data: {"type":"response.incomplete",'
            '"incomplete_details":{"reason":"max_output_tokens"}}',
        ]
        mock_client = Mock()
        mock_client.stream.side_effect = [
            _stream_response(lines=incomplete_lines, protocol=RESPONSES_PROTOCOL),
            _stream_response("翻译", protocol=RESPONSES_PROTOCOL),
        ]
        mock_client_class.return_value = mock_client

        options = TranscriptionOptions(
            llm_model="deepseek-chat", llm_prompt="Translate:"
        )
        translator = Translator(
            options,
            AdvancedSettingsDialog(transcription_options=options, parent=None),
        )

        assert translator._messages("Translate:", "Hello") == "翻译"
        assert mock_client.stream.call_count == 2
        assert any(
            "token budget exhausted" in record.message
            and "max_output_tokens" in record.message
            for record in caplog.records
        )

    @patch('buzz.translator.httpx.Client')
    @patch('buzz.translator.queue.Queue', autospec=True)
    def test_start(self, mock_queue, mock_client_class, qtbot, monkeypatch):
        monkeypatch.setenv("BUZZ_TRANSLATION_API_PROTOCOL", CHAT_COMPLETIONS_PROTOCOL)

        def side_effect(*args, **kwargs):
            if side_effect.call_count <= 1:
                side_effect.call_count += 1
                return ("Hello, how are you?", 1)

            # Finally return sentinel to stop
            return None

        side_effect.call_count = 0

        mock_queue.get.side_effect = side_effect
        mock_client = Mock()
        mock_client.stream.return_value = _stream_response(
            "AI Translated: Hello, how are you?"
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
        mock_client.stream.assert_called()

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
        monkeypatch.setenv("BUZZ_TRANSLATION_API_PROTOCOL", CHAT_COMPLETIONS_PROTOCOL)

        self.on_next_translation_called = False

        def on_next_translation(text: str):
            self.on_next_translation_called = True
            assert text.startswith("AI Translated:")

        mock_client = Mock()
        mock_client.stream.return_value = _stream_response(
            "AI Translated: Hello, how are you?"
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
        monkeypatch.setenv("BUZZ_TRANSLATION_API_PROTOCOL", CHAT_COMPLETIONS_PROTOCOL)

        mock_client = Mock()
        mock_client.stream.side_effect = [_failing_response(500)] * 3 + [
            _stream_response("AI Translated")
        ]
        mock_client_class.return_value = mock_client
        options = TranscriptionOptions(llm_model="gpt-4o-mini", llm_prompt="Translate:")
        translator = Translator(
            options,
            AdvancedSettingsDialog(transcription_options=options, parent=None),
        )

        assert translator._messages("Translate:", "Hello") == "AI Translated"
        assert mock_client.stream.call_count == 4

    @patch('buzz.translator.time.sleep')
    @patch('buzz.translator.httpx.Client')
    def test_messages_retries_429_with_retry_after(
        self, mock_client_class, mock_sleep, qtbot, monkeypatch
    ):
        monkeypatch.setenv("BUZZ_TRANSLATION_API_KEY", "openai-key")
        monkeypatch.setenv("BUZZ_TRANSLATION_API_BASE_URL", "https://api.openai.com/v1")
        monkeypatch.setenv("BUZZ_TRANSLATION_API_PROTOCOL", CHAT_COMPLETIONS_PROTOCOL)

        mock_client = Mock()
        mock_client.stream.side_effect = [
            _failing_response(429, retry_after="2"),
            _stream_response("AI Translated"),
        ]
        mock_client_class.return_value = mock_client
        options = TranscriptionOptions(llm_model="gpt-4o-mini", llm_prompt="Translate:")
        translator = Translator(
            options,
            AdvancedSettingsDialog(transcription_options=options, parent=None),
        )

        assert translator._messages("Translate:", "Hello") == "AI Translated"
        assert mock_client.stream.call_count == 2
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
        monkeypatch.setenv("BUZZ_TRANSLATION_API_PROTOCOL", CHAT_COMPLETIONS_PROTOCOL)

        mock_client = Mock()
        mock_client.stream.side_effect = [
            httpx.ConnectError("connection refused"),
            httpx.ConnectError("connection refused"),
            _stream_response("AI Translated"),
        ]
        mock_client_class.return_value = mock_client
        options = TranscriptionOptions(llm_model="gpt-4o-mini", llm_prompt="Translate:")
        translator = Translator(
            options,
            AdvancedSettingsDialog(transcription_options=options, parent=None),
        )

        assert translator._messages("Translate:", "Hello") == "AI Translated"
        assert mock_client.stream.call_count == 3

    @patch('buzz.translator.httpx.Client')
    def test_messages_does_not_retry_auth_error(
        self, mock_client_class, qtbot, monkeypatch
    ):
        monkeypatch.setenv("BUZZ_TRANSLATION_API_KEY", "openai-key")
        monkeypatch.setenv("BUZZ_TRANSLATION_API_BASE_URL", "https://api.openai.com/v1")
        monkeypatch.setenv("BUZZ_TRANSLATION_API_PROTOCOL", CHAT_COMPLETIONS_PROTOCOL)

        mock_client = Mock()
        mock_client.stream.side_effect = [_failing_response(401)]
        mock_client_class.return_value = mock_client
        options = TranscriptionOptions(llm_model="gpt-4o-mini", llm_prompt="Translate:")
        translator = Translator(
            options,
            AdvancedSettingsDialog(transcription_options=options, parent=None),
        )

        assert translator._messages("Translate:", "Hello") is None
        assert mock_client.stream.call_count == 1

    @patch('buzz.translator.httpx.Client')
    def test_messages_does_not_retry_unexpected_errors(
        self, mock_client_class, qtbot, monkeypatch
    ):
        monkeypatch.setenv("BUZZ_TRANSLATION_API_KEY", "openai-key")
        monkeypatch.setenv("BUZZ_TRANSLATION_API_BASE_URL", "https://api.openai.com/v1")
        monkeypatch.setenv("BUZZ_TRANSLATION_API_PROTOCOL", CHAT_COMPLETIONS_PROTOCOL)

        mock_client = Mock()
        mock_client.stream.side_effect = [Exception("temporary failure")]
        mock_client_class.return_value = mock_client
        options = TranscriptionOptions(llm_model="gpt-4o-mini", llm_prompt="Translate:")
        translator = Translator(
            options,
            AdvancedSettingsDialog(transcription_options=options, parent=None),
        )

        assert translator._messages("Translate:", "Hello") is None
        assert mock_client.stream.call_count == 1

    @patch('buzz.translator.time.sleep')
    @patch('buzz.translator.httpx.Client')
    def test_messages_retries_response_not_read(
        self, mock_client_class, mock_sleep, qtbot, monkeypatch
    ):
        """A dropped stream body must be retried, not lose the whole batch.

        httpx.ResponseNotRead is a StreamError/RuntimeError, not a
        TransportError, so it used to fall through to the catch-all handler
        and return None without a single retry.
        """
        monkeypatch.setenv("BUZZ_TRANSLATION_API_KEY", "openai-key")
        monkeypatch.setenv("BUZZ_TRANSLATION_API_BASE_URL", "https://api.openai.com/v1")
        monkeypatch.setenv("BUZZ_TRANSLATION_API_PROTOCOL", CHAT_COMPLETIONS_PROTOCOL)

        dropped = MagicMock()
        dropped.__enter__ = MagicMock(return_value=dropped)
        dropped.__exit__ = MagicMock(return_value=None)
        dropped.raise_for_status = MagicMock(return_value=None)
        dropped.iter_lines = MagicMock(side_effect=httpx.ResponseNotRead())

        mock_client = Mock()
        mock_client.stream.side_effect = [
            dropped,
            _stream_response("AI Translated"),
        ]
        mock_client_class.return_value = mock_client
        options = TranscriptionOptions(llm_model="gpt-4o-mini", llm_prompt="Translate:")
        translator = Translator(
            options,
            AdvancedSettingsDialog(transcription_options=options, parent=None),
        )

        assert translator._messages("Translate:", "Hello") == "AI Translated"
        assert mock_client.stream.call_count == 2

    @patch('buzz.translator.httpx.Client')
    def test_messages_falls_back_to_buffered_json(
        self, mock_client_class, qtbot, monkeypatch
    ):
        """A gateway that ignores `stream: true` still has to work."""
        monkeypatch.setenv("BUZZ_TRANSLATION_API_KEY", "openai-key")
        monkeypatch.setenv("BUZZ_TRANSLATION_API_BASE_URL", "https://api.openai.com/v1")
        monkeypatch.setenv("BUZZ_TRANSLATION_API_PROTOCOL", CHAT_COMPLETIONS_PROTOCOL)

        mock_client = Mock()
        mock_client.stream.return_value = _stream_response(
            lines=['{"choices": [{"message": {"content": "Buffered"}}]}']
        )
        mock_client_class.return_value = mock_client
        options = TranscriptionOptions(llm_model="gpt-4o-mini", llm_prompt="Translate:")
        translator = Translator(
            options,
            AdvancedSettingsDialog(transcription_options=options, parent=None),
        )

        assert translator._messages("Translate:", "Hello") == "Buffered"
