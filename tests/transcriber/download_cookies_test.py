import logging

import pytest

from buzz.transcriber.file_transcriber import (
    apply_cookie_options,
    parse_cookies_from_browser,
)


class TestParseCookiesFromBrowser:
    @pytest.mark.parametrize(
        "spec,expected",
        [
            ("chrome", ("chrome", None, None, None)),
            # yt-dlp lowercases the browser name, so GUI/env casing is forgiving.
            ("Chrome", ("chrome", None, None, None)),
            ("safari", ("safari", None, None, None)),
            ("firefox:default", ("firefox", "default", None, None)),
            ("firefox:/tmp/prof", ("firefox", "/tmp/prof", None, None)),
            # A single colon is a profile; a double colon is a container.
            ("chrome::Default", ("chrome", None, None, "Default")),
            ("chrome+KWALLET", ("chrome", None, "KWALLET", None)),
            ("chrome+kwallet:prof::cont", ("chrome", "prof", "KWALLET", "cont")),
        ],
    )
    def test_parses_cli_spec_syntax(self, spec, expected):
        assert parse_cookies_from_browser(spec) == expected

    @pytest.mark.parametrize(
        "spec", ["", "netscape", "notabrowser:prof", "chrome+NOPE"]
    )
    def test_rejects_unusable_spec(self, spec):
        with pytest.raises(ValueError):
            parse_cookies_from_browser(spec)


class TestApplyCookieOptions:
    def test_no_env_adds_nothing(self, monkeypatch):
        monkeypatch.delenv("BUZZ_DOWNLOAD_COOKIEFILE", raising=False)
        monkeypatch.delenv("BUZZ_DOWNLOAD_COOKIES_FROM_BROWSER", raising=False)
        assert apply_cookie_options({}) == {}

    def test_cookiefile_only(self, monkeypatch):
        monkeypatch.setenv("BUZZ_DOWNLOAD_COOKIEFILE", "/tmp/c.txt")
        monkeypatch.delenv("BUZZ_DOWNLOAD_COOKIES_FROM_BROWSER", raising=False)
        assert apply_cookie_options({}) == {"cookiefile": "/tmp/c.txt"}

    def test_browser_only(self, monkeypatch):
        monkeypatch.delenv("BUZZ_DOWNLOAD_COOKIEFILE", raising=False)
        monkeypatch.setenv("BUZZ_DOWNLOAD_COOKIES_FROM_BROWSER", "chrome")
        assert apply_cookie_options({}) == {
            "cookiesfrombrowser": ("chrome", None, None, None)
        }

    def test_both_sources_can_combine(self, monkeypatch):
        monkeypatch.setenv("BUZZ_DOWNLOAD_COOKIEFILE", "/tmp/c.txt")
        monkeypatch.setenv(
            "BUZZ_DOWNLOAD_COOKIES_FROM_BROWSER", "firefox:default"
        )
        assert apply_cookie_options({}) == {
            "cookiefile": "/tmp/c.txt",
            "cookiesfrombrowser": ("firefox", "default", None, None),
        }

    def test_value_is_stripped(self, monkeypatch):
        monkeypatch.delenv("BUZZ_DOWNLOAD_COOKIEFILE", raising=False)
        monkeypatch.setenv("BUZZ_DOWNLOAD_COOKIES_FROM_BROWSER", "  chrome  ")
        assert apply_cookie_options({}) == {
            "cookiesfrombrowser": ("chrome", None, None, None)
        }

    def test_preserves_existing_options(self, monkeypatch):
        monkeypatch.delenv("BUZZ_DOWNLOAD_COOKIEFILE", raising=False)
        monkeypatch.setenv("BUZZ_DOWNLOAD_COOKIES_FROM_BROWSER", "chrome")
        options = {"format": "bestaudio/best"}
        returned = apply_cookie_options(options)
        # Mutates in place and returns the same dict, so both call styles work.
        assert returned is options
        assert options["format"] == "bestaudio/best"
        assert options["cookiesfrombrowser"] == ("chrome", None, None, None)

    def test_bad_value_warns_instead_of_raising(self, monkeypatch, caplog):
        monkeypatch.delenv("BUZZ_DOWNLOAD_COOKIEFILE", raising=False)
        monkeypatch.setenv("BUZZ_DOWNLOAD_COOKIES_FROM_BROWSER", "nosuchbrowser")
        with caplog.at_level(logging.WARNING):
            options = apply_cookie_options({})
        # A typo must not fail the task; anonymous formats may still suffice.
        assert options == {}
        assert "BUZZ_DOWNLOAD_COOKIES_FROM_BROWSER" in caplog.text
