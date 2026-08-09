"""Cookie configuration for yt-dlp URL downloads.

Sites like bilibili serve a reduced set of formats -- or only a short preview --
to anonymous clients, so a download can appear to succeed while yielding the
wrong content. Two independent sources can supply the login, and both may be
combined:

* a Netscape-format cookie file on disk (yt-dlp's ``--cookies``)
* a live browser profile (yt-dlp's ``--cookies-from-browser``)

Resolution order for each source, evaluated fresh on every download:

1. the saved preference, if the user set one in the preferences dialog
2. the matching environment variable, which keeps the documented
   BUZZ_DOWNLOAD_COOKIEFILE behaviour working for CLI, headless and
   ``.desktop``/Flatpak launches that have no dialog
3. nothing -- download anonymously

A visible control wins over an environment variable so that the dialog never
lies about what a download will do.

yt_dlp is imported lazily so that merely importing this module (the preferences
widget does, to populate its browser dropdown) does not pull in the extractor
machinery.
"""

import logging
import os
import re
from typing import List, Optional, Tuple

COOKIEFILE_ENV_VAR = "BUZZ_DOWNLOAD_COOKIEFILE"
COOKIES_FROM_BROWSER_ENV_VAR = "BUZZ_DOWNLOAD_COOKIES_FROM_BROWSER"

# yt-dlp's CLI accepts --cookies-from-browser as a spec string, but the Python
# API wants an already-parsed 4-tuple. Mirror the CLI's own regex (see
# yt_dlp/__init__.py) so our setting takes the exact syntax users already know
# from the command line.
_SPEC_RE = re.compile(
    r"""(?x)
    (?P<name>[^+:]+)
    (?:\s*\+\s*(?P<keyring>[^:]+))?
    (?:\s*:\s*(?!:)(?P<profile>.+?))?
    (?:\s*::\s*(?P<container>.+))?
    """
)

CookiesFromBrowser = Tuple[str, Optional[str], Optional[str], Optional[str]]


def supported_browsers() -> List[str]:
    """Browser names yt-dlp can read cookies from, sorted for display."""
    from yt_dlp.cookies import SUPPORTED_BROWSERS

    return sorted(SUPPORTED_BROWSERS)


def parse_cookies_from_browser(spec: str) -> CookiesFromBrowser:
    """Parse a BROWSER[+KEYRING][:PROFILE][::CONTAINER] string into the tuple
    yt-dlp's ``cookiesfrombrowser`` option expects. Raises ValueError if the
    spec names an unknown browser or keyring."""
    from yt_dlp.cookies import SUPPORTED_BROWSERS, SUPPORTED_KEYRINGS

    match = _SPEC_RE.fullmatch(spec)
    if match is None:
        raise ValueError(f'invalid cookies-from-browser value: "{spec}"')

    browser_name, keyring, profile, container = match.group(
        "name", "keyring", "profile", "container"
    )

    browser_name = browser_name.lower()
    if browser_name not in SUPPORTED_BROWSERS:
        raise ValueError(
            f'unsupported browser for cookies: "{browser_name}". '
            f'Supported browsers are: {", ".join(sorted(SUPPORTED_BROWSERS))}'
        )

    if keyring is not None:
        keyring = keyring.upper()
        if keyring not in SUPPORTED_KEYRINGS:
            raise ValueError(
                f'unsupported keyring for cookies: "{keyring}". '
                f'Supported keyrings are: {", ".join(sorted(SUPPORTED_KEYRINGS))}'
            )

    return browser_name, profile, keyring, container


def _clean(value) -> str:
    """Normalise a QSettings/env value to a stripped string.

    QSettings round-trips a stored ``None`` as the *string* ``"None"``, which
    would otherwise be handed to yt-dlp as a literal browser or profile name.
    """
    if value is None:
        return ""
    text = str(value).strip()
    return "" if text == "None" else text


def _resolve(settings, key, env_var: str) -> str:
    """Saved preference first, then the environment variable, else empty."""
    if settings is not None:
        try:
            saved = _clean(settings.value(key=key, default_value=""))
        except Exception:
            # A missing/unreadable settings backend must not break downloading.
            logging.debug("Could not read cookie setting %s", key, exc_info=True)
            saved = ""
        if saved:
            return saved

    return _clean(os.getenv(env_var))


# Distinguishes "caller said nothing, load the saved preferences" from an
# explicit settings=None, which means "environment variables only".
_USE_SAVED_SETTINGS = object()


def apply_cookie_options(options: dict, settings=_USE_SAVED_SETTINGS) -> dict:
    """Add whichever cookie source the user configured to a yt-dlp options dict.

    By default the saved preferences are loaded. Pass ``settings=None`` to
    consult only the environment variables, which is what tests want so they do
    not depend on the developer's own preference file.
    """
    from buzz.settings.settings import Settings

    if settings is _USE_SAVED_SETTINGS:
        try:
            settings = Settings()
        except Exception:
            # No QSettings backend (bare CLI, odd sandbox) -- fall back to env.
            logging.debug("Could not open settings for cookies", exc_info=True)
            settings = None

    cookiefile = _resolve(
        settings, Settings.Key.DOWNLOAD_COOKIEFILE, COOKIEFILE_ENV_VAR
    )
    if cookiefile:
        options["cookiefile"] = cookiefile

    spec = _resolve(
        settings,
        Settings.Key.DOWNLOAD_COOKIES_FROM_BROWSER,
        COOKIES_FROM_BROWSER_ENV_VAR,
    )
    if spec:
        try:
            options["cookiesfrombrowser"] = parse_cookies_from_browser(spec)
        except ValueError as exc:
            # A malformed value should not fail the task outright; anonymous
            # formats may still be enough for what the user asked to transcribe.
            logging.warning("Ignoring cookies-from-browser value: %s", exc)

    return options
