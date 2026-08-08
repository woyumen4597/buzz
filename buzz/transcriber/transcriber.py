import copy
import datetime
import enum
import hashlib
import json
import os
import uuid
from dataclasses import dataclass, field
from random import randint
from typing import List, Optional, Tuple, Set

from dataclasses_json import dataclass_json, config, Exclude

from buzz.locale import _
from buzz.model_loader import TranscriptionModel
from buzz.settings.settings import Settings

DEFAULT_WHISPER_TEMPERATURE = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
TASK_OPTIONS_VERSION = 1
SEGMENT_CHECKPOINT_VERSION = 1
_FINGERPRINT_SAMPLE_BYTES = 64 * 1024


class Task(enum.Enum):
    TRANSLATE = "translate"
    TRANSCRIBE = "transcribe"


TASK_LABEL_TRANSLATIONS = {
    Task.TRANSLATE: _("Translate to English"),
    Task.TRANSCRIBE: _("Transcribe"),
}


@dataclass
class Segment:
    start: int  # start time in ms
    end: int  # end time in ms
    text: str
    translation: str = ""


LANGUAGES = {
    "en": _("English"),
    "zh": _("Chinese"),
    "de": _("German"),
    "es": _("Spanish"),
    "ru": _("Russian"),
    "ko": _("Korean"),
    "fr": _("French"),
    "ja": _("Japanese"),
    "pt": _("Portuguese"),
    "tr": _("Turkish"),
    "pl": _("Polish"),
    "ca": _("Catalan"),
    "nl": _("Dutch"),
    "ar": _("Arabic"),
    "sv": _("Swedish"),
    "it": _("Italian"),
    "id": _("Indonesian"),
    "hi": _("Hindi"),
    "fi": _("Finnish"),
    "vi": _("Vietnamese"),
    "he": _("Hebrew"),
    "uk": _("Ukrainian"),
    "el": _("Greek"),
    "ms": _("Malay"),
    "cs": _("Czech"),
    "ro": _("Romanian"),
    "da": _("Danish"),
    "hu": _("Hungarian"),
    "ta": _("Tamil"),
    "no": _("Norwegian"),
    "th": _("Thai"),
    "ur": _("Urdu"),
    "hr": _("Croatian"),
    "bg": _("Bulgarian"),
    "lt": _("Lithuanian"),
    "la": _("Latin"),
    "mi": _("Maori"),
    "ml": _("Malayalam"),
    "cy": _("Welsh"),
    "sk": _("Slovak"),
    "te": _("Telugu"),
    "fa": _("Persian"),
    "lv": _("Latvian"),
    "bn": _("Bengali"),
    "sr": _("Serbian"),
    "az": _("Azerbaijani"),
    "sl": _("Slovenian"),
    "kn": _("Kannada"),
    "et": _("Estonian"),
    "mk": _("Macedonian"),
    "br": _("Breton"),
    "eu": _("Basque"),
    "is": _("Icelandic"),
    "hy": _("Armenian"),
    "ne": _("Nepali"),
    "mn": _("Mongolian"),
    "bs": _("Bosnian"),
    "kk": _("Kazakh"),
    "sq": _("Albanian"),
    "sw": _("Swahili"),
    "gl": _("Galician"),
    "mr": _("Marathi"),
    "pa": _("Punjabi"),
    "si": _("Sinhala"),
    "km": _("Khmer"),
    "sn": _("Shona"),
    "yo": _("Yoruba"),
    "so": _("Somali"),
    "af": _("Afrikaans"),
    "oc": _("Occitan"),
    "ka": _("Georgian"),
    "be": _("Belarusian"),
    "tg": _("Tajik"),
    "sd": _("Sindhi"),
    "gu": _("Gujarati"),
    "am": _("Amharic"),
    "yi": _("Yiddish"),
    "lo": _("Lao"),
    "uz": _("Uzbek"),
    "fo": _("Faroese"),
    "ht": _("Haitian Creole"),
    "ps": _("Pashto"),
    "tk": _("Turkmen"),
    "nn": _("Nynorsk"),
    "mt": _("Maltese"),
    "sa": _("Sanskrit"),
    "lb": _("Luxembourgish"),
    "my": _("Myanmar"),
    "bo": _("Tibetan"),
    "tl": _("Tagalog"),
    "mg": _("Malagasy"),
    "as": _("Assamese"),
    "tt": _("Tatar"),
    "haw": _("Hawaiian"),
    "ln": _("Lingala"),
    "ha": _("Hausa"),
    "ba": _("Bashkir"),
    "jw": _("Javanese"),
    "su": _("Sundanese"),
    "yue": _("Cantonese"),
}


@dataclass()
class TranscriptionOptions:
    language: Optional[str] = None
    task: Task = Task.TRANSCRIBE
    model: TranscriptionModel = field(default_factory=TranscriptionModel)
    word_level_timings: bool = False
    extract_speech: bool = False
    temperature: Tuple[float, ...] = DEFAULT_WHISPER_TEMPERATURE
    initial_prompt: str = ""
    openai_access_token: str = field(
        default="", metadata=config(exclude=Exclude.ALWAYS)
    )
    enable_llm_translation: bool = False
    llm_prompt: str = ""
    llm_model: str = ""
    silence_threshold: float = 0.0025
    line_separator: str = "\n\n"
    transcription_step: float = 3.5
    use_vad: bool = False


def humanize_language(language: str) -> str:
    if language == "":
        return _("Detect Language")
    return LANGUAGES[language].title()


@dataclass()
class FileTranscriptionOptions:
    file_paths: Optional[List[str]] = None
    url: Optional[str] = None
    output_formats: Set["OutputFormat"] = field(default_factory=set)
    translate: bool = False


@dataclass_json
@dataclass
class FileTranscriptionTask:
    class Status(enum.Enum):
        QUEUED = "queued"
        IN_PROGRESS = "in_progress"
        COMPLETED = "completed"
        FAILED = "failed"
        CANCELED = "canceled"
        SKIPPED = "skipped"

    class Source(enum.Enum):
        FILE_IMPORT = "file_import"
        URL_IMPORT = "url_import"
        FOLDER_WATCH = "folder_watch"

    transcription_options: TranscriptionOptions
    file_transcription_options: FileTranscriptionOptions
    model_path: str
    # deprecated: use uid
    id: int = field(default_factory=lambda: randint(0, 100_000_000))
    uid: uuid.UUID = field(default_factory=uuid.uuid4)
    segments: List[Segment] = field(default_factory=list)
    status: Optional[Status] = None
    fraction_completed: float = 0.0
    error: Optional[str] = None
    queued_at: Optional[datetime.datetime] = None
    started_at: Optional[datetime.datetime] = None
    completed_at: Optional[datetime.datetime] = None
    output_directory: Optional[str] = None
    source: Source = Source.FILE_IMPORT
    file_path: Optional[str] = None
    original_file_path: Optional[str] = None  # Original path before speech extraction
    delete_source_file: bool = False
    url: Optional[str] = None
    fraction_downloaded: float = 0.0
    # Number of fully persisted API chunks.  It lets a recovered OpenAI API
    # task continue after the last durable segment checkpoint.
    checkpoint_next_chunk: int = 0

    def __post_init__(self):
        # Ensure shared UI settings do not affect queued task
        self.transcription_options = copy.deepcopy(self.transcription_options)


class OutputFormat(enum.Enum):
    TXT = "txt"
    SRT = "srt"
    VTT = "vtt"


def serialize_task_options(task: FileTranscriptionTask) -> str:
    """Serialize the replayable task configuration without credentials."""
    options = task.transcription_options
    file_options = task.file_transcription_options
    return json.dumps(
        {
            "version": TASK_OPTIONS_VERSION,
            "transcription_options": {
                "language": options.language,
                "task": options.task.value,
                "model": {
                    "model_type": options.model.model_type.value,
                    "whisper_model_size": (
                        options.model.whisper_model_size.value
                        if options.model.whisper_model_size
                        else None
                    ),
                    "hugging_face_model_id": options.model.hugging_face_model_id,
                },
                "word_level_timings": options.word_level_timings,
                "extract_speech": options.extract_speech,
                "use_vad": options.use_vad,
                "temperature": list(options.temperature),
                "initial_prompt": options.initial_prompt,
                "enable_llm_translation": options.enable_llm_translation,
                "llm_prompt": options.llm_prompt,
                "llm_model": options.llm_model,
                "silence_threshold": options.silence_threshold,
                "line_separator": options.line_separator,
                "transcription_step": options.transcription_step,
            },
            "file_transcription_options": {
                "file_paths": file_options.file_paths,
                "url": file_options.url,
                "output_formats": sorted(
                    output_format.value for output_format in file_options.output_formats
                ),
                "translate": file_options.translate,
            },
            "task": {
                "model_path": task.model_path,
                "source": task.source.value,
                "file_path": task.file_path,
                "original_file_path": task.original_file_path,
                "delete_source_file": task.delete_source_file,
                "url": task.url,
                "output_directory": task.output_directory,
            },
        },
        separators=(",", ":"),
        sort_keys=True,
    )


def deserialize_task_options(
    payload: str | None,
    fallback=None,
    openai_access_token: str = "",
) -> tuple[TranscriptionOptions, FileTranscriptionOptions]:
    """Load current task options, falling back to the legacy DB columns.

    Task option JSON deliberately never contains an API key.  Recovery reads
    the current keyring value through its caller instead.
    """
    from buzz.model_loader import ModelType, TranscriptionModel, WhisperModelSize

    def value(name, default=None):
        return getattr(fallback, name, default) if fallback is not None else default

    try:
        data = json.loads(payload) if payload else {}
    except (TypeError, json.JSONDecodeError):
        data = {}
    if not isinstance(data, dict) or data.get("version") != TASK_OPTIONS_VERSION:
        data = {}

    options_data = data.get("transcription_options", {})
    if not isinstance(options_data, dict):
        options_data = {}
    model_data = options_data.get("model", {})
    if not isinstance(model_data, dict):
        model_data = {}
    file_data = data.get("file_transcription_options", {})
    if not isinstance(file_data, dict):
        file_data = {}

    def as_bool(raw, default=False):
        if raw is None:
            return default
        if isinstance(raw, bool):
            return raw
        return str(raw).lower() in {"1", "true", "yes"}

    def enum_or_default(enum_type, raw, default):
        if isinstance(raw, enum_type):
            return raw
        try:
            return enum_type(raw)
        except (TypeError, ValueError):
            return default

    model_type = enum_or_default(
        ModelType,
        model_data.get("model_type", value("model_type")),
        ModelType.WHISPER,
    )
    raw_whisper_model_size = model_data.get(
        "whisper_model_size", value("whisper_model_size")
    )
    whisper_model_size = (
        enum_or_default(WhisperModelSize, raw_whisper_model_size, WhisperModelSize.TINY)
        if raw_whisper_model_size
        else (
            None
            if model_type in (ModelType.HUGGING_FACE, ModelType.OPEN_AI_WHISPER_API)
            else WhisperModelSize.TINY
        )
    )
    task = enum_or_default(
        Task, options_data.get("task", value("task")), Task.TRANSCRIBE
    )

    temperature = options_data.get("temperature", DEFAULT_WHISPER_TEMPERATURE)
    try:
        temperature = tuple(float(item) for item in temperature)
    except (TypeError, ValueError):
        temperature = DEFAULT_WHISPER_TEMPERATURE

    transcription_options = TranscriptionOptions(
        language=options_data.get("language", value("language")),
        task=task,
        model=TranscriptionModel(
            model_type=model_type,
            whisper_model_size=whisper_model_size,
            hugging_face_model_id=model_data.get(
                "hugging_face_model_id", value("hugging_face_model_id")
            ),
        ),
        word_level_timings=as_bool(
            options_data.get("word_level_timings", value("word_level_timings"))
        ),
        extract_speech=as_bool(
            options_data.get("extract_speech", value("extract_speech"))
        ),
        use_vad=as_bool(options_data.get("use_vad")),
        temperature=temperature,
        initial_prompt=options_data.get("initial_prompt", ""),
        openai_access_token=openai_access_token,
        enable_llm_translation=as_bool(
            options_data.get("enable_llm_translation"), False
        ),
        llm_prompt=options_data.get("llm_prompt", ""),
        llm_model=options_data.get("llm_model", ""),
        silence_threshold=options_data.get("silence_threshold", 0.0025),
        line_separator=options_data.get("line_separator", "\n\n"),
        transcription_step=options_data.get("transcription_step", 3.5),
    )

    output_formats = set()
    raw_formats = file_data.get("output_formats")
    if raw_formats is None:
        raw_formats = str(value("export_formats") or "").split(",")
    for raw_format in raw_formats:
        try:
            output_formats.add(OutputFormat(str(raw_format).strip()))
        except ValueError:
            continue

    return transcription_options, FileTranscriptionOptions(
        file_paths=file_data.get("file_paths"),
        url=file_data.get("url", value("url")),
        output_formats=output_formats,
        translate=as_bool(file_data.get("translate"), False),
    )


def deserialize_task_metadata(payload: str | None) -> dict:
    try:
        data = json.loads(payload) if payload else {}
    except (TypeError, json.JSONDecodeError):
        return {}
    if data.get("version") != TASK_OPTIONS_VERSION:
        return {}
    metadata = data.get("task", {})
    return metadata if isinstance(metadata, dict) else {}


def serialize_segment_checkpoint(task: FileTranscriptionTask) -> str:
    return json.dumps(
        {
            "version": SEGMENT_CHECKPOINT_VERSION,
            "next_chunk": task.checkpoint_next_chunk,
            "segments": [
                {
                    "start": segment.start,
                    "end": segment.end,
                    "text": segment.text,
                    "translation": segment.translation,
                }
                for segment in task.segments
            ],
        },
        separators=(",", ":"),
    )


def deserialize_segment_checkpoint(payload: str | None) -> tuple[List[Segment], int]:
    try:
        data = json.loads(payload) if payload else {}
    except (TypeError, json.JSONDecodeError):
        return [], 0
    if not isinstance(data, dict) or data.get("version") != SEGMENT_CHECKPOINT_VERSION:
        return [], 0

    segments = []
    for item in data.get("segments", []):
        try:
            segments.append(
                Segment(
                    start=int(item["start"]),
                    end=int(item["end"]),
                    text=str(item["text"]),
                    translation=str(item.get("translation", "")),
                )
            )
        except (KeyError, TypeError, ValueError):
            continue
    try:
        next_chunk = max(0, int(data.get("next_chunk", 0)))
    except (TypeError, ValueError):
        next_chunk = 0
    return segments, next_chunk


def source_file_fingerprint(path: str | None) -> str | None:
    """Return a bounded, versioned fingerprint for validating recovery input."""
    if not path or not os.path.isfile(path):
        return None
    stat = os.stat(path)
    digest = hashlib.sha256()
    digest.update(f"{stat.st_size}:{stat.st_mtime_ns}".encode())
    with open(path, "rb") as source:
        digest.update(source.read(_FINGERPRINT_SAMPLE_BYTES))
        if stat.st_size > _FINGERPRINT_SAMPLE_BYTES:
            source.seek(max(0, stat.st_size - _FINGERPRINT_SAMPLE_BYTES))
            digest.update(source.read(_FINGERPRINT_SAMPLE_BYTES))
    return f"v1:{stat.st_size}:{stat.st_mtime_ns}:{digest.hexdigest()}"


def source_file_matches_fingerprint(path: str | None, fingerprint: str | None) -> bool:
    return not fingerprint or source_file_fingerprint(path) == fingerprint


class Stopped(Exception):
    pass


SUPPORTED_AUDIO_FORMATS = "Media files (*.mp3 *.wav *.m4a *.ogg *.opus *.flac *.mp4 *.webm *.ogm *.mov *.mkv *.avi *.wmv);;\
Audio files (*.mp3 *.wav *.m4a *.ogg *.opus *.flac);;\
Video files (*.mp4 *.webm *.ogm *.mov *.mkv *.avi *.wmv);;\
All files (*.*)"


def get_output_file_path(
    file_path: str,
    task: Task,
    language: Optional[str],
    model: TranscriptionModel,
    output_format: OutputFormat,
    output_directory: str | None = None,
    export_file_name_template: str | None = None,
    variant: str = "",
):
    input_file_name = os.path.splitext(os.path.basename(file_path))[0]
    # Remove "_speech" suffix from extracted speech files
    if input_file_name.endswith("_speech"):
        input_file_name = input_file_name[:-7]
    date_time_now = datetime.datetime.now().strftime("%d-%b-%Y %H-%M-%S")

    export_file_name_template = (
        export_file_name_template
        if export_file_name_template is not None
        else Settings().get_default_export_file_template()
    )

    output_file_name = (
        export_file_name_template.replace("{{ input_file_name }}", input_file_name)
        .replace("{{ task }}", task.value)
        .replace("{{ language }}", language or "")
        .replace("{{ model_type }}", model.model_type.value)
        .replace(
            "{{ model_size }}",
            model.whisper_model_size.value
            if model.whisper_model_size is not None
            else "",
        )
        .replace("{{ date_time }}", date_time_now)
        + (f".{variant}" if variant else "")
        + f".{output_format.value}"
    )

    output_directory = output_directory or os.path.dirname(file_path)
    return os.path.join(output_directory, output_file_name)
