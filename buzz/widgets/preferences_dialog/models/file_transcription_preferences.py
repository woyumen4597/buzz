import copy
from dataclasses import dataclass
from typing import Optional, Tuple, Set, List

from PyQt6.QtCore import QSettings

from buzz.model_loader import TranscriptionModel
from buzz.transcriber.transcriber import (
    DEFAULT_USE_VAD,
    Task,
    OutputFormat,
    TranscriptionOptions,
    FileTranscriptionOptions,
)


#: Bumped when a stored default has to change for installations that already
#: have a settings file. Version 2 turned voice activity detection on.
SETTINGS_VERSION = 2


@dataclass()
class FileTranscriptionPreferences:
    language: Optional[str]
    task: Task
    model: TranscriptionModel
    word_level_timings: bool
    extract_speech: bool
    initial_prompt: str
    enable_llm_translation: bool
    llm_prompt: str
    llm_model: str
    output_formats: Set["OutputFormat"]
    #: Off by default (see DEFAULT_USE_VAD): silero reads quiet or breathy
    #: speech as silence, so VAD only runs when the user asks for it.
    use_vad: bool = DEFAULT_USE_VAD
    #: True only while the value above came from the user touching the checkbox.
    #: The form saves on close and on run, so a settings file written without an
    #: explicit choice holds the default of that moment; keeping it would pin
    #: that default forever, which is how a switched default used to get stuck.
    use_vad_explicit: bool = False

    def save(self, settings: QSettings) -> None:
        settings.setValue("settings_version", SETTINGS_VERSION)
        settings.setValue("language", self.language)
        settings.setValue("task", self.task)
        settings.setValue("model", self.model)
        settings.setValue("word_level_timings", self.word_level_timings)
        settings.setValue("extract_speech", self.extract_speech)
        settings.setValue("use_vad", self.use_vad)
        settings.setValue("use_vad_explicit", self.use_vad_explicit)
        settings.setValue("initial_prompt", self.initial_prompt)
        settings.setValue("enable_llm_translation", self.enable_llm_translation)
        settings.setValue("llm_model", self.llm_model)
        settings.setValue("llm_prompt", self.llm_prompt)
        settings.setValue(
            "output_formats",
            [output_format.value for output_format in self.output_formats],
        )

    @classmethod
    def load(cls, settings: QSettings) -> "FileTranscriptionPreferences":
        language = settings.value("language", None)
        task = settings.value("task", Task.TRANSCRIBE)
        model: TranscriptionModel = copy.deepcopy(
            settings.value("model", TranscriptionModel.default())
        )

        word_level_timings_value = settings.value("word_level_timings", False)
        word_level_timings = False if word_level_timings_value == "false" \
            else bool(word_level_timings_value)

        extract_speech_value = settings.value("extract_speech", False)
        extract_speech = False if extract_speech_value == "false" \
            else bool(extract_speech_value)

        use_vad_value = settings.value("use_vad", DEFAULT_USE_VAD)
        use_vad_stored = False if use_vad_value == "false" else bool(use_vad_value)
        use_vad_explicit_value = settings.value("use_vad_explicit", False)
        use_vad_explicit = (
            False
            if use_vad_explicit_value == "false"
            else bool(use_vad_explicit_value)
        )
        # Only what the user chose is kept. Anything else follows the default of
        # the running version, so a settings file that merely recorded an old
        # default -- and has since been saved again -- picks the new one up.
        # Version 2 turned VAD on for installations that had never chosen; that
        # cannot be told apart from a choice except through use_vad_explicit, so
        # a stored on without it is one of those and goes back to off.
        if not use_vad_explicit:
            use_vad = DEFAULT_USE_VAD
        else:
            use_vad = use_vad_stored

        initial_prompt = settings.value("initial_prompt", "")
        enable_llm_translation_value = settings.value("enable_llm_translation", True)
        enable_llm_translation = False if enable_llm_translation_value == "false" \
            else bool(enable_llm_translation_value)
        llm_model = settings.value("llm_model", "")
        llm_prompt = settings.value("llm_prompt", "")
        output_formats = settings.value("output_formats", []) or []
        return FileTranscriptionPreferences(
            language=language,
            task=task,
            model=model
            if model.model_type.is_available()
            else TranscriptionModel.default(),
            word_level_timings=word_level_timings,
            extract_speech=extract_speech,
            use_vad=use_vad,
            use_vad_explicit=use_vad_explicit,
            initial_prompt=initial_prompt,
            enable_llm_translation=enable_llm_translation,
            llm_model=llm_model,
            llm_prompt=llm_prompt,
            output_formats=set(
                [OutputFormat(output_format) for output_format in output_formats]
            ),
        )

    @classmethod
    def from_transcription_options(
        cls,
        transcription_options: TranscriptionOptions,
        file_transcription_options: FileTranscriptionOptions,
        use_vad_explicit: bool = False,
    ) -> "FileTranscriptionPreferences":
        return FileTranscriptionPreferences(
            task=transcription_options.task,
            language=transcription_options.language,
            initial_prompt=transcription_options.initial_prompt,
            enable_llm_translation=transcription_options.enable_llm_translation,
            llm_model=transcription_options.llm_model,
            llm_prompt=transcription_options.llm_prompt,
            word_level_timings=transcription_options.word_level_timings,
            extract_speech=transcription_options.extract_speech,
            use_vad=transcription_options.use_vad,
            use_vad_explicit=use_vad_explicit,
            model=transcription_options.model,
            output_formats=file_transcription_options.output_formats,
        )

    def to_transcription_options(
        self,
        openai_access_token: Optional[str],
        file_paths: Optional[List[str]] = None,
        url: Optional[str] = None,
    ) -> Tuple[TranscriptionOptions, FileTranscriptionOptions]:
        return (
            TranscriptionOptions(
                task=self.task,
                language=self.language,
                initial_prompt=self.initial_prompt,
                enable_llm_translation=self.enable_llm_translation,
                llm_model=self.llm_model,
                llm_prompt=self.llm_prompt,
                word_level_timings=self.word_level_timings,
                extract_speech=self.extract_speech,
                use_vad=self.use_vad,
                model=self.model,
                openai_access_token=openai_access_token,
            ),
            FileTranscriptionOptions(
                output_formats=self.output_formats,
                file_paths=file_paths,
                url=url,
            ),
        )
