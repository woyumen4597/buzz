import pytest
from unittest.mock import Mock, patch
from uuid import UUID, uuid4
from PyQt6.QtSql import QSqlDatabase, QSqlQuery

from buzz.db.dao.transcription_dao import TranscriptionDAO
from buzz.db.entity.transcription import Transcription
from buzz.model_loader import ModelType, TranscriptionModel, WhisperModelSize
from buzz.transcriber.transcriber import (
    deserialize_segment_checkpoint,
    deserialize_task_options,
    FileTranscriptionOptions,
    FileTranscriptionTask,
    OutputFormat,
    Segment,
    source_file_matches_fingerprint,
    Task,
    TranscriptionOptions,
)


@pytest.fixture
def db(qtbot):
    """Create an in-memory SQLite database for testing"""
    db = QSqlDatabase.addDatabase("QSQLITE")
    db.setDatabaseName(":memory:")
    assert db.open()
    
    # Create the transcription table with the new schema
    query = QSqlQuery(db)
    query.exec("""
        CREATE TABLE transcription (
            id TEXT PRIMARY KEY,
            error_message TEXT,
            export_formats TEXT,
            file TEXT,
            output_folder TEXT,
            progress DOUBLE PRECISION DEFAULT 0.0,
            language TEXT,
            model_type TEXT,
            source TEXT,
            status TEXT,
            task TEXT,
            time_ended TIMESTAMP,
            time_queued TIMESTAMP NOT NULL,
            time_started TIMESTAMP,
            url TEXT,
            whisper_model_size TEXT,
            hugging_face_model_id TEXT,
            word_level_timings BOOLEAN DEFAULT FALSE,
            extract_speech BOOLEAN DEFAULT FALSE,
            name TEXT,
            notes TEXT,
            task_options_json TEXT,
            source_file_fingerprint TEXT,
            download_progress DOUBLE PRECISION DEFAULT 0.0,
            segment_checkpoint_json TEXT
        )
    """)
    
    yield db
    db.close()


@pytest.fixture
def transcription_dao(db):
    """Create a TranscriptionDAO instance for testing"""
    return TranscriptionDAO(db)


@pytest.fixture
def sample_transcription():
    """Create a sample transcription for testing"""
    return Transcription(
        id=str(uuid4()),
        file="/path/to/test.mp3",
        status="completed",
        time_queued="2023-01-01T00:00:00",
        task="TRANSCRIBE",
        model_type="WHISPER",
        name="Test Transcription",
        notes="This is a test transcription"
    )


class TestTranscriptionDAO:
    def test_create_transcription_persists_recovery_state(
        self, transcription_dao, tmp_path
    ):
        source = tmp_path / "source.mp3"
        source.write_bytes(b"source audio")
        task = FileTranscriptionTask(
            transcription_options=TranscriptionOptions(
                language="zh",
                task=Task.TRANSLATE,
                model=TranscriptionModel(
                    model_type=ModelType.FASTER_WHISPER,
                    whisper_model_size=WhisperModelSize.SMALL,
                    hugging_face_model_id="repo/model",
                ),
                temperature=(0.1, 0.5),
                initial_prompt="keep names",
                openai_access_token="must-not-persist",
                enable_llm_translation=True,
                llm_prompt="translate faithfully",
                llm_model="gpt-test",
            ),
            file_transcription_options=FileTranscriptionOptions(
                file_paths=[str(source)],
                output_formats={OutputFormat.SRT},
                translate=True,
            ),
            model_path="/models/small",
            file_path=str(source),
            fraction_downloaded=0.25,
        )

        transcription_dao.create_transcription(task)
        transcription = transcription_dao.find_by_id(str(task.uid))

        assert transcription is not None
        assert [record.id for record in transcription_dao.get_unfinished_transcriptions()] == [
            str(task.uid)
        ]
        assert "must-not-persist" not in transcription.task_options_json
        restored_options, restored_file_options = deserialize_task_options(
            transcription.task_options_json
        )
        assert restored_options.temperature == (0.1, 0.5)
        assert restored_options.initial_prompt == "keep names"
        assert restored_options.llm_prompt == "translate faithfully"
        assert restored_options.llm_model == "gpt-test"
        assert restored_options.model.model_type == ModelType.FASTER_WHISPER
        assert restored_options.model.whisper_model_size == WhisperModelSize.SMALL
        assert restored_file_options.output_formats == {OutputFormat.SRT}
        assert source_file_matches_fingerprint(
            str(source), transcription.source_file_fingerprint
        )
        assert transcription.download_progress == 0.25
        assert not transcription.segment_checkpoint_json

        segments = [Segment(start=0, end=100, text="one")]
        transcription_dao.update_transcription_download_progress(task.uid, 0.75)
        task.segments = segments
        task.checkpoint_next_chunk = 1
        transcription_dao.update_transcription_segment_checkpoint(task.uid, task)
        transcription = transcription_dao.find_by_id(str(task.uid))
        assert transcription.download_progress == 0.75
        assert deserialize_segment_checkpoint(transcription.segment_checkpoint_json) == (
            segments,
            1,
        )

    def test_deserialize_task_options_accepts_old_rows(self):
        options, file_options = deserialize_task_options(None)
        assert options.initial_prompt == ""
        assert file_options.output_formats == set()

    def test_insert_transcription_with_name_and_notes(self, transcription_dao, sample_transcription):
        """Test inserting a transcription with name and notes fields"""
        transcription_dao.insert(sample_transcription)
        
        # Verify the transcription was inserted
        query = QSqlQuery(transcription_dao.db)
        query.prepare("SELECT * FROM transcription WHERE id = :id")
        query.bindValue(":id", sample_transcription.id)
        assert query.exec()
        assert query.next()
        
        # Check that name and notes were stored
        assert query.value("name") == "Test Transcription"
        assert query.value("notes") == "This is a test transcription"

    def test_update_transcription_name(self, transcription_dao, sample_transcription):
        """Test updating transcription name"""
        # Insert the transcription first
        transcription_dao.insert(sample_transcription)
        
        # Update the name
        new_name = "Updated Transcription Name"
        transcription_dao.update_transcription_name(UUID(sample_transcription.id), new_name)
        
        # Verify the name was updated
        query = QSqlQuery(transcription_dao.db)
        query.prepare("SELECT name FROM transcription WHERE id = :id")
        query.bindValue(":id", sample_transcription.id)
        assert query.exec()
        assert query.next()
        assert query.value("name") == new_name

    def test_update_transcription_notes(self, transcription_dao, sample_transcription):
        """Test updating transcription notes"""
        # Insert the transcription first
        transcription_dao.insert(sample_transcription)
        
        # Update the notes
        new_notes = "Updated transcription notes with more details"
        transcription_dao.update_transcription_notes(UUID(sample_transcription.id), new_notes)
        
        # Verify the notes were updated
        query = QSqlQuery(transcription_dao.db)
        query.prepare("SELECT notes FROM transcription WHERE id = :id")
        query.bindValue(":id", sample_transcription.id)
        assert query.exec()
        assert query.next()
        assert query.value("notes") == new_notes

    def test_update_transcription_language(self, transcription_dao, sample_transcription):
        """Test updating transcription language"""
        transcription_dao.insert(sample_transcription)

        transcription_dao.update_transcription_language(
            UUID(sample_transcription.id), "lv"
        )

        query = QSqlQuery(transcription_dao.db)
        query.prepare("SELECT language FROM transcription WHERE id = :id")
        query.bindValue(":id", sample_transcription.id)
        assert query.exec()
        assert query.next()
        assert query.value("language") == "lv"

    def test_update_transcription_language_nonexistent_id(self, transcription_dao):
        """Test updating language for non-existent transcription ID"""
        with pytest.raises(Exception):
            transcription_dao.update_transcription_language(uuid4(), "lv")

    def test_update_transcription_name_nonexistent_id(self, transcription_dao):
        """Test updating name for non-existent transcription ID"""
        nonexistent_id = uuid4()
        
        # This should raise an exception
        with pytest.raises(Exception):
            transcription_dao.update_transcription_name(nonexistent_id, "New Name")

    def test_update_transcription_notes_nonexistent_id(self, transcription_dao):
        """Test updating notes for non-existent transcription ID"""
        nonexistent_id = uuid4()
        
        # This should raise an exception
        with pytest.raises(Exception):
            transcription_dao.update_transcription_notes(nonexistent_id, "New Notes")

    def test_update_transcription_name_empty_string(self, transcription_dao, sample_transcription):
        """Test updating transcription name to empty string"""
        # Insert the transcription first
        transcription_dao.insert(sample_transcription)
        
        # Update the name to empty string
        transcription_dao.update_transcription_name(UUID(sample_transcription.id), "")
        
        # Verify the name was updated to empty string
        query = QSqlQuery(transcription_dao.db)
        query.prepare("SELECT name FROM transcription WHERE id = :id")
        query.bindValue(":id", sample_transcription.id)
        assert query.exec()
        assert query.next()
        assert query.value("name") == ""

    def test_update_transcription_notes_empty_string(self, transcription_dao, sample_transcription):
        """Test updating transcription notes to empty string"""
        # Insert the transcription first
        transcription_dao.insert(sample_transcription)
        
        # Update the notes to empty string
        transcription_dao.update_transcription_notes(UUID(sample_transcription.id), "")
        
        # Verify the notes were updated to empty string
        query = QSqlQuery(transcription_dao.db)
        query.prepare("SELECT notes FROM transcription WHERE id = :id")
        query.bindValue(":id", sample_transcription.id)
        assert query.exec()
        assert query.next()
        assert query.value("notes") == ""

    def test_update_transcription_name_with_none(self, transcription_dao, sample_transcription):
        """Test updating transcription name to None (should be stored as NULL)"""
        # Insert the transcription first
        transcription_dao.insert(sample_transcription)
        
        # Update the name to None
        transcription_dao.update_transcription_name(UUID(sample_transcription.id), None)
        
        # Verify the name was updated to None (NULL in database)
        query = QSqlQuery(transcription_dao.db)
        query.prepare("SELECT name FROM transcription WHERE id = :id")
        query.bindValue(":id", sample_transcription.id)
        assert query.exec()
        assert query.next()
        # In SQLite, None values are returned as empty strings
        assert query.value("name") == ""

    def test_update_transcription_notes_with_none(self, transcription_dao, sample_transcription):
        """Test updating transcription notes to None (should be stored as NULL)"""
        # Insert the transcription first
        transcription_dao.insert(sample_transcription)
        
        # Update the notes to None
        transcription_dao.update_transcription_notes(UUID(sample_transcription.id), None)
        
        # Verify the notes were updated to None (NULL in database)
        query = QSqlQuery(transcription_dao.db)
        query.prepare("SELECT notes FROM transcription WHERE id = :id")
        query.bindValue(":id", sample_transcription.id)
        assert query.exec()
        assert query.next()
        # In SQLite, None values are returned as empty strings
        assert query.value("notes") == ""

    def test_insert_transcription_without_name_and_notes(self, transcription_dao):
        """Test inserting a transcription without name and notes (should be NULL)"""
        transcription = Transcription(
            id=str(uuid4()),
            file="/path/to/test.mp3",
            status="completed",
            time_queued="2023-01-01T00:00:00",
            task="TRANSCRIBE",
            model_type="WHISPER"
            # name and notes not provided
        )
        
        transcription_dao.insert(transcription)
        
        # Verify the transcription was inserted with NULL name and notes
        query = QSqlQuery(transcription_dao.db)
        query.prepare("SELECT name, notes FROM transcription WHERE id = :id")
        query.bindValue(":id", transcription.id)
        assert query.exec()
        assert query.next()
        
        # In SQLite, NULL values are returned as empty strings
        assert query.value("name") == ""
        assert query.value("notes") == ""

    def test_database_error_handling(self, transcription_dao):
        """Test that database errors are properly handled"""
        # Mock a database error by using an invalid query
        with patch.object(transcription_dao, '_create_query') as mock_create_query:
            mock_query = Mock()
            mock_query.prepare.return_value = True
            mock_query.bindValue.return_value = None
            mock_query.exec.return_value = False
            mock_query.lastError.return_value.text.return_value = "Database error"
            mock_create_query.return_value = mock_query
            
            # This should raise an exception with the database error message
            with pytest.raises(Exception, match="Database error"):
                transcription_dao.update_transcription_name(uuid4(), "Test Name")
