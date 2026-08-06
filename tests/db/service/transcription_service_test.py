import pytest
from unittest.mock import Mock, patch
from uuid import uuid4

from buzz.db.service.transcription_service import TranscriptionService
from buzz.db.entity.transcription import Transcription
from buzz.transcriber.transcriber import FileTranscriptionTask, Segment


@pytest.fixture
def mock_transcription_dao():
    """Create a mock TranscriptionDAO for testing"""
    return Mock()


@pytest.fixture
def mock_transcription_segment_dao():
    """Create a mock TranscriptionSegmentDAO for testing"""
    return Mock()


@pytest.fixture
def transcription_service(mock_transcription_dao, mock_transcription_segment_dao):
    """Create a TranscriptionService instance for testing"""
    return TranscriptionService(mock_transcription_dao, mock_transcription_segment_dao)


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


class TestTranscriptionService:
    def test_recovery_updates_delegate_to_dao(
        self, transcription_service, mock_transcription_dao
    ):
        transcription_id = uuid4()
        task = Mock(spec=FileTranscriptionTask)

        transcription_service.update_transcription_download_progress(
            transcription_id, 0.5
        )
        transcription_service.update_transcription_segment_checkpoint(
            transcription_id, task
        )
        transcription_service.update_transcription_source_file_fingerprint(
            transcription_id, "v1:fingerprint"
        )
        transcription_service.get_unfinished_transcriptions()
        transcription_service.queue_transcription_for_recovery(transcription_id)

        mock_transcription_dao.update_transcription_download_progress.assert_called_once_with(
            transcription_id, 0.5
        )
        mock_transcription_dao.update_transcription_segment_checkpoint.assert_called_once_with(
            transcription_id, task
        )
        mock_transcription_dao.update_transcription_source_file_fingerprint.assert_called_once_with(
            transcription_id, "v1:fingerprint"
        )
        mock_transcription_dao.get_unfinished_transcriptions.assert_called_once_with()
        mock_transcription_dao.queue_transcription_for_recovery.assert_called_once_with(
            transcription_id
        )

    def test_update_transcription_name(self, transcription_service, mock_transcription_dao):
        """Test updating transcription name through service"""
        transcription_id = uuid4()
        new_name = "Updated Transcription Name"
        
        # Call the service method
        transcription_service.update_transcription_name(transcription_id, new_name)
        
        # Verify the DAO method was called with correct parameters
        mock_transcription_dao.update_transcription_name.assert_called_once_with(transcription_id, new_name)

    def test_update_transcription_notes(self, transcription_service, mock_transcription_dao):
        """Test updating transcription notes through service"""
        transcription_id = uuid4()
        new_notes = "Updated transcription notes with more details"

        # Call the service method
        transcription_service.update_transcription_notes(transcription_id, new_notes)

        # Verify the DAO method was called with correct parameters
        mock_transcription_dao.update_transcription_notes.assert_called_once_with(transcription_id, new_notes)

    def test_update_transcription_language(self, transcription_service, mock_transcription_dao):
        """Test updating transcription language through service"""
        transcription_id = uuid4()

        transcription_service.update_transcription_language(transcription_id, "lv")

        mock_transcription_dao.update_transcription_language.assert_called_once_with(
            transcription_id, "lv"
        )

    def test_update_transcription_name_with_empty_string(self, transcription_service, mock_transcription_dao):
        """Test updating transcription name to empty string"""
        transcription_id = uuid4()
        empty_name = ""
        
        # Call the service method
        transcription_service.update_transcription_name(transcription_id, empty_name)
        
        # Verify the DAO method was called with empty string
        mock_transcription_dao.update_transcription_name.assert_called_once_with(transcription_id, empty_name)

    def test_update_transcription_notes_with_empty_string(self, transcription_service, mock_transcription_dao):
        """Test updating transcription notes to empty string"""
        transcription_id = uuid4()
        empty_notes = ""
        
        # Call the service method
        transcription_service.update_transcription_notes(transcription_id, empty_notes)
        
        # Verify the DAO method was called with empty string
        mock_transcription_dao.update_transcription_notes.assert_called_once_with(transcription_id, empty_notes)

    def test_update_transcription_name_with_none(self, transcription_service, mock_transcription_dao):
        """Test updating transcription name to None"""
        transcription_id = uuid4()
        
        # Call the service method
        transcription_service.update_transcription_name(transcription_id, None)
        
        # Verify the DAO method was called with None
        mock_transcription_dao.update_transcription_name.assert_called_once_with(transcription_id, None)

    def test_update_transcription_notes_with_none(self, transcription_service, mock_transcription_dao):
        """Test updating transcription notes to None"""
        transcription_id = uuid4()
        
        # Call the service method
        transcription_service.update_transcription_notes(transcription_id, None)
        
        # Verify the DAO method was called with None
        mock_transcription_dao.update_transcription_notes.assert_called_once_with(transcription_id, None)

    def test_update_transcription_name_propagates_dao_exception(self, transcription_service, mock_transcription_dao):
        """Test that DAO exceptions are propagated from service"""
        transcription_id = uuid4()
        new_name = "Updated Name"
        
        # Configure the mock to raise an exception
        mock_transcription_dao.update_transcription_name.side_effect = Exception("Database error")
        
        # Call the service method and expect the exception to be raised
        with pytest.raises(Exception, match="Database error"):
            transcription_service.update_transcription_name(transcription_id, new_name)

    def test_update_transcription_notes_propagates_dao_exception(self, transcription_service, mock_transcription_dao):
        """Test that DAO exceptions are propagated from service"""
        transcription_id = uuid4()
        new_notes = "Updated notes"
        
        # Configure the mock to raise an exception
        mock_transcription_dao.update_transcription_notes.side_effect = Exception("Database error")
        
        # Call the service method and expect the exception to be raised
        with pytest.raises(Exception, match="Database error"):
            transcription_service.update_transcription_notes(transcription_id, new_notes)

    def test_update_transcription_name_with_string_uuid(self, transcription_service, mock_transcription_dao):
        """Test updating transcription name with string UUID (should be converted to UUID)"""
        transcription_id_str = str(uuid4())
        new_name = "Updated Name"
        
        # Call the service method
        transcription_service.update_transcription_name(transcription_id_str, new_name)
        
        # Verify the DAO method was called with UUID object
        mock_transcription_dao.update_transcription_name.assert_called_once()
        call_args = mock_transcription_dao.update_transcription_name.call_args[0]
        assert isinstance(call_args[0], str)  # The service should pass the string as-is
        assert call_args[1] == new_name

    def test_update_transcription_notes_with_string_uuid(self, transcription_service, mock_transcription_dao):
        """Test updating transcription notes with string UUID (should be converted to UUID)"""
        transcription_id_str = str(uuid4())
        new_notes = "Updated notes"
        
        # Call the service method
        transcription_service.update_transcription_notes(transcription_id_str, new_notes)
        
        # Verify the DAO method was called with UUID object
        mock_transcription_dao.update_transcription_notes.assert_called_once()
        call_args = mock_transcription_dao.update_transcription_notes.call_args[0]
        assert isinstance(call_args[0], str)  # The service should pass the string as-is
        assert call_args[1] == new_notes

    def test_update_transcription_name_multiple_calls(self, transcription_service, mock_transcription_dao):
        """Test multiple calls to update transcription name"""
        transcription_id = uuid4()
        
        # Make multiple calls
        transcription_service.update_transcription_name(transcription_id, "Name 1")
        transcription_service.update_transcription_name(transcription_id, "Name 2")
        transcription_service.update_transcription_name(transcription_id, "Name 3")
        
        # Verify all calls were made
        assert mock_transcription_dao.update_transcription_name.call_count == 3
        
        # Verify the last call has the correct parameters
        last_call = mock_transcription_dao.update_transcription_name.call_args_list[-1]
        assert last_call[0] == (transcription_id, "Name 3")

    def test_update_transcription_notes_multiple_calls(self, transcription_service, mock_transcription_dao):
        """Test multiple calls to update transcription notes"""
        transcription_id = uuid4()
        
        # Make multiple calls
        transcription_service.update_transcription_notes(transcription_id, "Notes 1")
        transcription_service.update_transcription_notes(transcription_id, "Notes 2")
        transcription_service.update_transcription_notes(transcription_id, "Notes 3")
        
        # Verify all calls were made
        assert mock_transcription_dao.update_transcription_notes.call_count == 3
        
        # Verify the last call has the correct parameters
        last_call = mock_transcription_dao.update_transcription_notes.call_args_list[-1]
        assert last_call[0] == (transcription_id, "Notes 3")

    def test_update_transcription_name_with_unicode(self, transcription_service, mock_transcription_dao):
        """Test updating transcription name with unicode characters"""
        transcription_id = uuid4()
        unicode_name = "Transcription avec des caractères spéciaux: ñáéíóú"
        
        # Call the service method
        transcription_service.update_transcription_name(transcription_id, unicode_name)
        
        # Verify the DAO method was called with unicode string
        mock_transcription_dao.update_transcription_name.assert_called_once_with(transcription_id, unicode_name)

    def test_update_transcription_notes_with_unicode(self, transcription_service, mock_transcription_dao):
        """Test updating transcription notes with unicode characters"""
        transcription_id = uuid4()
        unicode_notes = "Notes avec des caractères spéciaux: ñáéíóú et émojis 🎵🎤"
        
        # Call the service method
        transcription_service.update_transcription_notes(transcription_id, unicode_notes)
        
        # Verify the DAO method was called with unicode string
        mock_transcription_dao.update_transcription_notes.assert_called_once_with(transcription_id, unicode_notes)


class TestTranscriptionServiceTransactions:
    """Real-DAO tests: completion/segments and replace are atomic."""

    @pytest.fixture()
    def transcription_service(
        self, transcription_dao, transcription_segment_dao
    ) -> TranscriptionService:
        return TranscriptionService(transcription_dao, transcription_segment_dao)

    @pytest.fixture()
    def transcription(self, transcription_dao) -> Transcription:
        id = uuid4()
        transcription_dao.insert(
            Transcription(
                id=str(id),
                status="in_progress",
                file="/tmp/test.mp3",
                task="TRANSCRIBE",
                model_type="WHISPER",
            )
        )
        return transcription_dao.find_by_id(str(id))

    def test_completed_inserts_segments_then_marks_completed(
        self, transcription, transcription_service, transcription_segment_dao
    ):
        segments = [
            Segment(start=0, end=100, text="one"),
            Segment(start=100, end=200, text="two"),
        ]

        transcription_service.update_transcription_as_completed(
            transcription.id_as_uuid, segments
        )

        from PyQt6.QtSql import QSqlQuery
        query = QSqlQuery(transcription_segment_dao.db)
        query.prepare("SELECT status FROM transcription WHERE id = :id")
        query.bindValue(":id", transcription.id)
        assert query.exec() and query.next()
        assert query.value(0) == "completed"

        persisted = transcription_segment_dao.get_segments(
            transcription.id_as_uuid
        )
        assert [(s.start_time, s.end_time, s.text) for s in persisted] == [
            (0, 100, "one"),
            (100, 200, "two"),
        ]

    def test_completed_rolls_back_status_when_segment_insert_fails(
        self, transcription, transcription_service, transcription_segment_dao
    ):
        from buzz.db.dao.transcription_segment_dao import TranscriptionSegmentDAO

        with patch.object(
            TranscriptionSegmentDAO,
            "insert",
            side_effect=Exception("disk full"),
        ):
            with pytest.raises(Exception, match="disk full"):
                transcription_service.update_transcription_as_completed(
                    transcription.id_as_uuid,
                    [Segment(start=0, end=100, text="one")],
                )

        # Neither the status nor the segments were committed.
        from PyQt6.QtSql import QSqlQuery
        query = QSqlQuery(transcription_segment_dao.db)
        query.prepare("SELECT status FROM transcription WHERE id = :id")
        query.bindValue(":id", transcription.id)
        assert query.exec() and query.next()
        assert query.value(0) == "in_progress"
        assert transcription_segment_dao.get_segments(
            transcription.id_as_uuid
        ) == []

    def test_replace_segments_is_atomic(
        self, transcription, transcription_service, transcription_segment_dao
    ):
        transcription_service.update_transcription_as_completed(
            transcription.id_as_uuid,
            [Segment(start=0, end=100, text="old")],
        )

        transcription_service.replace_transcription_segments(
            transcription.id_as_uuid,
            [Segment(start=0, end=50, text="new1"),
             Segment(start=50, end=100, text="new2")],
        )

        persisted = transcription_segment_dao.get_segments(
            transcription.id_as_uuid
        )
        assert [(s.start_time, s.text) for s in persisted] == [
            (0, "new1"), (50, "new2"),
        ]

    def test_replace_rolls_back_on_failure(
        self, transcription, transcription_service, transcription_segment_dao
    ):
        transcription_service.update_transcription_as_completed(
            transcription.id_as_uuid,
            [Segment(start=0, end=100, text="old")],
        )

        from buzz.db.dao.transcription_segment_dao import TranscriptionSegmentDAO

        with patch.object(
            TranscriptionSegmentDAO,
            "insert",
            side_effect=Exception("disk full"),
        ):
            with pytest.raises(Exception, match="disk full"):
                transcription_service.replace_transcription_segments(
                    transcription.id_as_uuid,
                    [Segment(start=0, end=50, text="new1")],
                )

        # The delete was rolled back: the old segments are still there.
        persisted = transcription_segment_dao.get_segments(
            transcription.id_as_uuid
        )
        assert [(s.start_time, s.text) for s in persisted] == [(0, "old")]

    def test_get_segments_returns_ordered_by_start_time(
        self, transcription, transcription_service, transcription_segment_dao
    ):
        from buzz.db.entity.transcription_segment import TranscriptionSegment

        for start, text in [(500, "third"), (0, "first"), (200, "second")]:
            transcription_segment_dao.insert(
                TranscriptionSegment(
                    start_time=start,
                    end_time=start + 100,
                    text=text,
                    translation="",
                    transcription_id=transcription.id,
                )
            )

        persisted = transcription_segment_dao.get_segments(
            transcription.id_as_uuid
        )
        assert [s.text for s in persisted] == ["first", "second", "third"]
