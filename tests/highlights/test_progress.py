import logging
import time

from buzz.highlights.progress import phase


def test_phase_logs_start_and_finish(caplog):
    logger = logging.getLogger("buzz.highlights.test.phase")
    with caplog.at_level(logging.INFO, logger=logger.name):
        with phase("doing work", logger):
            pass
    messages = [record.getMessage() for record in caplog.records]
    assert any("doing work: started" in message for message in messages)
    assert any("doing work: finished in" in message for message in messages)


def test_phase_emits_heartbeat_while_running(caplog):
    logger = logging.getLogger("buzz.highlights.test.heartbeat")
    with caplog.at_level(logging.INFO, logger=logger.name):
        with phase("long work", logger, heartbeat_seconds=0.05):
            time.sleep(0.2)
    messages = [record.getMessage() for record in caplog.records]
    assert any("long work: still running" in message for message in messages)


def test_phase_logs_finish_even_when_the_body_raises(caplog):
    logger = logging.getLogger("buzz.highlights.test.raises")
    with caplog.at_level(logging.INFO, logger=logger.name):
        try:
            with phase("failing work", logger):
                raise RuntimeError("boom")
        except RuntimeError:
            pass
    messages = [record.getMessage() for record in caplog.records]
    assert any("failing work: finished in" in message for message in messages)
