"""Tests for chat client configuration and the logging helper."""

import logging

import pytest

from miniagents import ConfigurationError, configure_logging
from miniagents.chat_client import ChatClientWrapper


class TestChatClientConfig:
    def test_missing_api_key_raises(self, monkeypatch):
        monkeypatch.delenv("GROQ_API_KEY", raising=False)
        with pytest.raises(ConfigurationError, match="GROQ_API_KEY"):
            ChatClientWrapper(model_name="some-model")

    def test_explicit_api_key_accepted(self, monkeypatch):
        monkeypatch.delenv("GROQ_API_KEY", raising=False)
        client = ChatClientWrapper(model_name="some-model", api_key="explicit-key")
        assert client.model == "some-model"

    def test_empty_model_rejected(self):
        with pytest.raises(ConfigurationError, match="model_name"):
            ChatClientWrapper(model_name="  ", api_key="k")

    def test_empty_conversation_rejected(self):
        client = ChatClientWrapper(model_name="m", api_key="k")
        with pytest.raises(ValueError):
            client.generate_response([])


class TestConfigureLogging:
    def test_returns_package_logger(self):
        logger = configure_logging(level=logging.DEBUG)
        assert logger.name == "miniagents"
        assert logger.level == logging.DEBUG

    def test_no_duplicate_handlers_on_repeat_calls(self):
        configure_logging()
        configure_logging()
        logger = logging.getLogger("miniagents")
        stream_handlers = [
            h for h in logger.handlers
            if type(h) is logging.StreamHandler
        ]
        assert len(stream_handlers) == 1

    def test_log_file_handler_added(self, tmp_path):
        target = tmp_path / "logs" / "run.log"
        logger = configure_logging(log_file=target)
        logger.info("hello file")
        for h in logger.handlers:
            h.flush()
        assert target.exists()
        assert "hello file" in target.read_text()
