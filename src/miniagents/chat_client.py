"""Thin wrapper around the Groq chat completions API."""

import logging
import os
from typing import Dict, List, Optional

from dotenv import load_dotenv
from groq import Groq

from .exceptions import ConfigurationError

load_dotenv()

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

DEFAULT_TIMEOUT_SECONDS = 60.0
DEFAULT_MAX_RETRIES = 3


class ChatClientWrapper:
    """
    Wraps the Groq chat interface behind a single `generate_response` method.

    The API key is read from the GROQ_API_KEY environment variable (a local
    .env file is honored via python-dotenv) unless passed explicitly.
    """

    def __init__(
        self,
        model_name: str,
        api_key: Optional[str] = None,
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
        max_retries: int = DEFAULT_MAX_RETRIES,
    ) -> None:
        if not model_name or not model_name.strip():
            raise ConfigurationError("model_name must be a non-empty string")

        key = api_key or os.getenv("GROQ_API_KEY")
        if not key:
            raise ConfigurationError(
                "GROQ_API_KEY is not set. Export it or add it to a .env file."
            )

        # The Groq SDK retries transient failures (429s, connection errors)
        # with exponential backoff when max_retries > 0.
        self.client = Groq(api_key=key, timeout=timeout, max_retries=max_retries)
        self.model = model_name

    def generate_response(self, conversation: List[Dict[str, str]]) -> str:
        """
        Send the message list to Groq and return the assistant's text content.

        Raises whatever the Groq SDK raises after its own retries are
        exhausted; callers decide how to handle hard API failures.
        """
        if not conversation:
            raise ValueError("conversation must contain at least one message")

        try:
            resp = self.client.chat.completions.create(
                messages=conversation,
                model=self.model,
            )
        except Exception:
            logger.exception("Groq chat API call failed (model=%s)", self.model)
            raise

        content = resp.choices[0].message.content
        return content if content is not None else ""


__all__ = ["ChatClientWrapper"]
