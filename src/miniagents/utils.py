"""Parsing and message-history helpers shared across the framework."""

import logging
import re
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


class TagExtractionResult:
    """
    Outcome of parsing XML-like tag content out of model output.

    Attributes:
        items: The extracted strings (whitespace-trimmed).
        found: True if at least one item was extracted.
    """

    __slots__ = ("items", "found")

    def __init__(self, items: List[str]) -> None:
        self.items: List[str] = [s.strip() for s in items]
        self.found: bool = bool(self.items)


class TagParser:
    """
    Extracts every occurrence of content wrapped in a specific tag,
    e.g. <thought>...</thought>.
    """

    def __init__(self, tag_name: str) -> None:
        if not tag_name.isidentifier():
            raise ValueError(f"Invalid tag name: {tag_name!r}")
        self._tag = tag_name
        self._pattern = re.compile(
            rf"<{re.escape(tag_name)}>(.*?)</{re.escape(tag_name)}>",
            re.DOTALL,
        )

    def parse(self, text: str) -> TagExtractionResult:
        if not isinstance(text, str):
            raise TypeError(
                f"TagParser.parse expected str, got {type(text).__name__}"
            )
        return TagExtractionResult(self._pattern.findall(text))


def create_message(role: str, content: str, tag: Optional[str] = None) -> Dict[str, str]:
    """
    Build a single message dict for the chat API, optionally wrapping
    the content in an XML-like tag.
    """
    if role not in ("system", "user", "assistant"):
        raise ValueError(f"Invalid message role: {role!r}")
    if tag:
        content = f"<{tag}>{content}</{tag}>"
    return {"role": role, "content": content}


class MessageHistory:
    """
    Keeps a list of messages, dropping the oldest once max_size is reached.
    An unset or non-positive max_size means unbounded.
    """

    def __init__(self, max_size: Optional[int] = None) -> None:
        self._msgs: List[Dict[str, str]] = []
        self._max = max_size if (max_size and max_size > 0) else None

    def __len__(self) -> int:
        return len(self._msgs)

    def append(self, message: Dict[str, str]) -> None:
        if self._max and len(self._msgs) >= self._max:
            self._msgs.pop(0)
        self._msgs.append(message)

    def extend(self, messages: List[Dict[str, str]]) -> None:
        for msg in messages:
            self.append(msg)

    def all(self) -> List[Dict[str, str]]:
        return self._msgs.copy()


class PinnedMessageHistory(MessageHistory):
    """
    Like MessageHistory but always retains the first (pinned) message —
    typically the system prompt — even when the history is full.
    """

    def append(self, message: Dict[str, str]) -> None:
        if self._max and len(self._msgs) >= self._max and len(self._msgs) > 1:
            self._msgs.pop(1)
        self._msgs.append(message)


__all__ = [
    "TagParser",
    "TagExtractionResult",
    "create_message",
    "MessageHistory",
    "PinnedMessageHistory",
]
