"""Tests for tag parsing and message history helpers."""

import pytest

from miniagents.utils import (
    MessageHistory,
    PinnedMessageHistory,
    TagParser,
    create_message,
)


class TestTagParser:
    def test_extracts_single_tag(self):
        result = TagParser("response").parse("<response>hello</response>")
        assert result.found
        assert result.items == ["hello"]

    def test_extracts_multiple_tags(self):
        result = TagParser("tool_call").parse(
            "<tool_call>{'a':1}</tool_call> junk <tool_call>{'b':2}</tool_call>"
        )
        assert result.items == ["{'a':1}", "{'b':2}"]

    def test_multiline_content(self):
        result = TagParser("thought").parse("<thought>line1\nline2</thought>")
        assert result.items == ["line1\nline2"]

    def test_no_match(self):
        result = TagParser("response").parse("no tags here")
        assert not result.found
        assert result.items == []

    def test_whitespace_trimmed(self):
        result = TagParser("response").parse("<response>  padded  </response>")
        assert result.items == ["padded"]

    def test_invalid_tag_name_rejected(self):
        with pytest.raises(ValueError):
            TagParser("not a tag!")

    def test_non_string_input_rejected(self):
        with pytest.raises(TypeError):
            TagParser("response").parse(None)


class TestCreateMessage:
    def test_basic(self):
        assert create_message("user", "hi") == {"role": "user", "content": "hi"}

    def test_with_tag(self):
        msg = create_message("user", "hi", tag="question")
        assert msg["content"] == "<question>hi</question>"

    def test_invalid_role_rejected(self):
        with pytest.raises(ValueError):
            create_message("robot", "hi")


class TestMessageHistory:
    def test_unbounded_by_default(self):
        h = MessageHistory()
        for i in range(100):
            h.append({"role": "user", "content": str(i)})
        assert len(h) == 100

    def test_max_size_drops_oldest(self):
        h = MessageHistory(max_size=2)
        for i in range(3):
            h.append({"role": "user", "content": str(i)})
        assert [m["content"] for m in h.all()] == ["1", "2"]

    def test_all_returns_copy(self):
        h = MessageHistory()
        h.append({"role": "user", "content": "x"})
        h.all().clear()
        assert len(h) == 1


class TestPinnedMessageHistory:
    def test_pins_first_message(self):
        h = PinnedMessageHistory(max_size=3)
        h.append({"role": "system", "content": "pinned"})
        for i in range(5):
            h.append({"role": "user", "content": str(i)})
        msgs = h.all()
        assert msgs[0]["content"] == "pinned"
        assert len(msgs) <= 4  # pinned + up to max_size window

    def test_max_size_one_does_not_crash(self):
        h = PinnedMessageHistory(max_size=1)
        h.append({"role": "system", "content": "pinned"})
        h.append({"role": "user", "content": "u1"})  # regression: used to IndexError
        assert h.all()[0]["content"] == "pinned"
