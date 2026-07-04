"""Shared fixtures: a scriptable fake LLM client so no test touches the network."""

import pytest

import miniagents.react_agent as react_agent_module


class FakeChat:
    """Drop-in replacement for ChatClientWrapper with a scripted response queue."""

    def __init__(self, model_name: str, **kwargs) -> None:
        self.model = model_name
        self.script: list[str] = []
        self.calls = 0
        self.last_conversation: list[dict] | None = None

    def generate_response(self, conversation):
        self.calls += 1
        self.last_conversation = [dict(m) for m in conversation]
        if not self.script:
            raise AssertionError(
                "FakeChat script exhausted — test did not queue enough responses"
            )
        return self.script.pop(0)


@pytest.fixture(autouse=True)
def fake_llm(monkeypatch):
    """Replace the real Groq-backed client everywhere; no API key required."""
    monkeypatch.setattr(react_agent_module, "ChatClientWrapper", FakeChat)
    yield


@pytest.fixture
def weather_tool():
    from miniagents import tool

    @tool
    def get_current_weather(location: str, unit: str) -> dict:
        """Get the current weather for a location."""
        return {"location": location, "temperature": 27, "unit": unit}

    return get_current_weather
