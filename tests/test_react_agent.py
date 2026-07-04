"""Tests for the ReAct loop: tool rounds, error recovery, memory."""

import pytest

from miniagents import ReactAgent


TOOL_CALL_ROUND = (
    '<thought>need weather</thought>\n'
    '<tool_call>{"name":"get_current_weather",'
    '"arguments":{"location":"Madrid","unit":"celsius"},"id":0}</tool_call>'
)
FINAL = "<response>The weather in Madrid is 27°celsius.</response>"


class TestReactLoop:
    def test_direct_response_no_tools(self):
        agent = ReactAgent(model="mock")
        agent.chat.script = ["<response>direct answer</response>"]
        assert agent.run("hi") == "direct answer"
        assert agent.chat.calls == 1

    def test_tool_call_round_trip(self, weather_tool):
        agent = ReactAgent(tools=[weather_tool], model="mock")
        agent.chat.script = [TOOL_CALL_ROUND, FINAL]
        assert agent.run("Madrid temp?") == "The weather in Madrid is 27°celsius."
        # Observation with the tool result was fed back as a user message.
        obs = agent.chat.last_conversation[-1]
        assert obs["role"] == "user"
        assert "27" in obs["content"]
        assert "<observation>" in obs["content"]

    def test_multiple_tool_calls_in_one_turn(self, weather_tool):
        agent = ReactAgent(tools=[weather_tool], model="mock")
        agent.chat.script = [
            '<tool_call>{"name":"get_current_weather","arguments":{"location":"Madrid","unit":"celsius"},"id":0}</tool_call>'
            '<tool_call>{"name":"get_current_weather","arguments":{"location":"London","unit":"celsius"},"id":1}</tool_call>',
            "<response>done</response>",
        ]
        assert agent.run("both cities") == "done"
        obs = agent.chat.last_conversation[-1]["content"]
        assert "Madrid" in obs and "London" in obs

    def test_recovers_from_malformed_json(self, weather_tool):
        agent = ReactAgent(tools=[weather_tool], model="mock")
        agent.chat.script = [
            "<tool_call>{not valid json}</tool_call>",
            "<response>recovered</response>",
        ]
        assert agent.run("q") == "recovered"
        assert "Error: invalid tool call payload" in agent.chat.last_conversation[-1]["content"]

    def test_recovers_from_unknown_tool(self, weather_tool):
        agent = ReactAgent(tools=[weather_tool], model="mock")
        agent.chat.script = [
            '<tool_call>{"name":"nope","arguments":{},"id":0}</tool_call>',
            "<response>recovered</response>",
        ]
        assert agent.run("q") == "recovered"
        obs = agent.chat.last_conversation[-1]["content"]
        assert "not found" in obs and "get_current_weather" in obs

    def test_recovers_from_tool_exception(self):
        from miniagents import tool

        @tool
        def boom() -> str:
            """Always fails."""
            raise RuntimeError("kaput")

        agent = ReactAgent(tools=[boom], model="mock")
        agent.chat.script = [
            '<tool_call>{"name":"boom","arguments":{},"id":0}</tool_call>',
            "<response>recovered</response>",
        ]
        assert agent.run("q") == "recovered"
        assert "kaput" in agent.chat.last_conversation[-1]["content"]

    def test_nudges_on_formatless_output(self):
        agent = ReactAgent(model="mock")
        agent.chat.script = ["no tags at all", "<response>ok</response>"]
        assert agent.run("q") == "ok"
        # A nudge reminder was inserted after the formatless turn.
        contents = [m["content"] for m in agent.chat.last_conversation]
        assert any("Reminder" in c for c in contents)

    def test_max_rounds_fallback(self):
        agent = ReactAgent(model="mock")
        agent.chat.script = ["junk", "junk", "<response>fallback</response>"]
        assert agent.run("q", max_rounds=2) == "fallback"
        assert agent.chat.calls == 3  # 2 rounds + 1 final nudge call

    def test_invalid_max_rounds(self):
        agent = ReactAgent(model="mock")
        with pytest.raises(ValueError):
            agent.run("q", max_rounds=0)

    def test_tools_json_in_system_prompt(self, weather_tool):
        agent = ReactAgent(tools=[weather_tool], model="mock")
        assert "get_current_weather" in agent.react_prompt


class TestMemory:
    def test_memory_retains_conversation(self):
        agent = ReactAgent(model="mock", memory=True)
        agent.chat.script = ["<response>noted</response>"]
        agent.run("my favorite city is Madrid")
        agent.chat.script = ["<response>Madrid</response>"]
        agent.run("what is my favorite city?")
        convo = agent.chat.last_conversation
        assert any("favorite city is Madrid" in m["content"] for m in convo)
        assert any("noted" in m["content"] for m in convo if m["role"] == "assistant")

    def test_memory_keeps_single_system_message(self):
        agent = ReactAgent(model="mock", memory=True)
        for answer in ("<response>a</response>", "<response>b</response>"):
            agent.chat.script = [answer]
            agent.run("q")
        convo = agent.chat.last_conversation
        assert sum(1 for m in convo if m["role"] == "system") == 1

    def test_reset_clears_memory(self):
        agent = ReactAgent(model="mock", memory=True)
        agent.chat.script = ["<response>noted</response>"]
        agent.run("remember this")
        agent.reset()
        agent.chat.script = ["<response>fresh</response>"]
        agent.run("q")
        convo = agent.chat.last_conversation
        assert len(convo) == 2  # system + new question only
        assert not any(m["role"] == "assistant" for m in convo)

    def test_no_memory_starts_fresh_each_run(self):
        agent = ReactAgent(model="mock")
        agent.chat.script = ["<response>a</response>"]
        agent.run("first")
        agent.chat.script = ["<response>b</response>"]
        agent.run("second")
        assert len(agent.chat.last_conversation) == 2
