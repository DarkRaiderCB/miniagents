"""Tests for execution tracing and trace persistence."""

import json

from miniagents import Agent, Crew, Task, Tracer, current_tracer
from miniagents.tracing import preview, record


def make_agent(role="Writer", **kwargs) -> Agent:
    return Agent(role=role, goal="g", backstory="b", **kwargs)


class TestTracer:
    def test_events_get_sequential_ids_and_shared_run_id(self):
        t = Tracer()
        t.record("one")
        t.record("two")
        assert [e.seq for e in t.events] == [0, 1]
        assert all(e.run_id == t.run_id for e in t.events)

    def test_ambient_tracer_via_context_manager(self):
        assert current_tracer() is None
        with Tracer() as t:
            assert current_tracer() is t
            record("inside", name="x", detail=1)
        assert current_tracer() is None
        assert t.events[0].event == "inside"
        assert t.events[0].data == {"detail": 1}

    def test_record_is_noop_without_tracer(self):
        record("nothing-happens")  # must not raise

    def test_preview_truncates(self):
        assert preview("x" * 500).endswith("…")
        assert len(preview("x" * 500)) == 200
        assert preview("short") == "short"

    def test_summary_aggregates(self):
        t = Tracer()
        t.record("llm_call", elapsed_ms=10.0)
        t.record("llm_call", elapsed_ms=5.0)
        t.record("tool_call")
        t.record("tool_error", error="x")
        s = t.summary()
        assert s["llm_calls"] == 2
        assert s["llm_time_ms"] == 15.0
        assert s["tool_calls"] == 1
        assert s["errors"] == 1
        assert s["total_events"] == 4


class TestCrewTracing:
    def _run_traced_crew(self, tmp_path=None, weather_tool=None):
        agent = make_agent(tools=[weather_tool] if weather_tool else None)
        with Crew() as crew:
            Task(description="d", agent=agent, name="t1")
            agent.react_agent.chat.script = ["<response>done</response>"]
            kwargs = {"save_trace_to": tmp_path} if tmp_path else {}
            results = crew.run_all(**kwargs)
        return crew, results

    def test_run_all_produces_trace(self):
        crew, results = self._run_traced_crew()
        assert results == {"t1": "done"}
        events = [e.event for e in crew.last_trace.events]
        for expected in ("crew_start", "task_start", "task_prompt",
                         "agent_execute", "llm_call", "react_response",
                         "task_end", "crew_end"):
            assert expected in events, f"missing {expected} in {events}"

    def test_tool_calls_traced(self, weather_tool):
        agent = make_agent(tools=[weather_tool])
        with Crew() as crew:
            Task(description="d", agent=agent, name="t1")
            agent.react_agent.chat.script = [
                '<tool_call>{"name":"get_current_weather",'
                '"arguments":{"location":"Madrid","unit":"celsius"},"id":0}</tool_call>',
                "<response>done</response>",
            ]
            crew.run_all()
        events = crew.last_trace.events
        tool_events = [e for e in events if e.event in ("tool_call", "tool_result")]
        assert [e.event for e in tool_events] == ["tool_call", "tool_result"]
        assert tool_events[0].data["arguments"] == {"location": "Madrid", "unit": "celsius"}

    def test_task_error_traced_and_reraised(self):
        agent = make_agent()

        class ExplodingTask(Task):
            def run(self):
                raise RuntimeError("boom")

        with Crew() as crew:
            ExplodingTask(description="d", agent=agent, name="bad")
            try:
                crew.run_all()
                raised = False
            except RuntimeError:
                raised = True
        assert raised
        events = [e.event for e in crew.last_trace.events]
        assert "task_error" in events
        assert "crew_end" in events  # crew_end recorded even on failure

    def test_trace_saved_as_jsonl(self, tmp_path):
        target = tmp_path / "trace.jsonl"
        crew, _ = self._run_traced_crew(tmp_path=target)
        lines = target.read_text().strip().splitlines()
        assert len(lines) == len(crew.last_trace.events)
        parsed = [json.loads(line) for line in lines]
        assert parsed[0]["event"] == "crew_start"
        assert parsed[-1]["event"] == "crew_end"
        assert all(p["run_id"] == crew.last_trace.run_id for p in parsed)

    def test_shared_tracer_across_runs(self):
        shared = Tracer(run_id="shared-run")
        agent = make_agent()
        with Crew() as crew:
            Task(description="d", agent=agent, name="t1")
            agent.react_agent.chat.script = ["<response>a</response>"]
            crew.run_all(tracer=shared)
        assert crew.last_trace is shared
        assert shared.run_id == "shared-run"
