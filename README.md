# miniagents

A minimal, from-scratch multi-agent LLM framework (in the spirit of CrewAI) built on the
[Groq API](https://groq.com). Persona-defined agents run a prompt-engineered **ReAct loop**
(Thought → Action → Observation) with Python functions as tools, and execute **Tasks**
composed into a dependency DAG that runs in topological order. Every run produces a
structured, exportable **execution trace**.

## Project layout

```
src/miniagents/
├── __init__.py       Public API (Agent, Task, Crew, tool, Tracer, ...)
├── agent.py          Agent: reusable persona (role, goal, backstory, tools, memory)
├── task.py           Task: unit of work assigned to an agent; DAG node
├── crew.py           Crew: collects tasks, topo-sorts, runs them, traces the run
├── react_agent.py    The ReAct loop (tag parsing, tool rounds, memory)
├── tool.py           @tool decorator: signature introspection + argument validation
├── chat_client.py    Groq chat completions wrapper (timeouts, retries)
├── tracing.py        Tracer/TraceEvent: run_id-tagged JSONL execution records
├── log.py            configure_logging() helper
├── utils.py          Tag parsing and message-history helpers
└── exceptions.py     Framework exception hierarchy
tests/                Pytest suite (81 tests, no network needed)
examples.py           Runnable demos
```

## Setup

Requires Python 3.11+.

```bash
uv sync                 # or: pip install -e .
echo 'GROQ_API_KEY=your-key-here' > .env
```

Optional extras: `uv sync --extra viz` for `Crew.plot()` dependency graphs.

## Quick start

```python
from miniagents import Agent, Crew, Task, tool

@tool
def get_current_weather(location: str, unit: str) -> dict:
    """Get the current weather for a location."""
    return {"location": location, "temperature": 27, "unit": unit}

weather_agent = Agent(
    role="Weather Assistant",
    goal="Answer weather questions accurately using the available tools.",
    backstory="For any temperature query, you MUST call the get_current_weather tool.",
    tools=[get_current_weather],
)

with Crew() as crew:
    Task(
        description="What is the current temperature in Madrid in celsius?",
        expected_output="The weather in <location> is <temperature>°<unit>.",
        agent=weather_agent,
        name="madrid_weather",
    )
    results = crew.run_all(save_trace_to="traces/weather.jsonl")
    # {'madrid_weather': 'The weather in Madrid is 27°celsius.'}
```

Chain tasks with `>>` so downstream tasks receive upstream outputs as context;
one agent can serve many tasks, and `memory=True` agents remember earlier
executions even without a context edge:

```python
brainstorm >> refine >> compose
assistant = Agent(role="PA", goal="...", backstory="...", memory=True)
assistant.reset_memory()
```

## Logging and traceability

Applications opt into logging with:

```python
from miniagents import configure_logging
configure_logging(level="INFO", log_file="logs/miniagents.log")
```

Every `crew.run_all()` is recorded by a `Tracer` under a unique `run_id`:
crew/task boundaries, every LLM call (round, message count, latency, response
preview), every tool call (arguments, latency, result preview), and every
error. Inspect it programmatically or persist it as JSON Lines:

```python
results = crew.run_all(save_trace_to="traces/run.jsonl")
crew.last_trace.summary()
# {'run_id': 'a1b2c3...', 'duration_s': 1.9, 'llm_calls': 2,
#  'llm_time_ms': 1834.2, 'tool_calls': 1, 'errors': 0, ...}
```

Standalone (non-crew) usage can be traced too:

```python
from miniagents import Tracer
with Tracer() as t:
    agent.execute("...")
t.save("traces/adhoc.jsonl")
```

Tool-call failures (malformed JSON, unknown tools, bad arguments, tool
exceptions) are fed back to the model as observations so it can self-correct
instead of crashing the run — and each one is recorded as a `tool_error`
trace event.

## Tests

```bash
uv run pytest
```

The suite covers the ReAct loop (tool rounds, error recovery, nudging, max-rounds
fallback, memory), tool introspection/validation/coercion, Agent/Task/Crew
orchestration (context passing, cycles, duplicate names, re-run idempotency),
tracing, and logging — all against a scripted fake LLM, no API key needed.

## Examples

```bash
python examples.py weather    # single agent + task calling a tool
python examples.py pipeline   # three-task chained pipeline
python examples.py memory     # agent recalls earlier task via memory
```

Each example prints its trace summary and writes `traces/<run_id>.jsonl`.

## Notes

- Default model: `llama-3.3-70b-versatile`; override per-agent via `llm_model=`.
- `Crew` uses a process-global active-crew pointer; don't build crews
  concurrently from multiple threads.
- Tool functions should have type annotations and a docstring — both are shown
  to the model and used for argument validation.
- Task names must be unique within a crew (auto-generated if omitted).
