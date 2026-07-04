#!/usr/bin/env python3
"""
Runnable examples for miniagents.

Usage:
    python examples.py weather    # single agent + task with a tool (default)
    python examples.py pipeline   # three-task dependency pipeline
    python examples.py memory     # one agent remembering across tasks

Each run writes a full execution trace to traces/<run_id>.jsonl and prints
a trace summary (LLM calls, tool calls, timing) at the end.
"""

import json
import sys

from miniagents import Agent, Crew, Task, configure_logging, tool

configure_logging(level="INFO")


@tool
def get_current_weather(location: str, unit: str) -> dict:
    """
    Get the current weather for a location.
    location: city name, e.g. 'Madrid'
    unit: 'celsius' or 'fahrenheit'
    """
    mapping = {
        ("Madrid", "celsius"): {"location": "Madrid", "temperature": 27, "unit": "celsius"},
        ("New York", "fahrenheit"): {"location": "New York", "temperature": 77, "unit": "fahrenheit"},
        ("London", "celsius"): {"location": "London", "temperature": 15, "unit": "celsius"},
    }
    return mapping.get((location, unit), {"location": location, "temperature": 20, "unit": unit})


def run_and_report(crew: Crew) -> None:
    results = crew.run_all()
    trace = crew.last_trace
    saved = trace.save(f"traces/{trace.run_id}.jsonl")

    for name, output in results.items():
        print(f"\n[{name}] {output}")
    print(f"\nTrace summary: {json.dumps(trace.summary(), indent=2)}")
    print(f"Trace saved to: {saved}")


def run_weather_example() -> None:
    """Single agent assigned a task that requires a tool call."""
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
        run_and_report(crew)


def run_pipeline_example() -> None:
    """Three tasks chained so each consumes the previous one's output."""
    thinker = Agent(
        role="Creative Thinker",
        goal="Brainstorm original, practical ideas.",
        backstory="You are a creative thinker who brainstorms ideas.",
    )
    editor = Agent(
        role="Editor",
        goal="Turn rough ideas into clear, concise points.",
        backstory="You are an editor with a ruthless eye for clarity.",
    )
    writer = Agent(
        role="Professional Writer",
        goal="Craft engaging, polished prose.",
        backstory="You are a professional writer who crafts engaging paragraphs.",
    )

    with Crew() as crew:
        brainstorm = Task(
            description="Generate three novel approaches to improve user onboarding.",
            expected_output="- A bullet list of 3 distinct ideas",
            agent=thinker,
            name="brainstorm",
        )
        refine = Task(
            description="Take the brainstormed ideas and refine each into a single clear sentence.",
            expected_output="- One refined sentence per idea",
            agent=editor,
            name="refine",
        )
        compose = Task(
            description="Compose a short paragraph introducing the three refined ideas.",
            expected_output="A 3-sentence introductory paragraph",
            agent=writer,
            name="compose",
        )

        brainstorm >> refine >> compose
        run_and_report(crew)


def run_memory_example() -> None:
    """
    One agent with memory=True executing two tasks. The second task has NO
    dependency edge to the first — the agent answers from its own memory.
    """
    assistant = Agent(
        role="Personal Assistant",
        goal="Help the user and remember what they tell you.",
        backstory="You are attentive and never forget details the user shares.",
        memory=True,
    )

    with Crew() as crew:
        Task(
            description="The user's favorite city is Madrid. Acknowledge that you noted it.",
            expected_output="A one-sentence acknowledgement.",
            agent=assistant,
            name="remember_city",
        )
        Task(
            description="What is the user's favorite city? Answer from what you already know.",
            expected_output="The city name.",
            agent=assistant,
            name="recall_city",
        )
        # Note: no >> edge between the tasks — recall relies purely on agent memory.
        run_and_report(crew)


if __name__ == "__main__":
    example = sys.argv[1] if len(sys.argv) > 1 else "weather"
    runners = {
        "weather": run_weather_example,
        "pipeline": run_pipeline_example,
        "memory": run_memory_example,
    }
    if example not in runners:
        sys.exit(f"Unknown example {example!r}. Choose one of: {', '.join(runners)}.")
    runners[example]()
