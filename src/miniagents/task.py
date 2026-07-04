"""Task: a unit of work assigned to an Agent, wired into a Crew DAG."""

import itertools
import logging
from textwrap import dedent
from typing import List, Optional

from .agent import Agent
from .crew import Crew
from .tracing import preview, record

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


class Task:
    """
    A unit of work executed by an Agent.

    Tasks form the nodes of the Crew dependency graph: wire them with
    `task_a >> task_b` so that B receives A's output as context. One agent
    can be assigned to any number of tasks.

    Attributes:
        description: What the task should accomplish.
        expected_output: Template or guidelines for the deliverable.
        agent: The Agent that will execute this task.
        name: Identifier used in results and graph plots; auto-generated
            from the agent name if omitted.
        output: The result of the last run() (None until executed).
    """

    _ids = itertools.count(1)

    def __init__(
        self,
        description: str,
        agent: Agent,
        expected_output: Optional[str] = None,
        name: Optional[str] = None,
    ) -> None:
        if not description or not description.strip():
            raise ValueError("Task requires a non-empty description")
        if not isinstance(agent, Agent):
            raise TypeError("Task 'agent' must be an Agent instance")

        self.description: str = description
        self.expected_output: Optional[str] = expected_output
        self.agent: Agent = agent
        self.name: str = (name or f"{agent.name}_task_{next(Task._ids)}").strip()

        self.dependencies: List["Task"] = []
        self.dependents: List["Task"] = []
        self.output: Optional[str] = None

        # Automatically register in the active Crew (if any).
        Crew.register(self)
        logger.debug("Task %s (agent=%s) initialized", self.name, agent.name)

    def __repr__(self) -> str:
        return f"<Task {self.name!r} agent={self.agent.name!r}>"

    def __rshift__(self, other: "Task") -> "Task":
        """`task_a >> task_b` makes A a dependency of B; returns B for chaining."""
        other.add_dependency(self)
        return other

    def __lshift__(self, other: "Task") -> "Task":
        """`task_b << task_a` makes A a dependency of B; returns A for chaining."""
        self.add_dependency(other)
        return other

    def add_dependency(self, upstream: "Task") -> None:
        """Declare that this task depends on `upstream`'s output."""
        if not isinstance(upstream, Task):
            raise TypeError("Dependency must be a Task instance")
        if upstream is self:
            raise ValueError(f"Task {self.name!r} cannot depend on itself")
        if upstream not in self.dependencies:
            self.dependencies.append(upstream)
            upstream.dependents.append(self)
        logger.debug("%s now depends on %s", self.name, upstream.name)

    def _build_prompt(self) -> str:
        # Context is pulled from upstream outputs at run time; topological
        # execution order guarantees dependencies have already run.
        context_lines = [
            f"Output from upstream task {dep.name!r} (by agent {dep.agent.name!r}): {dep.output}"
            for dep in self.dependencies
            if dep.output is not None
        ]
        context_block = "\n".join(context_lines)
        return dedent(
            f"""
            <task_description>
            {self.description}
            </task_description>

            <task_expected_output>
            {self.expected_output or ""}
            </task_expected_output>

            <context>
            {context_block}
            </context>
            """
        ).strip()

    def run(self) -> str:
        """Build the prompt from description + upstream context and execute it."""
        prompt = self._build_prompt()
        logger.info("Task %s running (agent=%s)", self.name, self.agent.name)
        logger.debug("Task %s prompt:\n%s", self.name, prompt)
        record(
            "task_prompt",
            name=self.name,
            agent=self.agent.name,
            dependencies=[d.name for d in self.dependencies],
            prompt_preview=preview(prompt),
        )

        self.output = self.agent.execute(prompt)
        return self.output


__all__ = ["Task"]
