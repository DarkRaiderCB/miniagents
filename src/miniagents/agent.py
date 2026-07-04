"""High-level Agent: a reusable, persona-defined ReAct agent.

Agents carry no task; they execute any number of Task objects (see task.py),
mirroring CrewAI's Agent/Task split.
"""

import logging
from textwrap import dedent
from typing import List, Optional, Union

from .react_agent import DEFAULT_MODEL, ReactAgent
from .tool import Tool
from .tracing import preview, record

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


class Agent:
    """
    A persona-defined AI agent that can execute tasks.

    Attributes:
        role: The agent's function/expertise (e.g. "Senior Financial Analyst").
        goal: The individual objective that guides the agent's decisions.
        backstory: Narrative context that shapes the agent's behavior.
        tools: Tool instances available to this agent.
        llm_model: Name of the language model that drives the agent.
        memory: When True, the agent remembers prior task executions
            (conversation history persists across execute() calls).
        name: Display identifier; defaults to the role.
    """

    def __init__(
        self,
        role: str,
        goal: str,
        backstory: str,
        tools: Optional[Union[Tool, List[Tool]]] = None,
        llm_model: str = DEFAULT_MODEL,
        memory: bool = False,
        name: Optional[str] = None,
    ) -> None:
        if not role or not role.strip():
            raise ValueError("Agent role must be a non-empty string")
        if not goal or not goal.strip():
            raise ValueError(f"Agent {role!r} requires a goal")

        self.role: str = role.strip()
        self.goal: str = goal.strip()
        self.backstory: str = backstory
        self.name: str = (name or self.role).strip()
        if tools is None:
            tools = []
        self.tools: List[Tool] = tools if isinstance(tools, list) else [tools]
        self.llm_model: str = llm_model
        self.memory: bool = memory

        self.react_agent = ReactAgent(
            tools=self.tools,
            model=self.llm_model,
            system_prompt=self._compile_system_prompt(),
            memory=memory,
        )
        logger.debug("Agent %s initialized (model=%s, tools=%d, memory=%s)",
                     self.name, self.llm_model, len(self.tools), self.memory)

    def _compile_system_prompt(self) -> str:
        return dedent(
            f"""
            You are {self.role}.
            {self.backstory}

            Your personal goal is: {self.goal}
            """
        ).strip()

    def __repr__(self) -> str:
        return f"<Agent {self.name!r} role={self.role!r}>"

    def execute(self, prompt: str) -> str:
        """
        Execute a task prompt through the agent's ReAct loop.

        With memory enabled, each execution builds on the conversation
        from previous ones.
        """
        logger.info("Agent %s executing task", self.name)
        record("agent_execute", name=self.name, model=self.llm_model,
               prompt_preview=preview(prompt))
        result = self.react_agent.run(user_message=prompt)
        logger.info("Agent %s produced result: %s", self.name, result)
        return result

    def reset_memory(self) -> None:
        """Forget all prior task executions."""
        self.react_agent.reset()
        logger.debug("Agent %s memory reset", self.name)


__all__ = ["Agent"]
