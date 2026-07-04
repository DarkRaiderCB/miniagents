"""miniagents — a minimal multi-agent LLM framework.

Persona-defined agents run a prompt-engineered ReAct loop with Python
functions as tools, and execute Tasks composed into a dependency DAG.
"""

from .agent import Agent
from .crew import Crew, RunnableProtocol
from .exceptions import (
    CircularDependencyError,
    ConfigurationError,
    FrameworkError,
    ToolArgumentError,
    ToolError,
    ToolNotFoundError,
)
from .log import configure_logging
from .react_agent import DEFAULT_MAX_ROUNDS, DEFAULT_MODEL, ReactAgent
from .task import Task
from .tool import Tool, tool
from .tracing import TraceEvent, Tracer, current_tracer

__version__ = "0.2.0"

__all__ = [
    "Agent",
    "Task",
    "Crew",
    "RunnableProtocol",
    "ReactAgent",
    "Tool",
    "tool",
    "Tracer",
    "TraceEvent",
    "current_tracer",
    "configure_logging",
    "FrameworkError",
    "ConfigurationError",
    "ToolError",
    "ToolNotFoundError",
    "ToolArgumentError",
    "CircularDependencyError",
    "DEFAULT_MODEL",
    "DEFAULT_MAX_ROUNDS",
    "__version__",
]
