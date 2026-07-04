"""Exception hierarchy for the multi-agent framework.

All framework-raised errors derive from FrameworkError so callers can
catch everything from this library with a single except clause.
"""


class FrameworkError(Exception):
    """Base class for all errors raised by this framework."""


class ConfigurationError(FrameworkError):
    """Raised when required configuration (e.g. API keys) is missing or invalid."""


class ToolError(FrameworkError):
    """Base class for tool-related errors."""


class ToolNotFoundError(ToolError):
    """Raised when the model requests a tool that is not registered."""


class ToolArgumentError(ToolError):
    """Raised when tool arguments are missing or cannot be coerced to the expected type."""


class CircularDependencyError(FrameworkError):
    """Raised when the agent dependency graph contains a cycle."""


__all__ = [
    "FrameworkError",
    "ConfigurationError",
    "ToolError",
    "ToolNotFoundError",
    "ToolArgumentError",
    "CircularDependencyError",
]
