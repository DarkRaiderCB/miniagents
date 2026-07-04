"""Tool abstraction: wraps plain Python functions for LLM-driven invocation."""

import inspect
import json
import logging
from typing import Any, Callable, Dict, Optional, Type

from .exceptions import ToolArgumentError

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

type_aliases: Dict[str, Type[Any]] = {
    "int": int,
    "float": float,
    "str": str,
    "bool": bool,
    "list": list,
    "dict": dict,
}

_FALSY_STRINGS = {"false", "no", "0", "off", ""}
_TRUTHY_STRINGS = {"true", "yes", "1", "on"}


class FunctionSignature:
    """
    Models a function's name, documentation, and parameters so the LLM can be
    shown a JSON schema and arguments can be validated before invocation.
    """

    def __init__(self, func: Callable) -> None:
        sig = inspect.signature(func)
        self.name: str = func.__name__
        self.description: Optional[str] = (func.__doc__ or "").strip() or None
        self.parameters: Dict[str, Dict[str, Any]] = {}

        for param_name, param in sig.parameters.items():
            if param_name == "self":
                continue
            info: Dict[str, Any] = {}
            if param.annotation is not inspect.Parameter.empty:
                annot = param.annotation
                info["type"] = getattr(annot, "__name__", str(annot))
            if param.default is not inspect.Parameter.empty:
                info["default"] = param.default
            self.parameters[param_name] = info

        self.return_type: Optional[str] = None
        if sig.return_annotation is not inspect.Signature.empty:
            ret = sig.return_annotation
            self.return_type = getattr(ret, "__name__", str(ret))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
            "return_type": self.return_type,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict())


class ArgumentValidator:
    """
    Validates and coerces arguments against a FunctionSignature.
    """

    def __init__(self, type_map: Optional[Dict[str, Type[Any]]] = None) -> None:
        self._type_map = type_map or type_aliases

    def validate(self, args: Dict[str, Any], signature: FunctionSignature) -> Dict[str, Any]:
        unexpected = set(args) - set(signature.parameters)
        if unexpected:
            raise ToolArgumentError(
                f"Unexpected argument(s) for tool '{signature.name}': "
                f"{sorted(unexpected)}. Expected: {sorted(signature.parameters)}"
            )

        validated: Dict[str, Any] = {}
        for name, meta in signature.parameters.items():
            if name in args:
                validated[name] = self._coerce(signature.name, name, args[name], meta)
            elif "default" in meta:
                validated[name] = meta["default"]
            else:
                raise ToolArgumentError(
                    f"Missing required argument '{name}' for tool '{signature.name}'"
                )
        return validated

    def _coerce(self, tool_name: str, arg_name: str, value: Any, meta: Dict[str, Any]) -> Any:
        expected = meta.get("type")
        if not expected or expected not in self._type_map:
            return value

        target_type = self._type_map[expected]
        if isinstance(value, target_type):
            return value

        # bool("false") is True, so string-to-bool needs explicit handling.
        if target_type is bool and isinstance(value, str):
            lowered = value.strip().lower()
            if lowered in _TRUTHY_STRINGS:
                return True
            if lowered in _FALSY_STRINGS:
                return False
            raise ToolArgumentError(
                f"Argument '{arg_name}' of tool '{tool_name}' must be a boolean, "
                f"got {value!r}"
            )

        try:
            return target_type(value)
        except (TypeError, ValueError) as e:
            raise ToolArgumentError(
                f"Argument '{arg_name}' of tool '{tool_name}' must be of type "
                f"{expected}, got {type(value).__name__} ({value!r})"
            ) from e


class Tool:
    """
    Wraps a callable for standardized signature introspection and invocation.
    """

    def __init__(
        self,
        function: Callable,
        signature: FunctionSignature,
        validator: ArgumentValidator,
    ) -> None:
        self._fn = function
        self.signature = signature
        self._validator = validator

    @property
    def name(self) -> str:
        return self.signature.name

    def __repr__(self) -> str:
        return f"<Tool {self.name!r}>"

    def __call__(self, **kwargs: Any) -> Any:
        args = self._validator.validate(kwargs, self.signature)
        logger.debug("Invoking tool '%s' with arguments: %s", self.name, args)
        return self._fn(**args)

    def info(self) -> str:
        """Returns the JSON representation of this tool's signature."""
        return self.signature.to_json()


def tool(func: Callable) -> Tool:
    """
    Decorator that converts a function into a Tool with introspected signature.
    """
    return Tool(
        function=func,
        signature=FunctionSignature(func),
        validator=ArgumentValidator(),
    )


__all__ = [
    "FunctionSignature",
    "ArgumentValidator",
    "Tool",
    "tool",
]
