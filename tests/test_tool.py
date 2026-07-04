"""Tests for tool introspection, validation, and coercion."""

import json

import pytest

from miniagents import Tool, ToolArgumentError, tool


@tool
def sample(a: int, b: str = "default", flag: bool = False) -> str:
    """Sample tool for signature tests."""
    return f"{a}-{b}-{flag}"


class TestFunctionSignature:
    def test_introspects_name_and_description(self):
        assert sample.name == "sample"
        assert "Sample tool" in sample.signature.description

    def test_parameters_and_defaults(self):
        params = sample.signature.parameters
        assert params["a"] == {"type": "int"}
        assert params["b"] == {"type": "str", "default": "default"}
        assert params["flag"] == {"type": "bool", "default": False}

    def test_return_type(self):
        assert sample.signature.return_type == "str"

    def test_info_is_valid_json(self):
        payload = json.loads(sample.info())
        assert payload["name"] == "sample"
        assert "parameters" in payload


class TestValidation:
    def test_happy_path(self):
        assert sample(a=1, b="x", flag=True) == "1-x-True"

    def test_defaults_applied(self):
        assert sample(a=1) == "1-default-False"

    def test_type_coercion(self):
        assert sample(a="42") == "42-default-False"

    def test_bool_string_false_coerces_to_false(self):
        assert sample(a=1, flag="false") == "1-default-False"
        assert sample(a=1, flag="no") == "1-default-False"

    def test_bool_string_true_coerces_to_true(self):
        assert sample(a=1, flag="true") == "1-default-True"
        assert sample(a=1, flag="YES") == "1-default-True"

    def test_bad_bool_rejected(self):
        with pytest.raises(ToolArgumentError, match="must be a boolean"):
            sample(a=1, flag="maybe")

    def test_uncoercible_value_rejected(self):
        with pytest.raises(ToolArgumentError, match="must be of type int"):
            sample(a="not-a-number")

    def test_missing_required_argument(self):
        with pytest.raises(ToolArgumentError, match="Missing required argument 'a'"):
            sample(b="x")

    def test_unexpected_argument_rejected(self):
        with pytest.raises(ToolArgumentError, match="Unexpected argument"):
            sample(a=1, bogus=99)


class TestDecorator:
    def test_decorator_returns_tool_instance(self):
        assert isinstance(sample, Tool)

    def test_repr(self):
        assert repr(sample) == "<Tool 'sample'>"
