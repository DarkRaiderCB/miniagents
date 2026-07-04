"""ReAct (Thought -> Action -> Observation) agent loop over a chat model."""

import json
import logging
import os
import time
from typing import Any, Dict, List, Optional, Union

from dotenv import load_dotenv

from .chat_client import ChatClientWrapper
from .tool import Tool
from .tracing import preview, record
from .utils import MessageHistory, PinnedMessageHistory, TagParser, create_message

load_dotenv()

# Optional global system prompt prefix, settable via environment.
BASE_SYSTEM_PROMPT = os.getenv("BASE_SYSTEM_PROMPT", "")

DEFAULT_MODEL = "llama-3.3-70b-versatile"
DEFAULT_MAX_ROUNDS = 10

REACT_SYSTEM_PROMPT = """
You operate by running a loop with the following steps: Thought, Action, Observation.
You are provided with function signatures within <tools></tools> as a JSON array.
You may call one or more functions to assist with the user query. Don't make assumptions
about what values to plug into functions - follow their parameter schemas exactly.

For each function call, return a JSON object with the function name, arguments, and id
inside <tool_call> tags, for example:

<tool_call>
{"name": <function-name>, "arguments": <args-dict>, "id": <monotonic-id>}
</tool_call>

Here are the available tools / actions (as a JSON array):

<tools>
%s
</tools>

Example session:

<question>What's the temperature in Madrid?</question>
<thought>I need to get the weather</thought>
<tool_call>{"name":"get_current_weather","arguments":{"location":"Madrid"},"id":0}</tool_call>
<observation>{0: {"temperature":25}}</observation>
<response>The temperature is 25°C</response>

Constraints:
- If your answer does not require a tool, just emit it inside <response>...</response>.
- If an observation reports an error, correct your tool call and try again.
"""

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


class ReactAgent:
    """
    Drives a prompt-engineered ReAct loop: the model emits <thought>,
    <tool_call>, and <response> tags; tool calls are executed locally and
    their results are fed back as observations until the model responds.
    """

    def __init__(
        self,
        tools: Optional[Union[Tool, List[Tool]]] = None,
        model: str = DEFAULT_MODEL,
        system_prompt: str = BASE_SYSTEM_PROMPT,
        memory: bool = False,
        history_max_size: Optional[int] = None,
    ) -> None:
        """
        Args:
            memory: When True, conversation history persists across run()
                calls so the agent remembers earlier exchanges. Call reset()
                to clear it.
            history_max_size: Optional cap on retained messages when memory
                is enabled; the system prompt is always kept (pinned).
        """
        self.chat = ChatClientWrapper(model_name=model)
        self.memory = memory
        self.history_max_size = history_max_size
        self._history: Optional[MessageHistory] = None

        if tools is None:
            tools = []
        self.tools: List[Tool] = tools if isinstance(tools, list) else [tools]
        self.tools_dict: Dict[str, Tool] = {t.name: t for t in self.tools}

        tool_sigs = [t.signature.to_dict() for t in self.tools]
        tools_json = json.dumps(tool_sigs, indent=2)
        logger.debug("Registered tools for model %s:\n%s", model, tools_json)

        self.system_prompt = system_prompt
        self.react_prompt = REACT_SYSTEM_PROMPT % tools_json

        self._parse_response = TagParser("response")
        self._parse_thought = TagParser("thought")
        self._parse_tool = TagParser("tool_call")

    def reset(self) -> None:
        """Clear any persisted conversation memory."""
        self._history = None

    def _start_history(self) -> MessageHistory:
        if self.memory:
            history: MessageHistory = PinnedMessageHistory(max_size=self.history_max_size)
        else:
            history = MessageHistory()
        history.append(
            create_message("system", f"{self.system_prompt}\n{self.react_prompt}".strip())
        )
        return history

    def _call_model(self, history: MessageHistory, round_num: int) -> str:
        started = time.perf_counter()
        completion = self.chat.generate_response(history.all())
        elapsed_ms = round((time.perf_counter() - started) * 1000, 1)
        logger.debug(
            "LLM call round=%d model=%s messages=%d elapsed=%.1fms",
            round_num, self.chat.model, len(history), elapsed_ms,
        )
        record(
            "llm_call",
            name=self.chat.model,
            round=round_num,
            messages=len(history),
            elapsed_ms=elapsed_ms,
            completion_preview=preview(completion),
        )
        return completion

    def _run_tool_calls(self, raw_calls: List[str]) -> Dict[Any, Any]:
        """
        Execute the JSON tool-call payloads extracted from <tool_call> tags.

        Never raises on a bad call: malformed JSON, unknown tools, and tool
        failures are returned as error strings in the observations so the
        model can see what went wrong and retry.
        """
        observations: Dict[Any, Any] = {}
        for position, raw in enumerate(raw_calls):
            call_id: Any = position
            try:
                call = json.loads(raw)
                if not isinstance(call, dict):
                    raise ValueError(f"expected a JSON object, got {type(call).__name__}")
            except (json.JSONDecodeError, ValueError) as e:
                observations[call_id] = f"Error: invalid tool call payload ({e}): {raw!r}"
                record("tool_error", name="<unparseable>", error=str(e), raw=preview(raw))
                continue

            call_id = call.get("id", position)
            name = call.get("name")
            arguments = call.get("arguments") or {}

            if name not in self.tools_dict:
                observations[call_id] = (
                    f"Error: tool {name!r} not found. "
                    f"Available tools: {sorted(self.tools_dict)}"
                )
                record("tool_error", name=str(name), error="tool not found")
                continue
            if not isinstance(arguments, dict):
                observations[call_id] = (
                    f"Error: 'arguments' for tool {name!r} must be a JSON object, "
                    f"got: {arguments!r}"
                )
                record("tool_error", name=name, error="arguments not a JSON object")
                continue

            logger.info("Invoking tool %s with arguments %s", name, arguments)
            record("tool_call", name=name, arguments=arguments)
            started = time.perf_counter()
            try:
                result = self.tools_dict[name](**arguments)
            except Exception as e:
                logger.warning("Tool %s failed: %s", name, e)
                observations[call_id] = f"Error: tool {name!r} failed: {e}"
                record("tool_error", name=name, error=str(e))
                continue

            elapsed_ms = round((time.perf_counter() - started) * 1000, 1)
            logger.info("Result from tool %s (%.1fms): %s", name, elapsed_ms, result)
            record(
                "tool_result",
                name=name,
                elapsed_ms=elapsed_ms,
                result_preview=preview(result),
            )
            observations[call_id] = result
        return observations

    def run(self, user_message: str, max_rounds: int = DEFAULT_MAX_ROUNDS) -> str:
        """
        Run the ReAct loop for up to max_rounds model turns and return the
        content of the model's final <response> tag.

        With memory enabled, the conversation (including this exchange) is
        retained and reused by subsequent run() calls.
        """
        if max_rounds < 1:
            raise ValueError("max_rounds must be at least 1")

        if self.memory and self._history is not None:
            history = self._history
        else:
            history = self._start_history()
        if self.memory:
            self._history = history

        history.append(create_message("user", user_message, tag="question"))

        for round_num in range(1, max_rounds + 1):
            completion = self._call_model(history, round_num)

            response = self._parse_response.parse(completion)
            if response.found:
                history.append(create_message("assistant", completion))
                record("react_response", rounds=round_num, response_preview=preview(response.items[0]))
                return response.items[0]

            history.append(create_message("assistant", completion))

            thought = self._parse_thought.parse(completion)
            if thought.found:
                logger.debug("Round %d thought: %s", round_num, thought.items[0])

            tool_calls = self._parse_tool.parse(completion)
            if tool_calls.found:
                observations = self._run_tool_calls(tool_calls.items)
                history.append(
                    create_message(
                        "user",
                        json.dumps(observations, default=str),
                        tag="observation",
                    )
                )
                continue

            # Neither a response nor a tool call: nudge the model back on format.
            logger.debug("Round %d produced no response or tool call; nudging", round_num)
            record("react_nudge", round=round_num)
            history.append(
                create_message(
                    "user",
                    "Reminder: either call a tool inside <tool_call> tags or give "
                    "your final answer inside <response></response> tags.",
                )
            )

        # Out of rounds: ask once for a direct final answer.
        logger.warning("Max rounds (%d) reached without a <response>", max_rounds)
        record("react_max_rounds", max_rounds=max_rounds)
        history.append(
            create_message(
                "user",
                "You have run out of tool-use rounds. Give your best final answer "
                "now inside <response></response> tags.",
            )
        )
        completion = self._call_model(history, max_rounds + 1)
        history.append(create_message("assistant", completion))
        response = self._parse_response.parse(completion)
        return response.items[0] if response.found else completion.strip()


__all__ = ["ReactAgent", "DEFAULT_MODEL", "DEFAULT_MAX_ROUNDS"]
