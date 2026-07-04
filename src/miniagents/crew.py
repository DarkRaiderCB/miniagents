"""Crew: collects tasks, resolves their dependency DAG, and runs them in order."""

import logging
import time
from collections import deque
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, Union

from .exceptions import CircularDependencyError
from .tracing import Tracer, preview

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


class RunnableProtocol(Protocol):
    """Structural interface any node must satisfy to be managed by a Crew."""

    name: str
    dependencies: List["RunnableProtocol"]
    dependents: List["RunnableProtocol"]

    def run(self) -> str:
        ...


class Crew:
    """
    Manages a collection of tasks and executes them in dependency order.

    Used as a context manager so tasks auto-register on construction:

        with Crew() as crew:
            research = Task(description=..., agent=researcher)
            write = Task(description=..., agent=writer)
            research >> write
            results = crew.run_all()

    Every run_all() produces an execution trace (crew.last_trace) with a
    unique run_id covering every task, LLM call, and tool call.

    Note: the active-crew pointer is process-global, so crews should not be
    built concurrently from multiple threads.
    """

    _active: Optional["Crew"] = None

    def __init__(self) -> None:
        self._nodes: List[RunnableProtocol] = []
        self.last_trace: Optional[Tracer] = None

    def __enter__(self) -> "Crew":
        Crew._active = self
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        Crew._active = None

    @property
    def tasks(self) -> List[RunnableProtocol]:
        return list(self._nodes)

    @classmethod
    def register(cls, node: RunnableProtocol) -> None:
        """Register a task with the currently active crew context, if any."""
        if cls._active is None:
            logger.warning(
                "No active Crew context to register %r",
                getattr(node, "name", node),
            )
            return
        cls._active.add(node)

    def add(self, node: RunnableProtocol) -> None:
        """Add a task to this crew (idempotent)."""
        if not hasattr(node, "name") or not callable(getattr(node, "run", None)):
            raise TypeError("Task must have a 'name' attribute and a 'run()' method")
        if node in self._nodes:
            logger.debug("Task %s already in crew; skipping", node.name)
            return
        if any(existing.name == node.name for existing in self._nodes):
            raise ValueError(
                f"Duplicate task name {node.name!r} in crew; task names must be unique"
            )
        self._nodes.append(node)
        logger.debug("Task %s added to crew", node.name)

    def topological_sort(self) -> List[RunnableProtocol]:
        """
        Return this crew's tasks in dependency order (Kahn's algorithm).

        Only dependencies between tasks registered in this crew are
        considered; edges to outside tasks are ignored rather than
        deadlocking the sort.

        Raises:
            CircularDependencyError: if the dependency graph has a cycle.
        """
        members = set(id(n) for n in self._nodes)
        in_degree = {
            id(n): sum(1 for d in n.dependencies if id(d) in members)
            for n in self._nodes
        }
        by_id = {id(n): n for n in self._nodes}

        queue = deque(nid for nid, deg in in_degree.items() if deg == 0)
        sorted_list: List[RunnableProtocol] = []

        while queue:
            node = by_id[queue.popleft()]
            sorted_list.append(node)
            for dep in node.dependents:
                if id(dep) not in members:
                    continue
                in_degree[id(dep)] -= 1
                if in_degree[id(dep)] == 0:
                    queue.append(id(dep))

        if len(sorted_list) != len(self._nodes):
            unresolved = [n.name for n in self._nodes if n not in sorted_list]
            raise CircularDependencyError(
                f"Circular dependency detected among tasks: {unresolved}"
            )
        return sorted_list

    def plot(self, fmt: str = "png") -> Any:
        """
        Build a Graphviz Digraph of the task dependency graph.

        Requires the optional 'graphviz' package (pip install graphviz).
        """
        try:
            from graphviz import Digraph
        except ImportError as e:
            raise ImportError(
                "Crew.plot() requires the 'graphviz' package. "
                "Install it with: pip install graphviz"
            ) from e

        dot = Digraph(format=fmt)
        for node in self._nodes:
            dot.node(node.name)
            for prereq in node.dependencies:
                dot.edge(prereq.name, node.name)
        logger.debug("Generated dependency graph for %d tasks", len(self._nodes))
        return dot

    def run_all(
        self,
        tracer: Optional[Tracer] = None,
        save_trace_to: Optional[Union[str, Path]] = None,
    ) -> Dict[str, str]:
        """
        Execute each task in dependency order.

        Args:
            tracer: Optional pre-built Tracer (e.g. to share a run_id);
                a fresh one is created by default.
            save_trace_to: If given, write the run's trace there as JSONL.

        Returns:
            Mapping of task name to its output, in execution order.
            The full execution trace is available as `crew.last_trace`.
        """
        run_tracer = tracer or Tracer()
        self.last_trace = run_tracer
        results: Dict[str, str] = {}
        ordered = self.topological_sort()

        logger.info(
            "Crew run %s started: %d task(s): %s",
            run_tracer.run_id, len(ordered), [n.name for n in ordered],
        )
        with run_tracer:
            run_tracer.record("crew_start", tasks=[n.name for n in ordered])
            try:
                for node in ordered:
                    logger.info("[run %s] Running task %s", run_tracer.run_id, node.name)
                    run_tracer.record("task_start", name=node.name)
                    started = time.perf_counter()
                    try:
                        results[node.name] = node.run()
                    except Exception as e:
                        run_tracer.record("task_error", name=node.name, error=str(e))
                        logger.exception(
                            "[run %s] Task %s failed", run_tracer.run_id, node.name
                        )
                        raise
                    elapsed_ms = round((time.perf_counter() - started) * 1000, 1)
                    run_tracer.record(
                        "task_end",
                        name=node.name,
                        elapsed_ms=elapsed_ms,
                        output_preview=preview(results[node.name]),
                    )
                    logger.info(
                        "[run %s] Task %s finished in %.1fms",
                        run_tracer.run_id, node.name, elapsed_ms,
                    )
            finally:
                run_tracer.record("crew_end", completed=len(results), total=len(ordered))
                if save_trace_to is not None:
                    run_tracer.save(save_trace_to)

        logger.info(
            "Crew run %s finished: %d/%d task(s) completed",
            run_tracer.run_id, len(results), len(ordered),
        )
        return results


__all__ = ["Crew", "RunnableProtocol"]
