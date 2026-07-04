"""Tests for the Agent/Task/Crew orchestration layer."""

import pytest

from miniagents import Agent, CircularDependencyError, Crew, Task


def make_agent(role="Writer", **kwargs) -> Agent:
    return Agent(role=role, goal=f"{role} goal", backstory=f"{role} backstory", **kwargs)


class TestAgent:
    def test_persona_compiled_into_system_prompt(self):
        agent = Agent(role="Weather Assistant", goal="Answer weather questions.",
                      backstory="You must use tools.")
        prompt = agent.react_agent.system_prompt
        assert "You are Weather Assistant." in prompt
        assert "Your personal goal is: Answer weather questions." in prompt
        assert "You must use tools." in prompt

    def test_name_defaults_to_role(self):
        assert make_agent(role="Editor").name == "Editor"

    def test_explicit_name(self):
        assert make_agent(name="Bob").name == "Bob"

    def test_empty_role_rejected(self):
        with pytest.raises(ValueError):
            Agent(role="  ", goal="g", backstory="b")

    def test_empty_goal_rejected(self):
        with pytest.raises(ValueError):
            Agent(role="X", goal="", backstory="b")

    def test_reset_memory_delegates(self):
        agent = make_agent(memory=True)
        agent.react_agent.chat.script = ["<response>x</response>"]
        agent.execute("remember")
        agent.reset_memory()
        assert agent.react_agent._history is None


class TestTask:
    def test_requires_description(self):
        with pytest.raises(ValueError):
            Task(description="  ", agent=make_agent())

    def test_requires_agent_instance(self):
        with pytest.raises(TypeError):
            Task(description="d", agent="not-an-agent")

    def test_auto_name_generation(self):
        t = Task(description="d", agent=make_agent(role="Solo"))
        assert t.name.startswith("Solo_task_")

    def test_self_dependency_rejected(self):
        t = Task(description="d", agent=make_agent())
        with pytest.raises(ValueError):
            t.add_dependency(t)

    def test_duplicate_edges_ignored(self):
        a = Task(description="d", agent=make_agent(), name="a")
        b = Task(description="d", agent=make_agent(), name="b")
        a >> b
        a >> b
        assert b.dependencies == [a]
        assert a.dependents == [b]


class TestCrew:
    def test_pipeline_passes_context_downstream(self):
        writer, editor = make_agent("Writer"), make_agent("Editor")
        with Crew() as crew:
            fact = Task(description="produce a fact", agent=writer, name="fact")
            check = Task(description="check the fact", agent=editor, name="check")
            fact >> check
            writer.react_agent.chat.script = ["<response>the sky is blue</response>"]
            editor.react_agent.chat.script = ["<response>confirmed</response>"]
            results = crew.run_all()

        assert results == {"fact": "the sky is blue", "check": "confirmed"}
        assert fact.output == "the sky is blue"
        prompt = editor.react_agent.chat.last_conversation[1]["content"]
        assert "the sky is blue" in prompt
        assert "'fact'" in prompt and "'Writer'" in prompt

    def test_one_agent_many_tasks(self):
        solo = make_agent("Solo")
        with Crew() as crew:
            one = Task(description="one", agent=solo, name="one")
            two = Task(description="two", agent=solo, name="two")
            one >> two
            solo.react_agent.chat.script = [
                "<response>out1</response>", "<response>out2</response>",
            ]
            assert crew.run_all() == {"one": "out1", "two": "out2"}

    def test_rerun_does_not_accumulate_stale_context(self):
        solo = make_agent("Solo")
        with Crew() as crew:
            one = Task(description="one", agent=solo, name="one")
            two = Task(description="two", agent=solo, name="two")
            one >> two
            solo.react_agent.chat.script = [
                "<response>a</response>", "<response>b</response>",
            ]
            crew.run_all()
            solo.react_agent.chat.script = [
                "<response>c</response>", "<response>d</response>",
            ]
            crew.run_all()
        prompt = solo.react_agent.chat.last_conversation[1]["content"]
        assert prompt.count("Output from upstream task") == 1

    def test_duplicate_task_names_rejected(self):
        solo = make_agent()
        with pytest.raises(ValueError, match="Duplicate task name"):
            with Crew():
                Task(description="d", agent=solo, name="dup")
                Task(description="d", agent=solo, name="dup")

    def test_cycle_detected(self):
        solo = make_agent()
        with Crew() as crew:
            a = Task(description="d", agent=solo, name="a")
            b = Task(description="d", agent=solo, name="b")
            a >> b
            b >> a
        with pytest.raises(CircularDependencyError):
            crew.topological_sort()

    def test_dependency_outside_crew_ignored(self):
        solo = make_agent()
        outside = Task(description="d", agent=solo, name="outside")  # no active crew
        with Crew() as crew:
            inside = Task(description="d", agent=solo, name="inside")
            inside.add_dependency(outside)
            solo.react_agent.chat.script = ["<response>ok</response>"]
            assert crew.run_all() == {"inside": "ok"}

    def test_independent_tasks_run_in_registration_order(self):
        solo = make_agent()
        with Crew() as crew:
            Task(description="d1", agent=solo, name="first")
            Task(description="d2", agent=solo, name="second")
            solo.react_agent.chat.script = [
                "<response>1</response>", "<response>2</response>",
            ]
            assert list(crew.run_all()) == ["first", "second"]

    def test_task_without_crew_context_warns_not_raises(self, caplog):
        solo = make_agent()
        t = Task(description="d", agent=solo, name="orphan")
        assert t.name == "orphan"  # constructed fine, just not registered

    def test_agent_memory_across_tasks_without_edge(self):
        pa = make_agent("PA", memory=True)
        with Crew() as crew:
            Task(description="Favorite city is Madrid; acknowledge.", agent=pa, name="m1")
            Task(description="What is the favorite city?", agent=pa, name="m2")
            pa.react_agent.chat.script = [
                "<response>noted</response>", "<response>Madrid</response>",
            ]
            results = crew.run_all()
        assert results["m2"] == "Madrid"
        convo = pa.react_agent.chat.last_conversation
        assert any("Madrid; acknowledge" in m["content"] for m in convo)
