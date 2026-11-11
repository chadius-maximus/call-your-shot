"""End-to-end flow test with no Hermes process: drive the hooks and the tool the way core does.

Run from the repository's parent directory:
    python -m pytest call-your-shot/tests -q
or directly:
    python call-your-shot/tests/test_flow.py
"""

import importlib.util
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_plugin():
    spec = importlib.util.spec_from_file_location(
        "call_your_shot_under_test", ROOT / "__init__.py", submodule_search_locations=[str(ROOT)])
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    module.store.DB_PATH = Path(tempfile.mkdtemp()) / "shots.db"
    module._turns.clear()
    return module


def run_turn(p, session, turn, shot=None, calls_before=0, calls_after=0, end="post_llm_call"):
    p._on_pre_llm_call(session_id=session, turn_id=turn, model="test-model", user_message="please   fix the build")
    for _ in range(calls_before):
        p._on_post_tool_call(tool_name="read_file", session_id=session, turn_id=turn)
    result = None
    if shot:
        result = json.loads(p.call_shot(shot, session_id=session))
        p._on_post_tool_call(tool_name="call_shot", session_id=session, turn_id=turn)
    for _ in range(calls_after):
        p._on_post_tool_call(tool_name="terminal", session_id=session, turn_id=turn)
    if end == "post_llm_call":
        p._on_post_llm_call(session_id=session, turn_id=turn)
    elif end == "interrupted":
        p._on_session_end(session_id=session, completed=False, interrupted=True)
    return result


def test_shot_is_measured_from_the_call_and_scored():
    p = load_plugin()
    result = run_turn(p, "s1", "t1", {"task": "fix bug", "tool_calls": 4, "minutes": 1},
                      calls_before=2, calls_after=8)
    assert result["status"] == "shot called"
    assert "2 tool call(s) already ran" in result["note"]
    row = p.store.recent(1)[0]
    assert row["status"] == "settled"
    assert row["actual_calls"] == 8  # the 2 calls before the shot and the shot itself are not counted
    assert row["pred_calls"] == 4
    assert 0 <= row["score"] <= 100


def test_a_shot_cannot_be_changed_in_the_same_turn():
    p = load_plugin()
    p._on_pre_llm_call(session_id="s1", turn_id="t1")
    first = json.loads(p.call_shot({"task": "a", "tool_calls": 3, "minutes": 1}, session_id="s1"))
    second = json.loads(p.call_shot({"task": "a", "tool_calls": 30, "minutes": 9}, session_id="s1"))
    assert first["status"] == "shot called"
    assert "cannot be changed" in second["error"]


def test_bad_arguments_return_errors_not_exceptions():
    p = load_plugin()
    for args in ({}, {"task": "x"}, {"task": "x", "tool_calls": "many", "minutes": 1},
                 {"task": "x", "tool_calls": -1, "minutes": 1}, {"task": "x", "tool_calls": 1, "minutes": 0}):
        assert "error" in json.loads(p.call_shot(args, session_id="s1"))
    assert p.store.recent(5) == []


def test_interrupted_turn_is_void_and_not_scored():
    p = load_plugin()
    run_turn(p, "s1", "t1", {"task": "x", "tool_calls": 2, "minutes": 1}, calls_after=1, end="interrupted")
    assert p.store.recent(5) == []
    assert p.store.record()["settled"] == 0


def test_record_reports_bias_and_calibrates_the_next_shot():
    p = load_plugin()
    for i in range(3):
        run_turn(p, "s1", f"t{i}", {"task": f"job {i}", "tool_calls": 2, "minutes": 0.01}, calls_after=6)
    rec = p.store.record()
    assert rec["settled"] == 3
    assert rec["calls_bias"] == 3.0
    result = run_turn(p, "s1", "t9", {"task": "next", "tool_calls": 4, "minutes": 0.01}, calls_after=4)
    assert "underestimate" in result["your_record"]["tool_calls"]
    assert result["your_record"]["calibrated_for_this_shot"]["tool_calls"] == 12


def test_next_turn_is_told_how_the_last_shot_went_once():
    p = load_plugin()
    run_turn(p, "s1", "t1", {"task": "ship it", "tool_calls": 3, "minutes": 0.01}, calls_after=3)
    told = p._on_pre_llm_call(session_id="s1", turn_id="t2")
    assert "ship it" in told["context"] and "predicted 3 tool calls" in told["context"]
    assert p._on_pre_llm_call(session_id="s1", turn_id="t3") is None


def test_uncalled_multi_step_turns_are_counted_and_scoreboard_renders():
    p = load_plugin()
    run_turn(p, "s1", "t1", calls_before=5)
    run_turn(p, "s1", "t2", {"task": "a task with a rather long label that gets cut", "tool_calls": 5,
                             "minutes": 0.01}, calls_after=5)
    board = p.render_scoreboard()
    assert "1 multi-step turn(s) ran with no shot called" in board
    assert "5 -> 5" in board


def test_task_defaults_to_the_user_message_and_plumbing_is_not_counted():
    p = load_plugin()
    p._on_pre_llm_call(session_id="s1", turn_id="t1", user_message="please   fix the build")
    assert json.loads(p.call_shot({"tool_calls": 2, "minutes": 0.01}, session_id="s1"))["status"] == "shot called"
    for name in ("tool_search", "tool_describe", "tool_call", "terminal", "patch"):
        p._on_post_tool_call(tool_name=name, session_id="s1", turn_id="t1")
    p._on_post_llm_call(session_id="s1", turn_id="t1")
    row = p.store.recent(1)[0]
    assert row["task"] == "please fix the build"
    assert row["actual_calls"] == 2


def test_perfect_and_poor_calls_score_as_expected():
    p = load_plugin()
    assert p.store.score_shot(5, 5, 60, 60) == 100
    assert p.store.score_shot(2, 8, 60, 240) == 25
    assert p.store.score_shot(0, 0, 30, 30) == 100


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS {name}")
            except Exception as exc:  # report every failure, then exit non-zero
                failures += 1
                print(f"FAIL {name}: {exc!r}")
    sys.exit(1 if failures else 0)
