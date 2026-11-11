"""call-your-shot: the agent predicts its effort, hooks measure what really happened.

Flow: the agent calls ``call_shot`` with an estimate of the tool calls and minutes a task will
take. ``post_tool_call`` counts the calls that follow, ``post_llm_call`` settles the shot when the
turn ends, and the result of the next ``call_shot`` tells the agent how its past estimates held up.
All state is one local SQLite file. No network, no model calls.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Optional

from . import store

logger = logging.getLogger(__name__)

TOOL_NAME = "call_shot"
UNCALLED_TURN_MIN_CALLS = 2
# Tool-loading plumbing, not work: a tool_call wrapper only reaches the hook when it was rejected.
META_TOOLS = frozenset({"tool_search", "tool_describe", "tool_call"})
MAX_TOOL_CALLS = 500
MAX_MINUTES = 600.0

NUDGE = (
    "Call your shot: before starting any task where you will edit files or expect 2 or more tool "
    "calls, call the call_shot tool first, once: call_shot(tool_calls=<integer>, minutes=<number>, "
    "task=<short label>), through tool_call if it is listed as a deferred tool. Give your honest "
    "estimate of the tool calls and wall-clock minutes still ahead. Hooks measure what actually "
    "happens and keep your record; the tool result reports how your past estimates compared, so "
    "use it to adjust. Skip it for simple answers, and never call it after the work is done."
)

CALL_SHOT_SCHEMA = {
    "name": TOOL_NAME,
    "description": (
        "Call your shot before a multi-step task: state how many more tool calls and how many "
        "wall-clock minutes you expect the rest of this turn to take. The estimate is locked once "
        "called; the plugin counts the real tool calls and time from this moment to the end of the "
        "turn and scores the estimate. The result includes your track record so far (for example "
        "'actual tool calls run 1.8x your estimates') and a calibrated figure. Call it once, at the "
        "start, for any task where you will edit files or expect 2 or more tool calls. Do not count this call itself."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "task": {"type": "string", "description": "Short label for the task, under 100 characters. Defaults to the start of the user's message."},
            "tool_calls": {"type": "integer", "minimum": 0,
                           "description": "Tool calls you expect to make from now until the turn ends."},
            "minutes": {"type": "number", "exclusiveMinimum": 0,
                        "description": "Wall-clock minutes you expect from now until the turn ends. Fractions allowed (0.5 = 30 seconds)."},
        },
        "required": ["tool_calls", "minutes"],
    },
}


@dataclass
class _Turn:
    turn_id: str = ""
    model: str = ""
    label: str = ""
    calls: int = 0
    by_tool: dict = field(default_factory=dict)
    shot_id: Optional[int] = None
    shot_at: float = 0.0
    calls_at_shot: int = 0
    by_tool_at_shot: dict = field(default_factory=dict)


_turns: dict = {}
_turns_lock = threading.Lock()


def fmt_duration(seconds: float) -> str:
    seconds = max(0, round(float(seconds)))
    return f"{seconds // 60}m{seconds % 60:02d}s"


def _bias_phrase(bias: float, what: str) -> str:
    if bias > 1.15:
        return f"actual {what} came to {bias:.2f}x your estimates (you underestimate)"
    if bias < 0.87:
        return f"actual {what} came to {bias:.2f}x your estimates (you overestimate)"
    return f"your estimates are on target ({bias:.2f}x)"


def _turn_for(session_id: str, turn_id: str = "") -> _Turn:
    """Current turn state for a session; a new turn id starts a fresh count. Caller holds the lock."""
    turn = _turns.get(session_id)
    if turn is None or (turn_id and turn.turn_id and turn.turn_id != turn_id):
        turn = _Turn(turn_id=turn_id)
        _turns[session_id] = turn
    elif turn_id and not turn.turn_id:
        turn.turn_id = turn_id
    return turn


# ---------------------------------------------------------------- tool

def call_shot(args: dict, **kwargs) -> str:
    try:
        task = " ".join(str(args.get("task") or "").split())[:120]
        try:
            pred_calls = int(args.get("tool_calls"))
            minutes = float(args.get("minutes"))
        except (TypeError, ValueError):
            return json.dumps({"error": "tool_calls must be an integer and minutes a number"})
        if not 0 <= pred_calls <= MAX_TOOL_CALLS:
            return json.dumps({"error": f"tool_calls must be between 0 and {MAX_TOOL_CALLS}"})
        if not 0 < minutes <= MAX_MINUTES:
            return json.dumps({"error": f"minutes must be above 0 and at most {MAX_MINUTES:g}"})

        session_id = str(kwargs.get("session_id") or "")
        now = time.time()
        with _turns_lock:
            turn = _turn_for(session_id)
            task = task or turn.label or "untitled task"
            if turn.shot_id is not None:
                return json.dumps({
                    "error": "A shot is already called for this turn and cannot be changed.",
                    "shot_id": turn.shot_id,
                })
            turn.shot_id = store.open_shot(
                session_id=session_id, turn_id=turn.turn_id, model=turn.model, task=task,
                pred_calls=pred_calls, pred_seconds=minutes * 60, calls_before=turn.calls, now=now)
            turn.shot_at = now
            turn.calls_at_shot = turn.calls
            turn.by_tool_at_shot = dict(turn.by_tool)
            shot_id, late = turn.shot_id, turn.calls

        rec = store.record()
        result: dict[str, Any] = {
            "status": "shot called",
            "shot_id": shot_id,
            "predicted": {"tool_calls": pred_calls, "time": fmt_duration(minutes * 60)},
            "rules": "Locked. Tool calls and time are measured from now to the end of this turn.",
        }
        if late:
            result["note"] = f"Called after {late} tool call(s) already ran this turn; only the rest is measured."
        if rec.get("window", 0) >= store.MIN_SHOTS_FOR_RECORD:
            result["your_record"] = {
                "settled_shots": rec["settled"],
                "average_score": f"{rec['avg_score']}/100",
                "tool_calls": _bias_phrase(rec["calls_bias"], "tool calls"),
                "time": _bias_phrase(rec["time_bias"], "time"),
                "calibrated_for_this_shot": {
                    "tool_calls": max(0, round(pred_calls * rec["calls_bias"])),
                    "time": fmt_duration(minutes * 60 * rec["time_bias"]),
                },
            }
        else:
            result["your_record"] = (f"{rec.get('settled', 0)} settled shot(s) so far; a record appears "
                                     f"after {store.MIN_SHOTS_FOR_RECORD}.")
        return json.dumps(result)
    except Exception as exc:  # a tool handler never raises
        logger.warning("call_shot failed: %s", exc)
        return json.dumps({"error": f"call_shot failed: {exc}"})


# ---------------------------------------------------------------- hooks

def _on_pre_llm_call(session_id: str = "", turn_id: str = "", model: str = "", user_message=None, **kwargs):
    """Turn start: begin a fresh count and tell the agent how its last shot in this session went."""
    try:
        session_id = str(session_id or "")
        with _turns_lock:
            label = " ".join(user_message.split())[:60] if isinstance(user_message, str) else ""
            _turns[session_id] = _Turn(turn_id=str(turn_id or ""), model=str(model or ""), label=label)
        last = store.take_unnotified(session_id)
        if last:
            return {"context": (
                f"[call-your-shot] Your last shot settled: \"{last['task']}\" predicted "
                f"{last['pred_calls']} tool calls / {fmt_duration(last['pred_seconds'])}, actual "
                f"{last['actual_calls']} / {fmt_duration(last['actual_seconds'])}. Score {last['score']}/100."
            )}
    except Exception as exc:
        logger.warning("call-your-shot pre_llm_call failed: %s", exc)
    return None


def _on_post_tool_call(tool_name: str = "", session_id: str = "", turn_id: str = "", **kwargs) -> None:
    """Count every tool call of the turn except the shot itself and tool-loading plumbing."""
    try:
        if tool_name == TOOL_NAME or tool_name in META_TOOLS:
            return
        with _turns_lock:
            turn = _turn_for(str(session_id or ""), str(turn_id or ""))
            turn.calls += 1
            turn.by_tool[tool_name] = turn.by_tool.get(tool_name, 0) + 1
    except Exception as exc:
        logger.warning("call-your-shot post_tool_call failed: %s", exc)


def _close_turn(session_id: str, *, scored: bool) -> None:
    with _turns_lock:
        turn = _turns.pop(str(session_id or ""), None)
    if turn is None:
        return
    if turn.shot_id is None:
        if scored and turn.calls >= UNCALLED_TURN_MIN_CALLS:
            store.bump("uncalled_turns")
        return
    if not scored:
        store.void_shot(turn.shot_id)
        return
    by_tool = {name: n - turn.by_tool_at_shot.get(name, 0) for name, n in turn.by_tool.items()
               if n - turn.by_tool_at_shot.get(name, 0) > 0}
    store.settle_shot(turn.shot_id, actual_calls=turn.calls - turn.calls_at_shot,
                      actual_seconds=time.time() - turn.shot_at, by_tool=by_tool)


def _on_post_llm_call(session_id: str = "", **kwargs) -> None:
    """Turn end: settle the open shot against the measured numbers."""
    try:
        _close_turn(session_id, scored=True)
    except Exception as exc:
        logger.warning("call-your-shot post_llm_call failed: %s", exc)


def _on_session_end(session_id: str = "", completed: bool = True, interrupted: bool = False,
                    failed: bool = False, **kwargs) -> None:
    """Covers turns that end without post_llm_call. An interrupted or failed turn is not scored."""
    try:
        _close_turn(session_id, scored=bool(completed) and not interrupted and not failed)
    except Exception as exc:
        logger.warning("call-your-shot on_session_end failed: %s", exc)


# ---------------------------------------------------------------- scoreboard

def render_scoreboard(limit: int = 10) -> str:
    rec = store.record()
    rows = store.recent(limit)
    lines = ["CALL YOUR SHOT"]
    if not rec.get("window"):
        lines.append("No settled shots yet. The agent calls call_shot before multi-step work;")
        lines.append("the shot settles when that turn ends.")
    else:
        lines.append(f"{rec['settled']} shot(s) settled   average score {rec['avg_score']}/100"
                     f"   (record over the last {rec['window']})")
        lines.append(f"  tool calls: {_bias_phrase(rec['calls_bias'], 'tool calls')}")
        lines.append(f"  time:       {_bias_phrase(rec['time_bias'], 'time')}")
    if rec.get("uncalled_turns"):
        lines.append(f"  {rec['uncalled_turns']} multi-step turn(s) ran with no shot called")
    if rows:
        lines.append("")
        lines.append(f"{'#':>3}  {'when':<11}  {'task':<34}  {'calls':<9}  {'time':<16}  score")
        for r in rows:
            when = time.strftime("%m-%d %H:%M", time.localtime(r["called_at"]))
            task = r["task"] if len(r["task"]) <= 34 else r["task"][:31] + "..."
            if r["status"] == "settled":
                calls = f"{r['pred_calls']} -> {r['actual_calls']}"
                dur = f"{fmt_duration(r['pred_seconds'])} -> {fmt_duration(r['actual_seconds'])}"
                score = f"{r['score']:>5}"
            else:
                calls, dur, score = f"{r['pred_calls']} -> ?", f"{fmt_duration(r['pred_seconds'])} -> ?", " open"
            lines.append(f"{r['id']:>3}  {when:<11}  {task:<34}  {calls:<9}  {dur:<16}  {score}")
        lines.append("")
        lines.append("calls and time read predicted -> actual; score is 100 for a perfect call")
    return "\n".join(lines)


def _slash_shots(raw_args: str = "") -> str:
    try:
        arg = (raw_args or "").strip()
        limit = int(arg) if arg.isdigit() else 10
        return render_scoreboard(limit)
    except Exception as exc:
        return f"call-your-shot: could not read the scoreboard ({exc})"


def _cli_setup(parser) -> None:
    parser.add_argument("--limit", type=int, default=10, help="How many recent shots to list (default 10)")
    parser.add_argument("--json", action="store_true", help="Print the record and recent shots as JSON")
    parser.add_argument("--reset", action="store_true", help="Delete every recorded shot (asks first)")
    parser.add_argument("--yes", action="store_true", help="With --reset: do not ask for confirmation")


def _cli_handler(args) -> int:
    if getattr(args, "reset", False):
        if not getattr(args, "yes", False):
            try:
                answer = input("Delete every recorded shot? [y/N] ")
            except EOFError:
                answer = ""
            if answer.strip().lower() not in {"y", "yes"}:
                print("Nothing deleted.")
                return 1
        print(f"Deleted {store.reset()} shot(s).")
        return 0
    if getattr(args, "json", False):
        print(json.dumps({"record": store.record(), "shots": store.recent(args.limit)}, indent=2))
        return 0
    print(render_scoreboard(args.limit))
    return 0


def register(ctx) -> None:
    ctx.register_tool(name=TOOL_NAME, toolset="call-your-shot", schema=CALL_SHOT_SCHEMA,
                      handler=call_shot, emoji="🎯",
                      description="Predict the tool calls and minutes a task will take; hooks score it.")
    ctx.register_hook("pre_llm_call", _on_pre_llm_call)
    ctx.register_hook("post_tool_call", _on_post_tool_call)
    ctx.register_hook("post_llm_call", _on_post_llm_call)
    ctx.register_hook("on_session_end", _on_session_end)
    ctx.register_command("shots", _slash_shots, description="Show the call-your-shot scoreboard",
                         args_hint="[how many]")
    ctx.register_cli_command("shots", "Show the call-your-shot scoreboard", _cli_setup, _cli_handler,
                             description="Predicted versus measured tool calls and time for the agent's called shots.")
    ctx.register_system_prompt_section("call-your-shot.nudge", NUDGE)
