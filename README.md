# call-your-shot

A Hermes Agent plugin that makes the agent call its shot.

Before a multi-step task the agent states how many tool calls and how many minutes the work will
take. Hooks then measure what actually happened, score the estimate, and keep a scoreboard. The
next time the agent calls a shot, the tool result tells it its own record ("actual tool calls came to
1.8x your estimates") together with a calibrated figure, so the estimates can improve.

Agents give effort estimates all the time ("this is a quick fix"). Nothing checks them. This
plugin does, with numbers taken from the tool loop itself, not from the model.

## What you get

- **`call_shot` tool.** The agent calls it once at the start of a task with `tool_calls`,
  `minutes` and an optional `task` label. The estimate is locked for the turn.
- **Measurement by hooks.** `post_tool_call` counts the real tool calls after the shot,
  `post_llm_call` settles the shot when the turn ends. The model cannot report its own numbers.
- **Feedback to the agent.** A settled shot is reported back at the start of the next turn, and
  every new `call_shot` result carries the record over the last 20 shots plus the estimate
  adjusted by that record.
- **Scoreboard for you.** `/shots` in a session or `hermes shots` in a terminal.

## Example

Output of `hermes shots` after real sessions on the author's machine:

```
CALL YOUR SHOT
3 shot(s) settled   average score 45/100   (record over the last 3)
  tool calls: your estimates are on target (1.05x)
  time:       actual time came to 0.07x your estimates (you overestimate)

  #  when         task                                calls      time              score
  4  11-11 21:04  README for cart.py with verifie...  5 -> 6     3m00s -> 0m22s       48
  3  11-11 21:04  Type hints + docstrings in cart...  5 -> 4     3m00s -> 0m06s       42
  2  11-11 21:04  add remove_item + tests             5 -> 6     2m00s -> 0m09s       45

calls and time read predicted -> actual; score is 100 for a perfect call
```

`calls` and `time` read predicted -> actual.

## Install

```
hermes plugins install call-your-shot
hermes plugins enable call-your-shot
```

Start a new session afterwards; tools and the system prompt section load at session start.
Requires Hermes Agent 0.23.0 or later. No Python dependencies, no API key, no account.

## Commands

| Command | Where | What it does |
|---|---|---|
| `/shots [n]` | in a session | Scoreboard with the last `n` shots (default 10) |
| `hermes shots` | terminal | The same scoreboard |
| `hermes shots --json` | terminal | Record and recent shots as JSON |
| `hermes shots --reset` | terminal | Delete every recorded shot (asks first; `--yes` skips the question) |

## How a shot is scored

A shot is measured from the moment `call_shot` returns to the end of that turn.

- Closeness of each estimate is `min(predicted, actual) / max(predicted, actual)`, so being 2x
  over and 2x under cost the same.
- The score is the mean closeness of the tool-call estimate and the time estimate, as 0 to 100.
- The record's bias figures are geometric means of `actual / predicted` over the last 20 settled
  shots. Above 1 means the agent underestimates.
- The record is shown to the agent once 3 shots have settled.

What is and is not counted:

- Every tool the turn runs after the shot counts, including tools from other plugins and MCP
  servers. The `call_shot` call itself does not.
- Tool-loading plumbing (`tool_search`, `tool_describe`, and a rejected `tool_call` wrapper) does
  not count.
- A turn that is interrupted or fails is not scored; its shot is discarded.
- A turn that ran 2 or more tool calls with no shot is counted as an uncalled turn and shown on
  the scoreboard.

## Limits

- The plugin asks the agent to call its shot through a system prompt section. A model can ignore
  that; the scoreboard shows how many multi-step turns ran with no shot.
- One shot per turn. Work that spans several turns is scored per turn.
- Time is wall-clock time for the turn, so it includes time spent waiting for your approval.
- Tool calls made by delegated subagents run in their own sessions and are not added to the
  parent's count.
- A shot that is open when Hermes is killed is discarded after 12 hours.

## Hermes surfaces used

`ctx.register_tool` (`call_shot`), hooks `pre_llm_call`, `post_tool_call`, `post_llm_call` and
`on_session_end`, `ctx.register_command` (`/shots`), `ctx.register_cli_command` (`hermes shots`),
and `ctx.register_system_prompt_section`. No core code is patched.

## Disclosure

- **Network:** none. The plugin makes no network calls and no model calls of its own.
- **Stored data:** one local, unencrypted SQLite file at
  `HERMES_HOME/plugin-data/call-your-shot/shots.db`. Per shot it stores the task label (the
  agent's label, or the first 60 characters of your message when the agent gives none), the
  session id, the model name, the predicted and measured numbers, and a per-tool-name count. It
  stores no tool arguments and no tool results.
- **Context added:** a system prompt section of about 500 characters in every session, and one
  short line at the start of the turn after a shot settles. Both cost input tokens.
- **Shell commands, background processes, credentials, telemetry:** none.

## Development

```
python tests/test_flow.py                    # flow tests, standard library only
hermes plugins validate . --install-deps     # the catalog check
```

## License

MIT
