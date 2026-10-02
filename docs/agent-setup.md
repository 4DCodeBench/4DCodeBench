# Agent Setup

Each agent runs headlessly inside a container. During inference, `infer.py` mounts three read-only paths into the container: `/task` (task specification), `/input/reference.mp4` (reference video), and the agent's credential file. Agent execution commands are defined in [`harness/runtime/agents.py`](../harness/runtime/agents.py).

| Agent | CLI | Credential File | Reasoning Efforts |
|---|---|---|---|
| `claude` | Claude Code | `.auth/claude.json` | `low`, `medium`, `high`, `xhigh`, `max` |
| `codex` | Codex CLI | `.auth/gpt.json` | `low`, `medium`, `high`, `xhigh`, `max` |
| `antigravity` | Antigravity CLI | `.auth/gemini.json` | `low`, `medium`, `high` |
| `stirrup` | Artificial Analysis' Stirrup | None | Passed as `reasoning_effort` (omitted if `none`) |

## 1. Authenticate on the Host

Log in once on the host with the target billing account. The host CLIs generate the required credentials:

| Agent | Login Command | Credential File |
|---|---|---|
| `claude` | `claude`, then `/login` | `~/.claude/.credentials.json` |
| `codex` | `codex login` (or `printenv OPENAI_API_KEY \| codex login --with-api-key`) | `~/.codex/auth.json` |
| `antigravity` | `agy` (complete Google sign-in) | `~/.gemini/antigravity-cli/antigravity-oauth-token` |

## 2. Link Credentials into `.auth/`

From the repository root, create symbolic links:

```bash
mkdir -p .auth
ln -s ~/.claude/.credentials.json                        .auth/claude.json
ln -s ~/.codex/auth.json                                 .auth/gpt.json
ln -s ~/.gemini/antigravity-cli/antigravity-oauth-token  .auth/gemini.json
```

Use symbolic links rather than static copies. Because containers mount credentials read-only, they cannot refresh expired tokens internally. Symbolic links propagate host token refreshes into the container automatically. If a run fails with an authentication error, re-authenticate on the host.

To switch accounts, link an alternative file (e.g., `.auth/gpt_team.json`) and update the `credential` path in `runtime.toml` under `[agent.<name>]`.

## 3. Stirrup (Direct API Endpoints)

`stirrup` connects directly to any OpenAI-compatible Chat Completions endpoint without credential files. Configure `OPENAI_BASE_URL` and `OPENAI_API_KEY` under `[agent.stirrup.env]` in `runtime.toml`. The runtime forwards the `model` parameter directly to the endpoint.
