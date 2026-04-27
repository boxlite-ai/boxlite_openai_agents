# Examples

Each script is the BoxLite-equivalent of a section in the [OpenAI Agents SDK guide](https://developers.openai.com/api/docs/guides/agents). The agent definition is the *same* you would write for any other sandbox provider; only the `client=` line changes. Run each as:

```bash
python examples/01_quickstart.py
```

| File | Maps to | What it shows | LLM cost |
| --- | --- | --- | --- |
| `01_quickstart.py` | [Quickstart](https://developers.openai.com/api/docs/guides/agents/quickstart) | Three-line drop-in: swap any hosted client for `BoxLiteSandboxClient()`. | 1 turn |
| `02_persist_resume.py` | [Sandbox sessions](https://developers.openai.com/api/docs/guides/agents/sandboxes) | Native QCOW2 CoW snapshot via `persist_workspace()` / `resume()`. Second turn skips `pip install`. | 2 turns |
| `03_manifest.py` | [Sandbox agents → Manifest](https://developers.openai.com/api/docs/guides/agents/sandboxes) | `File` + `Dir` + `LocalDir` materialization (cloud `Mount` deferred to v0.2). | 1 turn |
| `04_capabilities.py` | [Sandbox agents → Capabilities](https://developers.openai.com/api/docs/guides/agents/sandboxes) | `Shell()` + `Memory()` across two runs of the same session. | 2 turns |
| `05_custom_llm.py` | [Models and providers](https://developers.openai.com/api/docs/guides/agents/models) | DeepSeek / Qwen / any OpenAI-compatible gateway driving a BoxLite sandbox. | 1 turn |
| `06_orchestration.py` | [Orchestration](https://developers.openai.com/api/docs/guides/agents/orchestration) | A non-sandbox triage agent handing off to a `SandboxAgent`. | 2 turns |
| `07_function_tool.py` | [Using tools](https://developers.openai.com/api/docs/guides/agents/tools) | Harness-side `function_tool` running alongside sandbox shell exec. | 2 turns |
| `08_guardrails.py` | [Guardrails](https://openai.github.io/openai-agents-python/guardrails/) | `input_guardrail` trips before the sandbox is ever spun up — free latency win. | 1 turn |
| `09_tracing.py` | [Tracing](https://openai.github.io/openai-agents-python/tracing/) | Sandbox ops appear as standard SDK spans inside a `trace("...")` context. | 1 turn |
| `10_airgapped.py` | (BoxLite-only) | Disabled SDK trace export + deny-all egress + non-OpenAI model = an agent that runs without phoning home. | 1 turn |

## Prerequisites

```bash
pip install -e ".[test]"
```

For OpenAI-hosted models:

```bash
export OPENAI_API_KEY=sk-...
```

For non-OpenAI providers (DeepSeek / Qwen / self-hosted vLLM, used by `05_custom_llm.py` and `10_airgapped.py`), edit `integration_design/KEY/custom_openai_llm.md`:

```text
model = deepseek-ai/DeepSeek-V4-Flash
base_url = https://api.siliconflow.com/v1/chat/completions
llm_key = sk-...
```

## Footguns to remember (mirrors README §Production runbook)

1. **`capabilities=[Shell()]`** — `Capabilities.default()` adds a hosted `apply_patch` tool that the Chat Completions API rejects on every non-OpenAI provider. Pin your capabilities explicitly when using DeepSeek / Qwen / vLLM.
2. **`egress_allowlist=()` is deny-all** — `pip install pandas` from inside the sandbox will hang until you opt domains in.
3. **`set_tracing_disabled(True)` for air-gapped** — the SDK exports traces to `api.openai.com` by default; if data residency matters, turn it off (see `10_airgapped.py`).
4. **Imperative prompts for reasoning models** — Qwen3.6, DeepSeek-R1, QwQ etc. tend to "think" their way to an answer instead of calling tools. Spell out *MUST use the exec_command tool* in `instructions` (see `05_custom_llm.py`).

## Stage / journey tests

These scripts are illustrative. The full compatibility matrix — every stage above plus negative tests, rerun loops, latency budgets — lives under `tests/journey/`. Run the entire matrix:

```bash
PER_STAGE_TIMEOUT=240 bash tests/journey/run_journey.sh
```

The most recent live run (Qwen3.6-35B-A3B + BoxLite) is reported in [`integration_design/05-compatibility-journey-report.md`](../integration_design/05-compatibility-journey-report.md).
