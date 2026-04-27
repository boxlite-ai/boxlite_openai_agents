# boxlite-openai-agents

> A local, embedded sandbox backend for the [OpenAI Agents SDK](https://openai.github.io/openai-agents-python/). Drop it into your existing `Runner` setup — your agents now execute code, write files, and run shells inside a real hardware-isolated MicroVM running on the same machine as your Python process.

## Why this exists

If you're already using the OpenAI Agents SDK, you've seen this pattern:

```python
run_config=RunConfig(sandbox=SandboxRunConfig(client=SomeHostedSandbox()))
```

Every first-wave sandbox provider behind that interface is a hosted SaaS (or a separate daemon you have to install and keep running). That's fine for prototypes — but it leaks your prompts, code, and intermediate artifacts to a third party, charges per-second, adds 100–400 ms of network latency to every tool call, and makes air-gapped or regulated deployments impossible.

`boxlite-openai-agents` is a drop-in `SandboxClient` that runs the agent's compute **inside your own process** on a KVM (Linux) or Hypervisor.framework (macOS) MicroVM. No Docker Desktop, no daemon, no remote round-trip. Snapshots and forks come from native QCOW2 copy-on-write — so a `Runner.run_sync(...)` that previously rebuilt environment state on every turn now resumes from a saved disk image in milliseconds.

The package implements the upstream `SandboxClient` / `SandboxSession` / `SnapshotBase` protocols 1:1 on top of the public [`boxlite`](https://pypi.org/project/boxlite/) Python SDK. Your `Runner`, `Agent`, tools, handoffs, and tracing code keep working unchanged.

## When to reach for this

- **Code-interpreter-style agents** that write, edit, and run files across many tool calls — local IO is faster than any RPC sandbox.
- **Long-running / branching agent runs** where you want to fork from a known-good state (e.g. evaluator harnesses, multi-candidate code generation, replay-based debugging).
- **Air-gapped / on-prem / regulated** deployments where prompts and code can't leave the host.
- **CI and offline tests** for agents — no hosted-sandbox account, no rate limit, no flaky network.

## Install

```bash
pip install boxlite-openai-agents
```

Requires `openai-agents>=0.14,<0.16`, `boxlite>=0.8.2`, Python 3.10+. On Linux you need `/dev/kvm`; on macOS Apple Silicon you need Hypervisor.framework (default, no setup).

## Quickstart

```python
from agents import Runner
from agents.run import RunConfig
from agents.sandbox import SandboxAgent, SandboxRunConfig
from boxlite_openai_agents import BoxLiteSandboxClient, BoxLiteSandboxClientOptions

agent = SandboxAgent(name="local-coder", instructions="You are a careful engineer.")

result = Runner.run_sync(
    agent,
    "Write fizzbuzz.py and run it.",
    run_config=RunConfig(
        sandbox=SandboxRunConfig(
            client=BoxLiteSandboxClient(),
            options=BoxLiteSandboxClientOptions(image="python:3.12-slim"),
        ),
    ),
)
print(result.final_output)
```

That's it — the agent loop, tool routing, and tracing all work exactly as documented in the OpenAI Agents SDK. Only the sandbox backend changed.

## ⚠️ Production runbook — three things to get right before you ship

Most BoxLite incidents in the wild come from one of these three. Read this section before you go to prod.

### 1. Non-OpenAI LLMs require `capabilities=[Shell()]`

`SandboxAgent` defaults to `Capabilities.default()`, which includes a hosted `apply_patch` tool. The OpenAI Chat Completions API used by every non-OpenAI provider (DeepSeek, Qwen, self-hosted vLLM, OpenRouter, ...) **rejects hosted tools** outright with `UserError: Hosted tools are not supported with the ChatCompletions API`.

```python
from agents.sandbox.capabilities import Shell

agent = SandboxAgent(
    name="...",
    instructions="...",
    capabilities=[Shell()],          # ← required for non-OpenAI providers
    model=OpenAIChatCompletionsModel(...),
)
```

### 2. `egress_allowlist=()` is **deny-all** — `pip install` and `git clone` will fail until you opt in

The default is intentional (BoxLite's whole point is *no surprises out the network boundary*). The first time your agent tries `pip install pandas` and it hangs, this is why.

```python
options=BoxLiteSandboxClientOptions(
    image="python:3.12-slim",
    egress_allowlist=(
        "pypi.org",
        "files.pythonhosted.org",
        "github.com",
        "raw.githubusercontent.com",
        # add your private registries / package mirrors here
    ),
)
```

For **truly air-gapped** workloads, leave this empty *and* pre-bake the dependencies into your OCI image.

### 3. Disable SDK tracing in air-gapped deployments

The OpenAI Agents SDK exports traces to `api.openai.com` by default. If you bought BoxLite for data-residency reasons, this silently undermines that promise. Turn it off explicitly, or wire in a local OpenTelemetry exporter:

```python
from agents import set_tracing_disabled
set_tracing_disabled(True)

# or, for self-hosted observability:
# from agents.tracing import add_trace_processor
# add_trace_processor(YourLocalOTLPExporter())
```

## Capabilities

| Capability | Default | Notes |
| --- | --- | --- |
| Embedded library, no daemon | ✅ | Pure `pip install`; no Docker, no service to run. |
| KVM / HVF hardware isolation | ✅ | Independent guest kernel per `SandboxSession`. |
| Native QCOW2 CoW snapshot | ✅ | Wired into `persist_workspace()` / `resume()` — fork an agent state in milliseconds. |
| Air-gapped / offline | ✅ | Default `egress_allowlist=[]` blocks all outbound network from the guest. |
| Tracing-compatible | ✅ | Standard OpenAI Agents SDK traces (per-tool spans, latency) work unchanged. |


## License

Apache-2.0
