"""Guardrails — block a request before BoxLite spins up.

Maps to https://openai.github.io/openai-agents-python/guardrails/

When an ``input_guardrail`` trips, the SDK raises
``InputGuardrailTripwireTriggered`` **before** the run reaches the sandbox.
With BoxLite that means we never pay the MicroVM cold-start cost on a
rejected prompt — a free latency win.

Implementation note: this guardrail uses a plain text "YES / NO" verdict
rather than a Pydantic ``output_type`` model. Reason: not every OpenAI-
compatible LLM provider supports ``response_format=json_schema`` (notably
DeepSeek-V4-Flash via the SiliconFlow ``.com`` route does not). Plain text
is the most portable shape.

Run::

    export OPENAI_API_KEY=sk-...
    python examples/08_guardrails.py
"""

from __future__ import annotations

import asyncio

from agents import (
    Agent,
    GuardrailFunctionOutput,
    InputGuardrailTripwireTriggered,
    Runner,
    input_guardrail,
)
from agents.run import RunConfig
from agents.sandbox import SandboxAgent, SandboxRunConfig
from agents.sandbox.capabilities import Shell

from boxlite_openai_agents import BoxLiteSandboxClient, BoxLiteSandboxClientOptions


# ---- Lightweight PII screen agent (single LLM call, plain-text verdict) ----
pii_screen = Agent(
    name="PII screen",
    instructions=(
        "Decide if the user prompt contains a US Social Security Number "
        "(pattern XXX-XX-XXXX). Reply with exactly one word: YES or NO."
    ),
)


@input_guardrail
async def pii_guardrail(ctx, agent, user_input):
    result = await Runner.run(pii_screen, user_input, context=ctx.context)
    verdict = (result.final_output or "").strip().upper()
    return GuardrailFunctionOutput(
        output_info=verdict,
        tripwire_triggered=verdict.startswith("YES"),
    )


async def main() -> None:
    sandbox_agent = SandboxAgent(
        name="Renewal Analyst",
        instructions="Help with the renewal request, citing /workspace files.",
        capabilities=[Shell()],
        input_guardrails=[pii_guardrail],
    )

    bad_prompt = "Please look up account for SSN 123-45-6789."
    print(f"Prompt: {bad_prompt!r}")

    try:
        await Runner.run(
            sandbox_agent,
            bad_prompt,
            run_config=RunConfig(
                sandbox=SandboxRunConfig(
                    client=BoxLiteSandboxClient(),
                    options=BoxLiteSandboxClientOptions(image="python:3.12-slim"),
                ),
                workflow_name="boxlite-guardrail-demo",
            ),
            max_turns=4,
        )
        print("guardrail did NOT trip (unexpected)")
    except InputGuardrailTripwireTriggered:
        print("✓ PII guardrail tripped — sandbox was never created.")


if __name__ == "__main__":
    asyncio.run(main())
