"""Provider seam: the graph asks for a role, not for a vendor.

Every node calls `get_llm("planner")` or `get_llm("synthesizer")` instead of
naming a model. Switching provider is then an env var, which is what makes the
provider comparison in the evaluation harness possible at all.
"""

import os

# Role -> model, per provider. The planner role needs reliable tool/JSON output;
# the synthesizer role needs long-form writing that stays faithful to sources.
_MODELS = {
    "groq": {
        "planner": "qwen/qwen3.8-27b",
        "synthesizer": "openai/gpt-oss-120b",
    },
    "anthropic": {
        "planner": "claude-haiku-5-5",
        "synthesizer": "claude-sonnet-5-5",
    },
}

_ROLE_TEMPERATURE = {
    # Decomposition should be stable across runs so the eval set is comparable.
    "planner": 0.0,
    # A little freedom helps the prose without loosening source fidelity.
    "synthesizer": 0.3,
}


def active_provider() -> str:
    """Resolve the provider from the environment, falling back to Groq."""
    provider = os.getenv("LLM_PROVIDER", "").strip().lower()
    if provider in _MODELS:
        return provider
    # Infer from whichever key is present rather than failing on a typo.
    if os.getenv("ANTHROPIC_API_KEY", "").strip():
        return "anthropic"
    return "groq"


def model_for(role: str, provider: str | None = None) -> str:
    """Return the model id this provider uses for the given role."""
    provider = provider or active_provider()
    try:
        return _MODELS[provider][role]
    except KeyError as exc:
        raise ValueError(f"Rôle ou fournisseur inconnu : {role} / {provider}") from exc


def get_llm(role: str, provider: str | None = None, model: str | None = None):
    """Build the chat model for a role. Raises if the provider key is missing."""
    provider = provider or active_provider()
    model = model or model_for(role, provider)
    temperature = _ROLE_TEMPERATURE.get(role, 0.2)

    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        api_key = os.getenv("ANTHROPIC_API_KEY", "").strip()
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY manquante.")
        return ChatAnthropic(
            api_key=api_key,
            model=model,
            temperature=temperature,
            max_tokens=8000,
            timeout=90,
        )

    from langchain_groq import ChatGroq

    api_key = os.getenv("GROQ_API_KEY", "").strip()
    if not api_key:
        raise ValueError("GROQ_API_KEY manquante.")
    return ChatGroq(
        api_key=api_key,
        model=model,
        temperature=temperature,
        max_retries=2,
        timeout=90,
    )
