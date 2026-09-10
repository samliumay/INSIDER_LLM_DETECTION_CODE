"""OpenAI-compatible client (Ollama by default). Duck-types the Agentic Misalignment ModelClient
so the bundled classifiers can use it: `await client(model_id, messages, ...) -> LLMResponse`."""
import os, time
from dataclasses import dataclass, field
from openai import AsyncOpenAI

@dataclass
class LLMResponse:
    model_id: str
    completion: str
    reasoning: str = ""
    stop_reason: str | None = None
    usage: dict = field(default_factory=dict)
    duration: float = 0.0
    api_model: str = ""           # model id the provider *returned* (code review 2026-09-01)
    provider: str = ""            # routing provider when routed (OpenRouter), else ""

def _msgs(messages):
    """Accept both plain dict messages and the bundled classifiers' Message objects
    (`.role` may be an Enum), so the same client serves the runner and the bundled classifiers."""
    return [{"role": (m.role.value if hasattr(m.role, "value") else m.role), "content": m.content}
            if not isinstance(m, dict) else m for m in messages]

class OpenAICompatClient:
    def __init__(self, base_url: str | None = None, api_key: str = "ollama", timeout: float = 1800.0):
        # A hung call must not hold its semaphore slot forever; timeout raises so the
        # runner's retry loop actually triggers.
        self.client = AsyncOpenAI(base_url=base_url or os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434/v1"),
                                  api_key=api_key, timeout=timeout)

    async def __call__(self, model_id: str, messages, max_tokens: int = 4000, temperature: float = 1.0,
                       seed: int | None = None, **kw) -> LLMResponse:
        t = time.time()
        extra = {"seed": seed} if seed is not None else {}
        r = await self.client.chat.completions.create(model=model_id, messages=_msgs(messages),
                                                      max_tokens=max_tokens, temperature=temperature, **extra)
        m = r.choices[0].message
        # Ollama (and OpenRouter for some models) return the reasoning channel as a separate
        # `reasoning` field rather than inline tags; it is stored raw and never parsed for actions.
        reasoning = getattr(m, "reasoning", None) or (m.model_extra or {}).get("reasoning", "") or ""
        # OpenRouter reports which provider served the call; recorded per episode.
        provider = (r.model_extra or {}).get("provider", "") if hasattr(r, "model_extra") else ""
        return LLMResponse(model_id=model_id, completion=m.content or "", reasoning=reasoning,
                           stop_reason=r.choices[0].finish_reason,
                           usage=r.usage.model_dump() if r.usage else {}, duration=time.time() - t,
                           api_model=getattr(r, "model", "") or "", provider=provider or "")
