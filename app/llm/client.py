import json
import os
import re
import time
import uuid
from dataclasses import dataclass, asdict
from typing import Any, Dict, Generator, Optional
import httpx

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass


_PRICING_PER_1M = {
    "openai/gpt-oss-20b": (0.10, 0.40),
    "openai/gpt-oss-120b": (0.25, 0.80),
    "qwen/qwen3.8-27b": (0.20, 0.60),
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4o": (2.50, 10.00),
    "ollama": (0.0, 0.0),
}


@dataclass
class LLMTrace:
    trace_id: str
    provider: str
    model: str
    latency_ms: float
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    estimated_cost_usd: float
    fallback_used: bool
    error: Optional[str] = None


class LLMGateway:
    """
    Production multi-provider LLM client (Groq / OpenAI / Ollama) with
    structured JSON parsing, streaming, and telemetry recording.
    """

    def __init__(self, trace_log_path: Optional[str] = None):
        root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        self.trace_log_path = trace_log_path or os.path.join(root, "artifacts", "llm_traces.jsonl")
        os.makedirs(os.path.dirname(self.trace_log_path), exist_ok=True)
        self.timeout_s = float(os.getenv("LLM_TIMEOUT_SECONDS", "25.0"))

    def resolve_provider(self) -> Dict[str, str]:
        """Resolves active provider endpoint, key, and model from environment."""
        provider = os.getenv("LLM_PROVIDER", "").strip().lower()
        groq_key = os.getenv("GROQ_API_KEY", "").strip()
        openai_key = os.getenv("OPENAI_API_KEY", "").strip()
        ollama_host = os.getenv("OLLAMA_HOST", "").strip()

        if (provider == "openai" or (not provider and openai_key and not groq_key)) and openai_key:
            return {
                "provider": "openai",
                "base_url": os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/"),
                "api_key": openai_key,
                "model": os.getenv("LLM_MODEL", "gpt-4o-mini"),
            }
        if (provider == "groq" or (not provider and groq_key)) and groq_key:
            return {
                "provider": "groq",
                "base_url": "https://api.groq.com/openai/v1",
                "api_key": groq_key,
                "model": os.getenv("LLM_MODEL", "qwen/qwen3.8-27b"),
            }
        if provider == "ollama" or ollama_host:
            host = (ollama_host or "http://127.0.0.1:11434").rstrip("/")
            return {
                "provider": "ollama",
                "base_url": f"{host}/v1",
                "api_key": "ollama",
                "model": os.getenv("LLM_MODEL", "qwen2.5:3b-instruct"),
            }
        return {"provider": "offline", "base_url": "", "api_key": "", "model": "deterministic-fallback"}

    @staticmethod
    def _strip_reasoning(text: str) -> str:
        return re.sub(r"<think>.*?</think>", "", text or "", flags=re.DOTALL).strip()

    @staticmethod
    def _estimate_cost(model: str, prompt_toks: int, comp_toks: int) -> float:
        in_p, out_p = _PRICING_PER_1M.get(model, (0.20, 0.60))
        return round((prompt_toks * in_p + comp_toks * out_p) / 1_000_000.0, 6)

    def _record_trace(self, trace: LLMTrace) -> None:
        try:
            with open(self.trace_log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(asdict(trace)) + "\n")
        except Exception:
            pass

    def complete(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.1,
        max_tokens: int = 700,
        json_mode: bool = False,
    ) -> Dict[str, Any]:
        cfg = self.resolve_provider()
        trace_id = uuid.uuid4().hex[:12]
        if cfg["provider"] == "offline":
            trace = LLMTrace(trace_id, "offline", cfg["model"], 0.0, 0, 0, 0, 0.0, True, "No API key or Ollama configured")
            self._record_trace(trace)
            return {"ok": False, "text": "", "trace": asdict(trace)}

        payload: Dict[str, Any] = {
            "model": cfg["model"],
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if json_mode and cfg["provider"] in {"groq", "openai"}:
            payload["response_format"] = {"type": "json_object"}

        t0 = time.perf_counter()
        try:
            res = httpx.post(
                f"{cfg['base_url']}/chat/completions",
                headers={"Authorization": f"Bearer {cfg['api_key']}", "Content-Type": "application/json"},
                json=payload,
                timeout=self.timeout_s,
            )
            latency_ms = round((time.perf_counter() - t0) * 1000, 2)
            res.raise_for_status()
            data = res.json()
            raw_content = data["choices"][0]["message"]["content"]
            content = self._strip_reasoning(raw_content)
            usage = data.get("usage", {})
            p_tok = int(usage.get("prompt_tokens", 0))
            c_tok = int(usage.get("completion_tokens", 0))
            t_tok = int(usage.get("total_tokens", p_tok + c_tok))
            trace = LLMTrace(
                trace_id=trace_id,
                provider=cfg["provider"],
                model=cfg["model"],
                latency_ms=latency_ms,
                prompt_tokens=p_tok,
                completion_tokens=c_tok,
                total_tokens=t_tok,
                estimated_cost_usd=self._estimate_cost(cfg["model"], p_tok, c_tok),
                fallback_used=False,
            )
            self._record_trace(trace)
            return {"ok": True, "text": content, "trace": asdict(trace)}
        except Exception as exc:
            latency_ms = round((time.perf_counter() - t0) * 1000, 2)
            trace = LLMTrace(trace_id, cfg["provider"], cfg["model"], latency_ms, 0, 0, 0, 0.0, True, str(exc)[:180])
            self._record_trace(trace)
            return {"ok": False, "text": "", "trace": asdict(trace)}

    def complete_json(self, system_prompt: str, user_prompt: str, max_tokens: int = 500) -> Dict[str, Any]:
        res = self.complete(system_prompt, user_prompt, temperature=0.0, max_tokens=max_tokens, json_mode=True)
        if not res["ok"]:
            return {"ok": False, "data": None, "trace": res["trace"]}
        txt = res["text"].strip()
        try:
            return {"ok": True, "data": json.loads(txt), "trace": res["trace"]}
        except Exception:
            m = re.search(r"\{.*\}", txt, flags=re.DOTALL)
            if m:
                try:
                    return {"ok": True, "data": json.loads(m.group(0)), "trace": res["trace"]}
                except Exception:
                    pass
        return {"ok": False, "data": None, "trace": res["trace"]}


llm_gateway = LLMGateway()
