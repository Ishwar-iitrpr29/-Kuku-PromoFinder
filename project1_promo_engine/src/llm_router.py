"""
Universal Pluggable LLM Router for Kuku PromoStudio.
Supports free-tier APIs (Google Gemini, Groq, OpenRouter, Mistral, OpenAI-compatible)
and seamless deterministic fallback when running without an API key.
"""

from __future__ import annotations
import json
import os
from pathlib import Path
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import requests


def _load_dotenv_into_environ() -> None:
    """Loads environment variables from workspace .env files and Streamlit Cloud st.secrets without exposing keys in UI."""
    this_dir = Path(__file__).resolve().parent
    candidate_paths = [
        this_dir.parent.parent / ".env",  # d:\projects\KukuFM\.env
        this_dir.parent / ".env",         # project1_promo_engine\.env
        Path.home() / ".env",             # ~/.env
    ]
    for env_path in candidate_paths:
        if env_path.is_file():
            try:
                for raw_line in env_path.read_text(encoding="utf-8", errors="ignore").splitlines():
                    line = raw_line.strip()
                    if not line or line.startswith("#") or "=" not in line:
                        continue
                    k, v = line.split("=", 1)
                    k = k.strip()
                    v = v.strip().strip('"').strip("'")
                    if k and v:
                        os.environ[k] = v
            except Exception:
                pass
    # Also support Streamlit Community Cloud st.secrets when deployed from GitHub
    try:
        import streamlit as st
        for secret_key in ("GEMINI_API_KEY", "GOOGLE_API_KEY", "GROQ_API_KEY", "OPENROUTER_API_KEY", "MISTRAL_API_KEY", "OPENAI_API_KEY"):
            if secret_key in st.secrets and st.secrets[secret_key]:
                os.environ[secret_key] = str(st.secrets[secret_key]).strip()
    except Exception:
        pass


_load_dotenv_into_environ()


@dataclass
class LLMConfig:
    provider: str = "gemini"  # gemini (primary via .env)
    model: str = "gemini-2.5-flash (Standard Hybrid Promo Engine)"
    api_key: Optional[str] = None
    temperature: float = 0.2
    max_tokens: int = 2048
    timeout_sec: int = 20


PROVIDER_DEFAULTS: Dict[str, Dict[str, str]] = {
    "gemini": {
        "model": "gemini-2.5-flash",
        "env_key": "GEMINI_API_KEY",
        "endpoint": "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
        "free_limits": "10 RPM | 250,000 TPM | 250 RPD (1M context window)",
    },
    "groq": {
        "model": "llama-3.3-70b-versatile",
        "env_key": "GROQ_API_KEY",
        "endpoint": "https://api.groq.com/openai/v1/chat/completions",
        "free_limits": "30 RPM | 12,000 TPM | 1,000 RPD (~300 tok/s LPU)",
    },
    "openrouter": {
        "model": "meta-llama/llama-3.3-70b-instruct:free",
        "env_key": "OPENROUTER_API_KEY",
        "endpoint": "https://openrouter.ai/api/v1/chat/completions",
        "free_limits": "20 RPM | 50-200 RPD (Multi-model free router)",
    },
    "mistral": {
        "model": "mistral-small-latest",
        "env_key": "MISTRAL_API_KEY",
        "endpoint": "https://api.mistral.ai/v1/chat/completions",
        "free_limits": "2 RPM | 500,000 TPM | 1B tokens/month",
    },
    "openai": {
        "model": "gpt-4o-mini",
        "env_key": "OPENAI_API_KEY",
        "endpoint": "https://api.openai.com/v1/chat/completions",
        "free_limits": "Standard OpenAI / Custom base_url compatible",
    },
    "deterministic": {
        "model": "rule-hybrid-v1 (Zero-API Local Engine)",
        "env_key": "",
        "endpoint": "local",
        "free_limits": "Unlimited | <1s for 50 episodes | $0.00 cost",
    },
}

PROVIDER_MODELS: Dict[str, List[str]] = {
    "gemini": [
        "gemini-2.5-flash",
        "gemini-2.5-pro",
        "gemini-2.0-flash",
        "gemini-2.0-flash-lite",
    ],
    "groq": [
        "llama-3.3-70b-versatile",
        "llama-3.1-8b-instant",
        "mixtral-8x7b-32768",
        "gemma2-9b-it",
    ],
    "openrouter": [
        "meta-llama/llama-3.3-70b-instruct:free",
        "google/gemini-2.0-flash-exp:free",
        "deepseek/deepseek-r1:free",
        "qwen/qwen-2.5-72b-instruct:free",
    ],
    "mistral": [
        "mistral-small-latest",
        "mistral-large-latest",
        "open-mistral-nemo",
    ],
    "openai": [
        "gpt-4o-mini",
        "gpt-4o",
        "o3-mini",
    ],
}


def get_available_providers_from_env() -> Tuple[List[str], List[str]]:
    """
    Returns (selectable_providers, env_configured_providers) based on non-empty
    keys present in .env / environment variables. Never exposes key strings.
    """
    _load_dotenv_into_environ()
    configured: List[str] = []
    for prov in ["gemini", "groq", "openrouter", "mistral", "openai"]:
        env_key = PROVIDER_DEFAULTS[prov]["env_key"]
        val = os.environ.get(env_key, "").strip()
        if prov == "gemini" and not val:
            val = os.environ.get("GOOGLE_API_KEY", "").strip()
        if val:
            configured.append(prov)
    selectable = configured if configured else ["gemini"]
    return selectable, configured


class UniversalLLMRouter:
    """
    Interview-ready model router. Allows instant swapping between Gemini, Groq,
    OpenRouter, Mistral, OpenAI, or zero-key deterministic execution.
    """

    def __init__(self, config: Optional[LLMConfig] = None):
        _load_dotenv_into_environ()
        self.config = config or self._auto_detect_config()
        p_info = PROVIDER_DEFAULTS.get(self.config.provider, PROVIDER_DEFAULTS["gemini"])
        if self.config.model == "auto" or not self.config.model:
            self.config.model = p_info["model"]
        if not self.config.api_key and p_info.get("env_key"):
            self.config.api_key = os.environ.get(p_info["env_key"]) or os.environ.get("GOOGLE_API_KEY")

    @staticmethod
    def _auto_detect_config() -> LLMConfig:
        _load_dotenv_into_environ()
        for prov in ["gemini", "groq", "openrouter", "mistral", "openai"]:
            env_name = PROVIDER_DEFAULTS[prov]["env_key"]
            if env_name and os.environ.get(env_name):
                return LLMConfig(
                    provider=prov,
                    model=PROVIDER_DEFAULTS[prov]["model"],
                    api_key=os.environ.get(env_name),
                )
        return LLMConfig(
            provider="gemini",
            model=PROVIDER_DEFAULTS["gemini"]["model"],
            api_key=os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"),
        )

    def _clean_model_id(self) -> str:
        raw = (self.config.model or "gemini-2.5-flash").strip()
        return raw.split("(")[0].strip() or "gemini-2.5-flash"

    def is_live_llm(self) -> bool:
        return self.config.provider != "deterministic" and bool(self.config.api_key)

    def complete_json(
        self,
        system_prompt: str,
        user_prompt: str,
        fallback_fn: Optional[ Any ] = None,
    ) -> Dict[str, Any]:
        """Calls the configured LLM and parses JSON output, falling back gracefully on error or offline mode."""
        if not self.is_live_llm():
            return fallback_fn() if callable(fallback_fn) else {"mode": "deterministic"}

        try:
            raw_text = self.complete_text(system_prompt, user_prompt, json_mode=True)
            cleaned = raw_text.strip()
            if cleaned.startswith("```"):
                lines = cleaned.splitlines()
                cleaned = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
            return json.loads(cleaned)
        except Exception as exc:
            if callable(fallback_fn):
                res = fallback_fn()
                if isinstance(res, dict):
                    res["_llm_fallback_reason"] = str(exc)
                return res
            return {"error": str(exc)}

    def complete_text(
        self,
        system_prompt: str,
        user_prompt: str,
        json_mode: bool = False,
    ) -> str:
        if not self.is_live_llm():
            raise RuntimeError("Router is in deterministic offline mode (no API key provided).")

        prov = self.config.provider.lower()
        if prov == "gemini":
            return self._call_gemini(system_prompt, user_prompt, json_mode=json_mode)
        else:
            return self._call_openai_compatible(prov, system_prompt, user_prompt, json_mode=json_mode)

    def _call_gemini(self, system_prompt: str, user_prompt: str, json_mode: bool = False) -> str:
        clean_model = self._clean_model_id()
        url = PROVIDER_DEFAULTS["gemini"]["endpoint"].format(model=clean_model)
        params = {"key": self.config.api_key}
        gen_config: Dict[str, Any] = {
            "temperature": self.config.temperature,
            "maxOutputTokens": self.config.max_tokens,
        }
        if json_mode:
            gen_config["responseMimeType"] = "application/json"

        payload = {
            "systemInstruction": {"parts": [{"text": system_prompt}]},
            "contents": [{"role": "user", "parts": [{"text": user_prompt}]}],
            "generationConfig": gen_config,
        }
        resp = requests.post(url, params=params, json=payload, timeout=self.config.timeout_sec)
        resp.raise_for_status()
        data = resp.json()
        return data["candidates"][0]["content"]["parts"][0]["text"]

    def _call_openai_compatible(
        self, prov: str, system_prompt: str, user_prompt: str, json_mode: bool = False
    ) -> str:
        url = os.environ.get("CUSTOM_OPENAI_BASE_URL") or PROVIDER_DEFAULTS[prov]["endpoint"]
        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json",
        }
        payload: Dict[str, Any] = {
            "model": self._clean_model_id(),
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
        }
        if json_mode and prov in ("groq", "openai", "mistral"):
            payload["response_format"] = {"type": "json_object"}

        resp = requests.post(url, headers=headers, json=payload, timeout=self.config.timeout_sec)
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"]
