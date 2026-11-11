"""Client LLM — T3 : MistralClient (original) + GeminiClient (ajout).

Les deux clients partagent la même interface : ``chat(system, user, *, json_mode)``
renvoie un ``LlmOutput``. Le transport est urllib stdlib-only, aucune dépendance
externe. ``create_app_from_env`` choisit Gemini si ``GEMINI_API_KEY`` est défini,
sinon Mistral si ``MISTRAL_API_KEY`` est défini, sinon ``None`` (mode dégradé).
"""

from __future__ import annotations

import json
import logging
import os
import urllib.error
import urllib.request
from typing import Any, Callable

from pydantic import BaseModel, ConfigDict, Field

__all__ = [
    "MistralClient",
    "GeminiClient",
    "LlmError",
    "LlmOutput",
    "LlmProtocolError",
]

logger = logging.getLogger(__name__)

DEFAULT_ENDPOINT = "https://api.mistral.ai/v1/chat/completions"
GEMINI_ENDPOINT_TPL = (
    "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
)

#: Modèle Gemini principal par défaut. ``gemini-2.5-flash`` a été retiré de
#: l'API (404 « no longer available to new users ») : ne pas y revenir.
DEFAULT_GEMINI_MODEL = "gemini-flash-lite-latest"
#: Replis essayés dans l'ordre quand le principal échoue (404 / 503 / timeout).
#: Surchargeable par ``NEXUS_LLM_FALLBACKS`` (liste séparée par des virgules).
DEFAULT_GEMINI_FALLBACKS: tuple[str, ...] = (
    "gemini-3.5-flash-lite",
    "gemini-3.1-flash-lite",
)

#: Transport injectable pour les tests : (url, headers, body_bytes, timeout) -> dict
Transport = Callable[[str, dict[str, str], bytes, float], dict[str, Any]]


class LlmError(RuntimeError):
    """Le transport a échoué (réseau, HTTP non-2xx, réponse mal formée)."""


class LlmProtocolError(LlmError):
    """Le LLM a répondu 200 mais un contenu hors contrat (JSON invalide)."""


class LlmOutput(BaseModel):
    """Sortie d'un appel : le texte brut + la consommation réelle."""

    model_config = ConfigDict(extra="forbid")

    model: str | None = None
    content: str = ""
    #: Set uniquement quand ``json_mode=True`` : le dict analysé.
    data: dict[str, Any] | None = None
    token_usage: dict[str, int] = Field(default_factory=dict)

    @property
    def total_tokens(self) -> int:
        return int(self.token_usage.get("total_tokens", 0))


def _default_transport(
    url: str, headers: dict[str, str], body: bytes, timeout: float
) -> dict[str, Any]:
    """Le transport réel : un POST urllib, tout en un, comme github_client."""
    request = urllib.request.Request(url, data=body, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = response.read()
    except urllib.error.HTTPError as exc:
        raise LlmError(f"HTTP {exc.code} : {exc.reason}") from exc
    except urllib.error.URLError as exc:
        raise LlmError(f"réseau inaccessible : {exc.reason}") from exc
    except OSError as exc:
        # TimeoutError / connexion coupée : un timeout doit être un LlmError,
        # sinon il remonte brut jusqu'à l'API et devient un HTTP 500.
        raise LlmError(f"transport interrompu après {timeout:g}s : {exc}") from exc
    try:
        return json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise LlmError("réponse non-JSON de l'API") from exc


class MistralClient:
    """Wrapper minimal sur l'API de chat completion Mistral."""

    def __init__(
        self,
        api_key: str,
        *,
        model: str = "mistral-small-latest",
        endpoint: str = DEFAULT_ENDPOINT,
        timeout_s: float = 45.0,
        transport: Transport | None = None,
    ) -> None:
        if not api_key:
            raise ValueError("api_key vide : pas d'appel Mistral possible")
        self._api_key = api_key
        self._model = model
        self._endpoint = endpoint
        self._timeout_s = timeout_s
        self._transport = transport or _default_transport

    @classmethod
    def from_env(cls, *, default_timeout_s: float = 45.0) -> "MistralClient | None":
        """Fabrique : lit ``MISTRAL_API_KEY`` (+ ``NEXUS_LLM_MODEL`` si présent).

        Renvoie ``None`` si la clé est absente.
        """
        api_key = os.environ.get("MISTRAL_API_KEY", "").strip()
        if not api_key:
            return None
        model = os.environ.get("NEXUS_LLM_MODEL", "").strip() or "mistral-small-latest"
        return cls(api_key, model=model, timeout_s=default_timeout_s)

    def chat(self, system: str, user: str, *, json_mode: bool = False) -> LlmOutput:
        """UN appel, un texte. Le quota est géré côté état, pas ici."""
        payload: dict[str, Any] = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}

        body = json.dumps(payload).encode("utf-8")
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": "NEXUS-DevIntel-agents/0.1",
        }
        try:
            data = self._transport(self._endpoint, headers, body, self._timeout_s)
        except LlmError:
            raise
        except urllib.error.HTTPError as exc:
            raise LlmError(f"HTTP {exc.code} : {exc.reason}") from exc
        except urllib.error.URLError as exc:
            raise LlmError(f"réseau inaccessible : {exc.reason}") from exc

        choices = data.get("choices") if isinstance(data, dict) else None
        if not isinstance(choices, list) or not choices:
            raise LlmError("réponse sans 'choices' : API Mistral incompatible")
        message = choices[0].get("message") if isinstance(choices[0], dict) else None
        content = (message or {}).get("content")

        output = LlmOutput(
            model=data.get("model") or self._model,
            content=content if isinstance(content, str) else "",
            token_usage=self._token_usage(data),
        )
        if json_mode:
            output = output.model_copy(
                update={"data": self._strict_json(output.content)}
            )
        return output

    @staticmethod
    def _token_usage(data: dict[str, Any]) -> dict[str, int]:
        usage = data.get("usage") if isinstance(data, dict) else None
        if not isinstance(usage, dict):
            return {}
        keys = ("prompt_tokens", "completion_tokens", "total_tokens",)
        return {key: int(usage[key]) for key in keys if isinstance(usage.get(key), int)}

    @staticmethod
    def _strict_json(content: str) -> dict[str, Any]:
        """Parse strict : un JSON invalide est du protocole erratique."""
        try:
            data = json.loads(content)
        except json.JSONDecodeError as exc:
            raise LlmProtocolError(f"JSON du LLM invalide : {exc}") from exc
        if not isinstance(data, dict):
            raise LlmProtocolError("le LLM a renvoyé un JSON non-objet")
        return data


# --------------------------------------------------------------------------- #
# Client Gemini (Google AI Studio) — même interface que MistralClient         #
# --------------------------------------------------------------------------- #

class GeminiClient:
    """Wrapper minimal sur l'API Gemini (Google AI Studio).

    Interface identique à ``MistralClient`` : ``chat(system, user, *, json_mode)``
    renvoie un ``LlmOutput``. Transport urllib stdlib-only, zéro dépendance.

    ``json_mode=True`` passe ``response_mime_type: application/json`` dans
    ``generationConfig`` — Gemini renvoie alors du JSON garanti valide.
    """

    def __init__(
        self,
        api_key: str,
        *,
        model: str = DEFAULT_GEMINI_MODEL,
        fallback_models: tuple[str, ...] = DEFAULT_GEMINI_FALLBACKS,
        timeout_s: float = 45.0,
        transport: Transport | None = None,
    ) -> None:
        if not api_key:
            raise ValueError("api_key vide : pas d'appel Gemini possible")
        self._api_key = api_key
        self._model = model
        self._fallback_models = tuple(
            m for m in fallback_models if m and m != model
        )
        self._timeout_s = timeout_s
        self._transport = transport or _default_transport

    @property
    def model_chain(self) -> tuple[str, ...]:
        """Modèles essayés dans l'ordre : le principal, puis les replis."""
        return (self._model, *self._fallback_models)

    @classmethod
    def from_env(cls, *, default_timeout_s: float = 45.0) -> "GeminiClient | None":
        """Fabrique : lit ``GEMINI_API_KEY`` (+ ``NEXUS_LLM_MODEL`` si présent).

        ``NEXUS_LLM_FALLBACKS`` (modèles séparés par des virgules) remplace la
        chaîne de repli par défaut. Renvoie ``None`` si la clé est absente.
        """
        api_key = os.environ.get("GEMINI_API_KEY", "").strip()
        if not api_key:
            return None
        model = os.environ.get("NEXUS_LLM_MODEL", "").strip() or DEFAULT_GEMINI_MODEL
        raw = os.environ.get("NEXUS_LLM_FALLBACKS", "").strip()
        fallbacks = (
            tuple(m.strip() for m in raw.split(",") if m.strip())
            if raw
            else DEFAULT_GEMINI_FALLBACKS
        )
        return cls(
            api_key,
            model=model,
            fallback_models=fallbacks,
            timeout_s=default_timeout_s,
        )

    def chat(self, system: str, user: str, *, json_mode: bool = False) -> LlmOutput:
        """UN appel logique : le modèle principal, puis les replis.

        Un modèle retiré (404), surchargé (503) ou muet (timeout) ne doit pas
        remonter jusqu'à l'agent : on passe au suivant, et on ne lève qu'une
        fois la chaîne épuisée.
        """
        errors: list[str] = []
        last_exc: LlmError | None = None
        for model in self.model_chain:
            try:
                return self._chat_once(model, system, user, json_mode=json_mode)
            except LlmError as exc:
                last_exc = exc
                errors.append(f"{model} -> {exc}")
                logger.warning(json.dumps({
                    "event": "llm_model_failed",
                    "model": model,
                    "error": str(exc),
                }, ensure_ascii=False))
        # On conserve la nature de la dernière panne : un JSON hors contrat
        # reste un LlmProtocolError, un 404/503 reste un LlmError.
        message = "chaîne de modèles épuisée : " + " | ".join(errors)
        raise type(last_exc)(message) from last_exc

    def _chat_once(
        self, model: str, system: str, user: str, *, json_mode: bool = False
    ) -> LlmOutput:
        """UN appel vers l'API Gemini generateContent, sur un modèle donné.

        Les clés au format ``AQ.`` (nouveau format Google AI Studio) sont
        transmises via l'en-tête ``x-goog-api-key`` plutôt qu'en query-param
        ``?key=`` pour éviter les 404 de routage de l'API.
        """
        url = GEMINI_ENDPOINT_TPL.format(model=model)

        payload: dict[str, Any] = {
            "system_instruction": {"parts": [{"text": system}]},
            "contents": [{"role": "user", "parts": [{"text": user}]}],
        }
        if json_mode:
            payload["generationConfig"] = {"response_mime_type": "application/json"}

        body = json.dumps(payload).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": "NEXUS-DevIntel-agents/0.1",
            # Les clés AQ. nécessitent le header x-goog-api-key (pas query-param)
            "x-goog-api-key": self._api_key,
        }

        try:
            data = self._transport(url, headers, body, self._timeout_s)
        except LlmError:
            raise
        except urllib.error.HTTPError as exc:
            raise LlmError(f"HTTP {exc.code} : {exc.reason}") from exc
        except urllib.error.URLError as exc:
            raise LlmError(f"réseau inaccessible : {exc.reason}") from exc

        # Extraction du texte depuis la réponse Gemini
        candidates = data.get("candidates") if isinstance(data, dict) else None
        if not isinstance(candidates, list) or not candidates:
            raise LlmError(f"réponse Gemini sans 'candidates': {data}")

        parts = (candidates[0].get("content") or {}).get("parts") or []
        content = parts[0].get("text", "") if parts else ""

        usage_meta = data.get("usageMetadata") or {}
        token_usage = {
            "prompt_tokens": int(usage_meta.get("promptTokenCount", 0)),
            "completion_tokens": int(usage_meta.get("candidatesTokenCount", 0)),
            "total_tokens": int(usage_meta.get("totalTokenCount", 0)),
        }

        output = LlmOutput(
            model=model,
            content=content if isinstance(content, str) else "",
            token_usage=token_usage,
        )
        if json_mode:
            output = output.model_copy(
                update={"data": MistralClient._strict_json(output.content)}
            )
        return output
