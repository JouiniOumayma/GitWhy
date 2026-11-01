"""Client LLM Mistral — T3 : minimum viable, budget au niveau des agents.

Choix retenu (option "SDK brut" du plan): un appel HTTP **direct** à
l'endpoint officiel ``/chat/completions`` via ``urllib.request`` — le même
stdlib-only que ``ingestion/github_client.py`` — au lieu du SDK
``mistralai``, qui ramènerait httpx et torch-adjacent pour un seul POST.
Ce qui change pour la soutenance : rien, l'API est la même ; ce qu'on
gagne : aucune dépendance, un transport injectable en 3 lignes pour les
tests.

Séparation des responsabilités (à expliquer en soutenenance) :

* **transport** (ce module) : POST + timeout + parse de la réponse.
    Il ne connaît ni le graphe, ni l'état des agents ;
* **budget** (``agents.state``) : chaque nœud LangGraph vérifie
  ``can_call_llm`` *avant* d'appeler ``chat()`` et fusionne
  ``llm_call_patch()`` *après* — le quota reste ainsi visible dans l'état
  de l'agent, pas enfoui dans un client ;
* **outils** (``agents.tools``) : les preuves proviennent uniquement de la
  whitelist read-only, jamais des dires d'un LLM.

``json_mode=True`` : demande la sortie JSON structurée à l'API (le Critic
l'utilise) et la parse de façon stricte — un JSON invalide est une
:class:`LlmProtocolError`, jamais un "content" approximatif passé au Critic.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Any, Callable

from pydantic import BaseModel, ConfigDict, Field

__all__ = [
    "MistralClient",
    "LlmError",
    "LlmOutput",
    "LlmProtocolError",
]

DEFAULT_ENDPOINT = "https://api.mistral.ai/v1/chat/completions"

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
    try:
        return json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise LlmError("réponse non-JSON de l'API Mistral") from exc


class MistralClient:
    """Wrapper minimal sur l'API de chat completion.

    ``transport`` est injectable (tests) : par défaut le POST urllib.
    """

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
    def from_env(cls, *, default_timeout_s: float = 45.0) -> MistralClient | None:
        """Fabrique : lit ``MISTRAL_API_KEY`` (+ ``NEXUS_LLM_MODEL`` si présent).

        Renvoie ``None`` si la clé est absente : les agents doivent alors
        fonctionner **sans** LLM — réponses par preuve seule, escalade.
        """
        api_key = os.environ.get("MISTRAL_API_KEY", "").strip()
        if not api_key:
            return None
        model = os.environ.get("NEXUS_LLM_MODEL", "").strip() or "mistral-small-latest"
        return cls(api_key, model=model, timeout_s=default_timeout_s)

    # ------------------------------------------------------------------ #
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
        keys = (
            "prompt_tokens", "completion_tokens", "total_tokens",
        )
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
