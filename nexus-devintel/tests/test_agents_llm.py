"""Tests de ``agents/llm.py`` — T3 : transport injectable, zéro réseau réel."""

import json
from urllib.error import HTTPError, URLError

import pytest

from agents.llm import (
    DEFAULT_ENDPOINT,
    GeminiClient,
    LlmError,
    LlmOutput,
    LlmProtocolError,
    MistralClient,
)

API_KEY = "test-key"


def _completion(content: str, model: str = "mistral-small-latest", usage: dict | None = None) -> dict:
    return {
        "model": model,
        "choices": [{"message": {"role": "assistant", "content": content}}],
        "usage": usage or {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }


def _recording_transport(payload_response: dict) -> tuple["object", list[dict]]:
    """Un transport qui enregistre (url, headers, body, timeout) puis répond."""
    calls: list[dict] = []

    def transport(url, headers, body, timeout):  # noqa: ANN001
        calls.append(
            {"url": url, "headers": headers, "payload": json.loads(body), "timeout": timeout}
        )
        return payload_response

    return transport, calls


# --------------------------------------------------------------------------- #
# Appel nominal
# --------------------------------------------------------------------------- #
def test_chat_sends_system_then_user_with_api_key() -> None:
    transport, calls = _recording_transport(_completion("bonjour"))
    client = MistralClient(API_KEY, transport=transport)
    out = client.chat("Tu es un assistant d'analyse de code.", "Que casse si je change ssl_.py ?")

    assert out.content == "bonjour"
    assert out.data is None                     # json_mode absent
    assert out.total_tokens == 15
    assert out.model == "mistral-small-latest"

    (call,) = calls
    assert call["url"] == DEFAULT_ENDPOINT
    assert call["headers"]["Authorization"].startswith("Bearer ")
    assert call["headers"]["Content-Type"] == "application/json"
    assert [m["role"] for m in call["payload"]["messages"]] == ["system", "user"]
    assert "response_format" not in call["payload"]


def test_chat_respects_the_timeout_from_budget() -> None:
    """Le timeout demande par le budget est bien passe au transport."""
    transport, calls = _recording_transport(_completion("ok"))
    client = MistralClient(API_KEY, timeout_s=7.5, transport=transport)
    client.chat("s", "u")
    assert calls[0]["timeout"] == 7.5


def test_custom_model_is_forwarded() -> None:
    transport, calls = _recording_transport(_completion("ok", model="mistral-large-latest"))
    client = MistralClient(API_KEY, model="mistral-large-latest", transport=transport)
    out = client.chat("s", "u")
    assert calls[0]["payload"]["model"] == "mistral-large-latest"
    assert out.model == "mistral-large-latest"


# --------------------------------------------------------------------------- #
# json_mode : sorties strictes pour le Critic
# --------------------------------------------------------------------------- #
def test_json_mode_returns_parsed_dict_only() -> None:
    payload = {"supported": True, "reason": "chaîne de hops complète"}
    transport, calls = _recording_transport(_completion(json.dumps(payload)))
    client = MistralClient(API_KEY, transport=transport)
    out = client.chat("critique", "verifie", json_mode=True)

    assert out.data == payload
    assert out.content == json.dumps(payload)
    assert calls[0]["payload"]["response_format"] == {"type": "json_object"}


def test_invalid_json_raises_protocol_error_not_a_guess() -> None:
    transport, _ = _recording_transport(_completion("pas du json {{"))
    client = MistralClient(API_KEY, transport=transport)
    with pytest.raises(LlmProtocolError):
        client.chat("critique", "verifie", json_mode=True)


def test_non_object_json_raises_protocol_error() -> None:
    transport, _ = _recording_transport(_completion("[1, 2]"))
    client = MistralClient(API_KEY, transport=transport)
    with pytest.raises(LlmProtocolError):
        client.chat("critique", "verifie", json_mode=True)


# --------------------------------------------------------------------------- #
# Erreurs : jamais un crash nu vers le graphe
# --------------------------------------------------------------------------- #
def test_http_error_maps_to_llm_error() -> None:
    def transport(url, headers, body, timeout):  # noqa: ANN001
        raise HTTPError(url, 429, "rate limited", None, None)

    client = MistralClient(API_KEY, transport=transport)
    with pytest.raises(LlmError, match="429"):
        client.chat("s", "u")


def test_network_error_maps_to_llm_error() -> None:
    def transport(url, headers, body, timeout):  # noqa: ANN001
        raise URLError("DNS en panne")

    client = MistralClient(API_KEY, transport=transport)
    with pytest.raises(LlmError, match="DNS"):
        client.chat("s", "u")


def test_response_without_choices_maps_to_llm_error() -> None:
    transport, _ = _recording_transport({"error": "boom"})
    client = MistralClient(API_KEY, transport=transport)
    with pytest.raises(LlmError, match="choices"):
        client.chat("s", "u")


def test_non_string_content_is_score_empty_not_garbage() -> None:
    transport, _ = _recording_transport(_completion(None))  # type: ignore[arg-type]
    client = MistralClient(API_KEY, transport=transport)
    out = client.chat("s", "u")
    assert out.content == ""


# --------------------------------------------------------------------------- #
# Fabrique env + garde-fous d'initialisation
# --------------------------------------------------------------------------- #
def test_from_env_returns_none_without_key(monkeypatch) -> None:
    monkeypatch.delenv("MISTRAL_API_KEY", raising=False)
    assert MistralClient.from_env() is None


def test_from_env_reads_key_and_model(monkeypatch) -> None:
    monkeypatch.setenv("MISTRAL_API_KEY", "sk-live")
    monkeypatch.setenv("NEXUS_LLM_MODEL", "mistral-large-latest")
    client = MistralClient.from_env()
    assert client is not None
    assert client._model == "mistral-large-latest"  # noqa: SLF001 - assertion de contrat
    monkeypatch.delenv("NEXUS_LLM_MODEL", raising=False)
    client2 = MistralClient.from_env()
    assert client2 is not None
    assert client2._model == "mistral-small-latest"  # noqa: SLF001


def test_empty_api_key_is_refused_at_construction() -> None:
    with pytest.raises(ValueError):
        MistralClient("")


def test_llm_output_forbids_extra_fields() -> None:
    with pytest.raises(ValueError):
        LlmOutput(content="x", surprise=1)  # type: ignore[call-arg]


# =========================================================================== #
# GeminiClient — miroir des tests MistralClient, transport injectable          #
# =========================================================================== #

GEMINI_KEY = "test-gemini-key"


def _gemini_response(text: str, prompt_tokens: int = 10, completion_tokens: int = 5) -> dict:
    """Réponse Gemini minimale valide."""
    return {
        "candidates": [{"content": {"parts": [{"text": text}]}}],
        "usageMetadata": {
            "promptTokenCount": prompt_tokens,
            "candidatesTokenCount": completion_tokens,
            "totalTokenCount": prompt_tokens + completion_tokens,
        },
    }


def test_gemini_chat_nominal() -> None:
    """Appel nominal : le contenu est extrait, l'URL contient le modèle et la clé."""
    transport, calls = _recording_transport(_gemini_response("pong"))
    client = GeminiClient(GEMINI_KEY, transport=transport)
    out = client.chat("Tu es un assistant.", "ping")

    assert out.content == "pong"
    assert out.data is None
    assert out.total_tokens == 15
    assert out.model == "gemini-2.5-flash"

    (call,) = calls
    assert "gemini-2.5-flash" in call["url"]
    # Les clés AQ. passent via header x-goog-api-key, pas en query-param
    assert GEMINI_KEY not in call["url"]
    assert call["headers"]["x-goog-api-key"] == GEMINI_KEY
    assert call["headers"]["Content-Type"] == "application/json"
    assert "Authorization" not in call["headers"]
    assert call["payload"]["system_instruction"]["parts"][0]["text"] == "Tu es un assistant."
    assert call["payload"]["contents"][0]["parts"][0]["text"] == "ping"
    assert "generationConfig" not in call["payload"]


def test_gemini_chat_json_mode() -> None:
    payload = {"supported": True, "reason": "ok"}
    transport, calls = _recording_transport(_gemini_response(json.dumps(payload)))
    client = GeminiClient(GEMINI_KEY, transport=transport)
    out = client.chat("s", "u", json_mode=True)

    assert out.data == payload
    assert calls[0]["payload"]["generationConfig"]["response_mime_type"] == "application/json"


def test_gemini_invalid_json_raises_protocol_error() -> None:
    transport, _ = _recording_transport(_gemini_response("pas du json {{"))
    client = GeminiClient(GEMINI_KEY, transport=transport)
    with pytest.raises(LlmProtocolError):
        client.chat("s", "u", json_mode=True)


def test_gemini_no_candidates_raises_llm_error() -> None:
    transport, _ = _recording_transport({"candidates": []})
    client = GeminiClient(GEMINI_KEY, transport=transport)
    with pytest.raises(LlmError, match="candidates"):
        client.chat("s", "u")


def test_gemini_http_error_maps_to_llm_error() -> None:
    from urllib.error import HTTPError

    def transport(url, headers, body, timeout):  # noqa: ANN001
        raise HTTPError(url, 429, "quota", None, None)

    client = GeminiClient(GEMINI_KEY, transport=transport)
    with pytest.raises(LlmError, match="429"):
        client.chat("s", "u")


def test_gemini_network_error_maps_to_llm_error() -> None:
    from urllib.error import URLError

    def transport(url, headers, body, timeout):  # noqa: ANN001
        raise URLError("réseau indisponible")

    client = GeminiClient(GEMINI_KEY, transport=transport)
    with pytest.raises(LlmError, match="réseau"):
        client.chat("s", "u")


def test_gemini_from_env_returns_none_without_key(monkeypatch) -> None:
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    assert GeminiClient.from_env() is None


def test_gemini_from_env_reads_key_and_model(monkeypatch) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "ai-live-key")
    monkeypatch.setenv("NEXUS_LLM_MODEL", "gemini-1.5-pro")
    client = GeminiClient.from_env()
    assert client is not None
    assert client._model == "gemini-1.5-pro"  # noqa: SLF001
    monkeypatch.delenv("NEXUS_LLM_MODEL", raising=False)
    client2 = GeminiClient.from_env()
    assert client2 is not None
    assert client2._model == "gemini-2.5-flash"  # noqa: SLF001


def test_gemini_empty_api_key_is_refused() -> None:
    with pytest.raises(ValueError):
        GeminiClient("")


def test_gemini_timeout_forwarded() -> None:
    transport, calls = _recording_transport(_gemini_response("ok"))
    client = GeminiClient(GEMINI_KEY, timeout_s=12.0, transport=transport)
    client.chat("s", "u")
    assert calls[0]["timeout"] == 12.0
