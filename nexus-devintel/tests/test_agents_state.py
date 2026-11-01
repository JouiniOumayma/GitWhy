"""Tests de ``agents/state.py`` — T1, sans LangGraph ni LLM."""

import pytest
from pydantic import ValidationError

from agents import (
    AgentBudget,
    CriticVerdict,
    can_call_llm,
    can_retry,
    initial_state,
    llm_call_patch,
    llm_budget_left,
    retry_patch,
)
from models import AnswerIntent


# --------------------------------------------------------------------------- #
# AgentBudget
# --------------------------------------------------------------------------- #
def test_budget_defaults_match_the_brief() -> None:
    """Brief Phase 3 : retries du Critic bornees (1-2), timeout present."""
    budget = AgentBudget()
    assert budget.max_llm_calls > 0
    assert 1 <= budget.max_retries <= 2
    assert budget.llm_timeout_s >= 1


@pytest.mark.parametrize(
    "field, value",
    [
        ("max_llm_calls", 0),      # ni 0 ni negatif
        ("max_llm_calls", 51),     # au-dessus de la borne
        ("max_retries", -1),
        ("max_retries", 6),
        ("llm_timeout_s", 0),
    ],
)
def test_budget_rejects_out_of_range_values(field: str, value: int) -> None:
    with pytest.raises(ValidationError):
        AgentBudget(**{field: value})


def test_budget_forbids_extra_fields() -> None:
    """Modele de contrat : extra=forbid, comme les autres modeles du projet."""
    with pytest.raises(ValidationError):
        AgentBudget(max_llm_calls=6, anything=1)


# --------------------------------------------------------------------------- #
# initial_state
# --------------------------------------------------------------------------- #
def test_initial_state_defaults() -> None:
    state = initial_state("pourquoi ? httpie/cli::httpie/ssl_.py", "httpie/cli",
                          route=AnswerIntent.CHANGE_IMPACT)
    assert state["question"].startswith("pourquoi ?")
    assert state["repo_id"] == "httpie/cli"
    assert state["route"] == AnswerIntent.CHANGE_IMPACT
    assert state["llm_calls"] == 0
    assert state["retries"] == 0
    assert state["evidence_path"] is None
    assert state["evidences"] == []
    assert state["draft_answer"] == ""
    assert state["critic_verdict"] is None
    assert state["critic_notes"] == []
    assert state["answer"] is None
    assert isinstance(state["budget"], AgentBudget)


def test_initial_state_accepts_a_custom_budget() -> None:
    budget = AgentBudget(max_llm_calls=3, max_retries=1)
    state = initial_state("q", "httpie/cli", route=AnswerIntent.ROOT_CAUSE, budget=budget)
    assert state["budget"] is budget


def test_state_keys_are_stable() -> None:
    """LangGraph joint les chunk par cle : le schema ne doit pas derouter."""
    state = initial_state("q", "httpie/cli", route=AnswerIntent.GENERAL)
    assert set(state) == {
        "question", "repo_id", "route", "budget",
        "llm_calls", "retries",
        "evidence_path", "evidences",
        "draft_answer", "critic_verdict", "critic_notes",
        "answer",
    }


# --------------------------------------------------------------------------- #
# Budget LLM : purement fonctionnel (pas de mutation in place)
# --------------------------------------------------------------------------- #
def test_llm_budget_left_cannot_go_negative() -> None:
    state = initial_state("q", "httpie/cli", route=AnswerIntent.GENERAL,
                          budget=AgentBudget(max_llm_calls=2))
    assert llm_budget_left(state) == 2
    spent = {**state, **llm_call_patch(state)}
    spent = {**spent, **llm_call_patch(spent)}
    assert llm_budget_left(spent) == 0
    assert not can_call_llm(spent)
    # l'etat initial n'a pas ete mute
    assert state["llm_calls"] == 0
    assert can_call_llm(state)


def test_llm_call_patch_refuses_after_budget_is_spent() -> None:
    state = initial_state("q", "httpie/cli", route=AnswerIntent.GENERAL,
                          budget=AgentBudget(max_llm_calls=1))
    spent = {**state, **llm_call_patch(state)}
    with pytest.raises(RuntimeError):
        llm_call_patch(spent)


# --------------------------------------------------------------------------- #
# Retries du Critic
# --------------------------------------------------------------------------- #
def test_retry_uses_the_budget_limit() -> None:
    state = initial_state("q", "httpie/cli", route=AnswerIntent.GENERAL,
                          budget=AgentBudget(max_retries=2))
    assert can_retry(state)
    state = {**state, **retry_patch(state)}
    assert state["retries"] == 1
    assert can_retry(state)
    state = {**state, **retry_patch(state)}
    assert state["retries"] == 2
    assert not can_retry(state)
    with pytest.raises(RuntimeError):
        retry_patch(state)


def test_critic_verdict_values() -> None:
    """Les trois decisions du brief : accept / retry / escalate."""
    assert {v.value for v in CriticVerdict} == {"accept", "retry", "escalate"}
