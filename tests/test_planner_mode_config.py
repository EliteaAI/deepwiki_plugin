"""Tests for ``OptimizedWikiAgent._resolve_planner_choice``.

The planner choice is configuration-driven: it flows from the UI
(``tool_params.planner_type``) through the worker payload, gets bridged
through the ``DEEPWIKI_STRUCTURE_PLANNER`` env var by
``wiki_subprocess_worker``, and finally lands on the agent's
``RunnableConfig.configurable``. The resolver normalises the value to one
of ``"agent"`` / ``"cluster"`` / ``"auto"``.
"""

from __future__ import annotations

import os
from unittest.mock import patch

import pytest

from plugin_implementation.agents.wiki_graph_optimized import OptimizedWikiGenerationAgent


@pytest.fixture()
def agent():
    """Construct a minimal agent instance without invoking heavy deps.

    ``_resolve_planner_choice`` only reads ``self`` for method binding, so
    we can avoid the real ``__init__`` machinery via ``__new__``.
    """
    return OptimizedWikiGenerationAgent.__new__(OptimizedWikiGenerationAgent)


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("agent", "agent"),
        ("Agentic", "agent"),
        ("DEEPAGENTS", "agent"),
        ("cluster", "cluster"),
        ("Cluster", "cluster"),
        ("auto", "auto"),
        ("", "auto"),
        ("unknown_value", "auto"),
    ],
)
def test_resolve_from_configurable_planner_mode(agent, raw, expected):
    config = {"configurable": {"planner_mode": raw}}
    assert agent._resolve_planner_choice(config) == expected


def test_planner_type_alias_takes_precedence_over_env(agent):
    """``planner_type`` is an alias of ``planner_mode`` in the configurable."""
    config = {"configurable": {"planner_type": "cluster"}}
    with patch.dict(os.environ, {"DEEPWIKI_STRUCTURE_PLANNER": "agent"}):
        assert agent._resolve_planner_choice(config) == "cluster"


def test_env_var_used_when_configurable_missing(agent):
    """When the configurable does not specify a planner, the env-bridge
    set by the subprocess worker is used.
    """
    with patch.dict(os.environ, {"DEEPWIKI_STRUCTURE_PLANNER": "cluster"}):
        assert agent._resolve_planner_choice(None) == "cluster"


def test_returns_auto_when_nothing_set(agent):
    with patch.dict(os.environ, {}, clear=False):
        os.environ.pop("DEEPWIKI_STRUCTURE_PLANNER", None)
        assert agent._resolve_planner_choice(None) == "auto"
        assert agent._resolve_planner_choice({}) == "auto"
        assert agent._resolve_planner_choice({"configurable": {}}) == "auto"
