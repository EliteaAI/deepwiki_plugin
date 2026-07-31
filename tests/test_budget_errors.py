import pytest

from plugin_implementation.budget_errors import (
    BUDGET_MESSAGES,
    BudgetExceededError,
    budget_error_result,
    budget_exceeded_from,
    raise_if_budget_exceeded,
)
from plugin_implementation.unified_db import UnifiedWikiDB
from plugin_implementation.wiki_structure_planner.cluster_planner import (
    ClusterStructurePlanner,
)


class FakeProviderError(Exception):
    def __init__(self, body):
        super().__init__(f"Error code: 400 - {body}")
        self.body = body


def test_detects_wrapped_member_budget_body():
    error = FakeProviderError({
        "error": {
            "message": "The budget for shared models has been reached.",
            "type": "budget_exceeded",
            "code": "member_budget_exceeded",
        }
    })

    detected = budget_exceeded_from(error)

    assert detected.scope == "member_budget_exceeded"
    assert str(detected) == BUDGET_MESSAGES["member_budget_exceeded"]


def test_detects_budget_error_through_exception_chain():
    provider_error = FakeProviderError({
        "type": "budget_exceeded",
        "code": "project_budget_exceeded",
    })
    try:
        raise RuntimeError("model invocation failed") from provider_error
    except RuntimeError as wrapped:
        detected = budget_exceeded_from(wrapped)

    assert detected.scope == "project_budget_exceeded"


def test_raise_if_budget_exceeded_ignores_ordinary_errors():
    raise_if_budget_exceeded(RuntimeError("temporary model failure"))


def test_result_contract_is_friendly_and_machine_readable():
    result = budget_error_result({
        "error_category": "budget_exceeded",
        "budget_error_code": "project_budget_exceeded",
        "error": "raw provider detail",
    })

    assert result == {
        "success": False,
        "error": BUDGET_MESSAGES["project_budget_exceeded"],
        "error_type": "BudgetExceededError",
        "error_category": "budget_exceeded",
        "budget_error_code": "project_budget_exceeded",
    }


def test_existing_budget_error_is_preserved():
    original = BudgetExceededError("member_budget_exceeded", "provider detail")

    assert budget_exceeded_from(original) is original

    try:
        raise_if_budget_exceeded(original)
    except BudgetExceededError as caught:
        assert caught is original
        assert caught.__cause__ is None


def test_embedding_population_stops_on_first_budget_rejection(tmp_path):
    calls = 0
    error = FakeProviderError({
        "error": {
            "type": "budget_exceeded",
            "code": "project_budget_exceeded",
        }
    })

    def reject_budget(_texts):
        nonlocal calls
        calls += 1
        raise error

    with UnifiedWikiDB(tmp_path / "wiki.db") as database:
        database.conn.executemany(
            "INSERT INTO repo_nodes (node_id, source_text) VALUES (?, ?)",
            [("one", "first source"), ("two", "second source")],
        )
        # The rejection happens before vector storage, so this isolates the
        # batching control flow even when sqlite-vec is unavailable in CI.
        database._vec_available = True

        try:
            database.populate_embeddings(reject_budget, batch_size=1)
        except BudgetExceededError as caught:
            assert caught.scope == "project_budget_exceeded"
        else:
            raise AssertionError("budget rejection should terminate embedding population")

    assert calls == 1


def test_cluster_planner_does_not_fall_back_after_budget_rejection():
    error = FakeProviderError({
        "type": "budget_exceeded",
        "code": "member_budget_exceeded",
    })

    class RejectingLlm:
        def invoke(self, _messages):
            raise error

    planner = ClusterStructurePlanner.__new__(ClusterStructurePlanner)
    planner.llm_low = RejectingLlm()
    planner._get_page_symbols = lambda _node_ids: []

    with pytest.raises(BudgetExceededError) as caught:
        planner._batched_macro_naming(
            macro_id=1,
            micro_map={1: ["node-1"]},
            node_count=1,
            file_count=1,
        )

    assert caught.value.scope == "member_budget_exceeded"
