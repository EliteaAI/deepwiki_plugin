"""Terminal handling for platform budget rejections.

The platform proxy returns budget failures as ordinary HTTP 400 errors.  DeepWiki
normally treats model and embedding failures as recoverable, but retrying after a
budget rejection can never succeed until an administrator changes the limit or the
budget resets.  This module keeps the proxy-specific contract in one place.
"""

from __future__ import annotations

from typing import Any, Dict, Optional


BUDGET_ERROR_TYPE = "budget_exceeded"
DEFAULT_BUDGET_SCOPE = "project_budget_exceeded"
BUDGET_SCOPES = (DEFAULT_BUDGET_SCOPE, "member_budget_exceeded")

BUDGET_MESSAGES = {
    DEFAULT_BUDGET_SCOPE: (
        "This project's budget has been reached. AI requests are unavailable until "
        "the budget resets or a project admin increases the limit."
    ),
    "member_budget_exceeded": (
        "Your budget for this project has been reached. Your AI requests are unavailable "
        "until the budget resets or a project admin increases your limit."
    ),
}


class BudgetExceededError(Exception):
    """A non-retryable project or member budget rejection."""

    def __init__(self, scope: str = DEFAULT_BUDGET_SCOPE, provider_message: str = ""):
        self.scope = scope if scope in BUDGET_SCOPES else DEFAULT_BUDGET_SCOPE
        self.provider_message = provider_message
        super().__init__(BUDGET_MESSAGES[self.scope])


def _budget_detail(value: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(value, dict):
        return None

    detail = value.get("error") if isinstance(value.get("error"), dict) else value
    if detail.get("type") == BUDGET_ERROR_TYPE:
        return detail

    if value.get("error_category") == BUDGET_ERROR_TYPE:
        return {
            "type": BUDGET_ERROR_TYPE,
            "code": value.get("budget_error_code"),
            "message": value.get("error") or value.get("message"),
        }

    return None


def budget_exceeded_from(value: Any) -> Optional[BudgetExceededError]:
    """Return a canonical budget error when *value* contains the proxy contract."""
    if isinstance(value, BudgetExceededError):
        return value

    current = value
    seen = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))

        scope = getattr(current, "scope", None)
        if scope in BUDGET_SCOPES:
            return BudgetExceededError(scope, str(current))

        detail = _budget_detail(current)
        if detail is None:
            detail = _budget_detail(getattr(current, "body", None))
        if detail is None:
            for arg in getattr(current, "args", ()):
                detail = _budget_detail(arg)
                if detail is not None:
                    break

        if detail is not None:
            return BudgetExceededError(
                detail.get("code") or DEFAULT_BUDGET_SCOPE,
                str(detail.get("message") or current),
            )

        text = str(current)
        if BUDGET_ERROR_TYPE in text or any(code in text for code in BUDGET_SCOPES):
            detected_scope = next(
                (code for code in BUDGET_SCOPES if code in text),
                DEFAULT_BUDGET_SCOPE,
            )
            return BudgetExceededError(detected_scope, text)

        current = getattr(current, "__cause__", None) or getattr(current, "__context__", None)

    return None


def raise_if_budget_exceeded(exc: Exception) -> None:
    """Raise a canonical terminal error when *exc* is a budget rejection."""
    budget_error = budget_exceeded_from(exc)
    if budget_error is not None:
        if budget_error is exc:
            raise budget_error
        raise budget_error from exc


def budget_error_result(value: Any) -> Optional[Dict[str, Any]]:
    """Build the stable failed-result contract used by all DeepWiki workers."""
    budget_error = budget_exceeded_from(value)
    if budget_error is None:
        return None
    return {
        "success": False,
        "error": str(budget_error),
        "error_type": type(budget_error).__name__,
        "error_category": BUDGET_ERROR_TYPE,
        "budget_error_code": budget_error.scope,
    }
