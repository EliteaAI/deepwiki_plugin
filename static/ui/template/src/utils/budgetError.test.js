import assert from 'node:assert/strict';
import test from 'node:test';

import { BUDGET_ERROR_MESSAGES, resolveBudgetError } from './budgetError.js';

test('uses the member-specific actionable message', () => {
  const error = resolveBudgetError({
    error_category: 'budget_exceeded',
    budget_error_code: 'member_budget_exceeded',
  });

  assert.deepEqual(error, {
    code: 'member_budget_exceeded',
    message: BUDGET_ERROR_MESSAGES.member_budget_exceeded,
  });
});

test('recognizes the raw proxy error without showing technical details', () => {
  const error = resolveBudgetError(
    "Error code: 400 - {'error': {'message': 'The budget for shared models has been reached', 'type': 'budget_exceeded'}}",
  );

  assert.equal(error.message, BUDGET_ERROR_MESSAGES.project_budget_exceeded);
  assert.doesNotMatch(error.message, /400|shared models|budget_exceeded/);
});

test('ignores ordinary application token budgets', () => {
  assert.equal(resolveBudgetError('Token budget reached while formatting context'), null);
});

test('ignores unrelated error codes', () => {
  assert.equal(resolveBudgetError({ code: 'invalid_request' }), null);
});
