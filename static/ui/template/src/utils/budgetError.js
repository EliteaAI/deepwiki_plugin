export const BUDGET_ERROR_MESSAGES = {
  project_budget_exceeded:
    "This project's budget has been reached. AI requests are unavailable until the budget resets or a project admin increases the limit.",
  member_budget_exceeded:
    'Your budget for this project has been reached. Your AI requests are unavailable until the budget resets or a project admin increases your limit.',
};

const asObject = (value) => {
  if (value && typeof value === 'object') return value;
  if (typeof value !== 'string') return null;
  try {
    const parsed = JSON.parse(value);
    return parsed && typeof parsed === 'object' ? parsed : null;
  } catch {
    return null;
  }
};

export const resolveBudgetError = (...values) => {
  let combinedText = '';
  let scope = null;
  let matched = false;

  values.forEach((value) => {
    const objectValue = asObject(value);
    const detail = objectValue?.error && typeof objectValue.error === 'object'
      ? objectValue.error
      : objectValue;

    if (detail) {
      const category = detail.error_category || detail.type;
      if (category === 'budget_exceeded') matched = true;
      const candidateScope = detail.budget_error_code || detail.code;
      if (
        candidateScope === 'member_budget_exceeded'
        || candidateScope === 'project_budget_exceeded'
      ) {
        scope = scope || candidateScope;
      }
    }

    if (typeof value === 'string') combinedText += ` ${value}`;
    else if (value) combinedText += ` ${JSON.stringify(value)}`;
  });

  if (combinedText.includes('budget_exceeded')) matched = true;
  if (combinedText.includes('The budget for shared models has been reached')) matched = true;
  if (combinedText.includes('member_budget_exceeded')) scope = 'member_budget_exceeded';
  if (combinedText.includes('project_budget_exceeded')) scope = scope || 'project_budget_exceeded';

  if (!matched) return null;
  const normalizedScope = scope === 'member_budget_exceeded'
    ? scope
    : 'project_budget_exceeded';
  return {
    code: normalizedScope,
    message: BUDGET_ERROR_MESSAGES[normalizedScope],
  };
};
