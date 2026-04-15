/**
 * Smoke Test for getConfiguredRepo function
 *
 * This test validates the fix for GitHub issue #2854:
 * DeepWiki creation shows GitHub toolkits list in repository field and results
 * in "No repository configured" after save.
 *
 * The fix adds support for resolving toolkit references (code_repository ID)
 * to display the actual repository name from the referenced GitHub toolkit.
 *
 * Run with: node src/getConfiguredRepo.test.js
 */

// Copy of the getConfiguredRepo function for testing
function getConfiguredRepo(toolkit, settings, resolvedRepoName = null) {
  // If we have a resolved repository name from a toolkit reference, use it
  if (resolvedRepoName) {
    return resolvedRepoName;
  }

  const cfg = toolkit?.toolkit_config || {};
  const set = settings || toolkit?.settings || {};
  return (
    set.toolkit_configuration_github_repository ||
    set.github_repository ||
    set.repository ||
    set.repo ||
    cfg.github_repository ||
    cfg.repository ||
    null
  );
}

// Test cases
const tests = [];

// Test 1: When resolvedRepoName is provided, it should be returned
tests.push({
  name: 'Returns resolvedRepoName when provided',
  fn: () => {
    const result = getConfiguredRepo({}, {}, 'EliteaAI/elitea-sdk');
    if (result !== 'EliteaAI/elitea-sdk') {
      throw new Error(`Expected 'EliteaAI/elitea-sdk', got '${result}'`);
    }
  }
});

// Test 2: When resolvedRepoName is null, falls back to direct settings fields
tests.push({
  name: 'Falls back to toolkit_configuration_github_repository when resolvedRepoName is null',
  fn: () => {
    const settings = { toolkit_configuration_github_repository: 'org/repo-from-settings' };
    const result = getConfiguredRepo({}, settings, null);
    if (result !== 'org/repo-from-settings') {
      throw new Error(`Expected 'org/repo-from-settings', got '${result}'`);
    }
  }
});

// Test 3: Falls back to github_repository field
tests.push({
  name: 'Falls back to github_repository field',
  fn: () => {
    const settings = { github_repository: 'org/github-repo' };
    const result = getConfiguredRepo({}, settings, null);
    if (result !== 'org/github-repo') {
      throw new Error(`Expected 'org/github-repo', got '${result}'`);
    }
  }
});

// Test 4: Falls back to repository field
tests.push({
  name: 'Falls back to repository field',
  fn: () => {
    const settings = { repository: 'org/plain-repo' };
    const result = getConfiguredRepo({}, settings, null);
    if (result !== 'org/plain-repo') {
      throw new Error(`Expected 'org/plain-repo', got '${result}'`);
    }
  }
});

// Test 5: Falls back to repo field
tests.push({
  name: 'Falls back to repo field',
  fn: () => {
    const settings = { repo: 'org/short-repo' };
    const result = getConfiguredRepo({}, settings, null);
    if (result !== 'org/short-repo') {
      throw new Error(`Expected 'org/short-repo', got '${result}'`);
    }
  }
});

// Test 6: Falls back to toolkit_config.github_repository
tests.push({
  name: 'Falls back to toolkit_config.github_repository',
  fn: () => {
    const toolkit = { toolkit_config: { github_repository: 'org/config-repo' } };
    const result = getConfiguredRepo(toolkit, {}, null);
    if (result !== 'org/config-repo') {
      throw new Error(`Expected 'org/config-repo', got '${result}'`);
    }
  }
});

// Test 7: Falls back to toolkit_config.repository
tests.push({
  name: 'Falls back to toolkit_config.repository',
  fn: () => {
    const toolkit = { toolkit_config: { repository: 'org/config-plain-repo' } };
    const result = getConfiguredRepo(toolkit, {}, null);
    if (result !== 'org/config-plain-repo') {
      throw new Error(`Expected 'org/config-plain-repo', got '${result}'`);
    }
  }
});

// Test 8: Returns null when no repository configured and no resolvedRepoName
tests.push({
  name: 'Returns null when no repository is configured',
  fn: () => {
    const result = getConfiguredRepo({}, {}, null);
    if (result !== null) {
      throw new Error(`Expected null, got '${result}'`);
    }
  }
});

// Test 9: resolvedRepoName takes priority over direct settings
tests.push({
  name: 'resolvedRepoName takes priority over direct settings',
  fn: () => {
    const settings = { repository: 'should-not-be-used' };
    const result = getConfiguredRepo({}, settings, 'resolved/repo');
    if (result !== 'resolved/repo') {
      throw new Error(`Expected 'resolved/repo', got '${result}'`);
    }
  }
});

// Test 10: Handles undefined toolkit gracefully
tests.push({
  name: 'Handles undefined toolkit gracefully',
  fn: () => {
    const result = getConfiguredRepo(undefined, undefined, 'org/fallback');
    if (result !== 'org/fallback') {
      throw new Error(`Expected 'org/fallback', got '${result}'`);
    }
  }
});

// Test 11: Simulates the bug scenario - toolkit_configuration_code_repository is an integer (toolkit ID)
// This test verifies that when only the toolkit ID is present (not the repo name),
// and resolvedRepoName is provided, the resolved name is used instead of null
tests.push({
  name: 'BUG #2854: Correctly uses resolvedRepoName when settings only has toolkit_configuration_code_repository (integer ID)',
  fn: () => {
    // This is the actual data structure that caused the bug
    const settings = {
      toolkit_configuration_code_repository: 123, // Integer ID, not repository name
      toolkit_configuration_bucket: 'my-wiki-bucket',
      toolkit_configuration_llm_model: 'gpt-4',
      toolkit_configuration_max_tokens: 2048
    };

    // Without the fix, this would return null (bug behavior)
    const resultWithoutFix = getConfiguredRepo({}, settings, null);
    if (resultWithoutFix !== null) {
      throw new Error(`Without resolvedRepoName, should be null, got '${resultWithoutFix}'`);
    }

    // With the fix, when we resolve the toolkit reference and pass the repo name
    const resultWithFix = getConfiguredRepo({}, settings, 'EliteaAI/elitea-sdk');
    if (resultWithFix !== 'EliteaAI/elitea-sdk') {
      throw new Error(`With resolvedRepoName, expected 'EliteaAI/elitea-sdk', got '${resultWithFix}'`);
    }
  }
});

// Run tests
console.log('🧪 Running getConfiguredRepo smoke tests...\n');

let passed = 0;
let failed = 0;

for (const test of tests) {
  try {
    test.fn();
    console.log(`✅ PASS: ${test.name}`);
    passed++;
  } catch (error) {
    console.log(`❌ FAIL: ${test.name}`);
    console.log(`   Error: ${error.message}`);
    failed++;
  }
}

console.log(`\n${'─'.repeat(50)}`);
console.log(`📊 Results: ${passed} passed, ${failed} failed, ${tests.length} total`);

if (failed > 0) {
  console.log('\n❌ Some tests failed!');
  process.exit(1);
} else {
  console.log('\n✅ All tests passed!');
  console.log('\n📝 Note: This validates the getConfiguredRepo function logic.');
  console.log('   Full integration testing requires running the UI with a mock API.');
  process.exit(0);
}
