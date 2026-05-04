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

function extractAdoOrganization(adoConfig) {
  const organizationUrl = adoConfig?.organization_url || adoConfig?.url;
  if (!organizationUrl || typeof organizationUrl !== 'string') return null;

  try {
    const parsedUrl = new URL(organizationUrl);
    if (parsedUrl.hostname === 'dev.azure.com') {
      return parsedUrl.pathname.split('/').filter(Boolean)[0] || null;
    }
    if (parsedUrl.hostname.endsWith('.visualstudio.com')) {
      return parsedUrl.hostname.split('.')[0] || null;
    }
  } catch (e) {
    return null;
  }

  return null;
}

function mergePlainObjects(...values) {
  return values.reduce((merged, value) => {
    if (!value || typeof value !== 'object' || Array.isArray(value)) return merged;
    return { ...merged, ...value };
  }, {});
}

function getAdoConfig(settings) {
  if (!settings || typeof settings !== 'object') return {};
  return mergePlainObjects(settings.ado_configuration, settings.toolkit_configuration_ado_configuration);
}

function getToolkitConfigPayload(toolkit) {
  if (!toolkit || typeof toolkit !== 'object') return {};
  const configurationParameters = toolkit.configuration?.parameters;
  return mergePlainObjects(toolkit.toolkit_config, configurationParameters);
}

function mergeToolkitIdentitySettings(settings, toolkitConfig) {
  const merged = mergePlainObjects(toolkitConfig, settings);
  const adoConfig = mergePlainObjects(getAdoConfig(toolkitConfig), getAdoConfig(settings));
  if (Object.keys(adoConfig).length > 0) {
    merged.ado_configuration = adoConfig;
    merged.toolkit_configuration_ado_configuration = adoConfig;
  }
  return merged;
}

function buildAdoRepositoryIdentifier(settings) {
  if (!settings || typeof settings !== 'object') return null;

  const adoConfig = getAdoConfig(settings);
  const repository = settings.toolkit_configuration_repository_id || settings.repository_id || settings.repository || settings.repo;
  const project = adoConfig.project || settings.toolkit_configuration_project || settings.project;
  const organization = adoConfig.organization || settings.organization || extractAdoOrganization(adoConfig);

  if (organization && project && repository) {
    return `${organization}/${project}/${repository}`;
  }

  return repository || null;
}

function normalizeWikiIdPart(value) {
  if (!value || typeof value !== 'string') return null;
  return value
    .toLowerCase()
    .replace(/[^a-z0-9-]/g, '-')
    .replace(/-+/g, '-')
    .replace(/^-|-$/g, '') || null;
}

function parseRepositoryIdentity(value) {
  if (!value) return { repository: null, branch: null };

  if (typeof value === 'object') {
    const nested = parseRepositoryIdentity(
      value.repository || value.repo || value.canonical_repo_identifier || value.repo_identifier || value.identifier
    );
    return {
      repository: nested.repository,
      branch: value.branch || value.active_branch || value.base_branch || nested.branch || null,
    };
  }

  if (typeof value !== 'string') return { repository: null, branch: null };

  let repoPath = value.trim();
  let branch = null;

  if (!repoPath) return { repository: null, branch: null };

  if (repoPath.startsWith('http://') || repoPath.startsWith('https://')) {
    try {
      const url = new URL(repoPath);
      const hostname = url.hostname.toLowerCase();
      repoPath = url.pathname.replace(/^\/+|\/+$/g, '');
      if (repoPath.includes('/_git/')) {
        const [projectPath, repoName] = repoPath.split('/_git/');
        if (hostname.endsWith('.visualstudio.com')) {
          const organization = hostname.split('.')[0];
          repoPath = [organization, projectPath, repoName].filter(Boolean).join('/');
        } else {
          repoPath = `${projectPath}/${repoName}`;
        }
      }
    } catch {
      return { repository: null, branch: null };
    }
  } else if (repoPath.startsWith('git@') && repoPath.includes(':')) {
    repoPath = repoPath.split(':').slice(1).join(':');
  } else if (repoPath.includes(':')) {
    const parts = repoPath.split(':');
    repoPath = parts[0];
    branch = parts[1] || null;
  }

  repoPath = repoPath.replace(/\.git$/, '').replace(/^\/+|\/+$/g, '');

  return {
    repository: repoPath || null,
    branch,
  };
}

function normalizeRepoToWikiIdPrefix(repoIdentity) {
  return getRepositoryMatchInfo(repoIdentity).prefix;
}

function getRepositoryMatchInfo(repoIdentity) {
  const parsed = parseRepositoryIdentity(repoIdentity);
  if (!parsed.repository) {
    return {
      parsed,
      prefix: null,
      repositoryLeaf: null,
      branchPart: null,
      leafBranchSuffix: null,
      hasBranch: false,
    };
  }

  const parts = parsed.repository.split('/').filter(Boolean).map(normalizeWikiIdPart).filter(Boolean);
  const branchPart = normalizeWikiIdPart(parsed.branch);
  const repositoryLeaf = parts.length > 0 ? parts[parts.length - 1] : null;

  return {
    parsed,
    prefix: parts.length > 0 ? (branchPart ? [...parts, branchPart].join('--') : parts.join('--')) : null,
    repositoryLeaf,
    branchPart,
    leafBranchSuffix: repositoryLeaf && branchPart ? `${repositoryLeaf}--${branchPart}` : null,
    hasBranch: Boolean(parsed.branch),
  };
}

function getBranchFromSettings(settings) {
  if (!settings || typeof settings !== 'object') return null;

  return (
    settings.active_branch ||
    settings.toolkit_configuration_active_branch ||
    settings.base_branch ||
    settings.toolkit_configuration_base_branch ||
    settings.github_base_branch ||
    settings.toolkit_configuration_github_base_branch ||
    settings.branch ||
    settings.toolkit_configuration_branch ||
    null
  );
}

function getRepositoryFromSettings(settings) {
  if (!settings || typeof settings !== 'object') return null;

  return (
    settings.toolkit_configuration_github_repository ||
    settings.github_repository ||
    buildAdoRepositoryIdentifier(settings) ||
    settings.repository ||
    settings.repo ||
    null
  );
}

function getCodeToolkitReference(settings) {
  if (!settings || typeof settings !== 'object') return null;

  return (
    settings.toolkit_configuration_code_toolkit ||
    settings.code_toolkit ||
    settings.toolkit_configuration_code_repository ||
    settings.code_repository ||
    null
  );
}

function getConfiguredRepoIdentity(toolkit, settings, resolvedRepoIdentity = null) {
  if (resolvedRepoIdentity) {
    const resolved = parseRepositoryIdentity(resolvedRepoIdentity);
    if (resolved.repository) return resolved;
  }

  const cfg = getToolkitConfigPayload(toolkit);
  const set = settings || toolkit?.settings || {};
  const merged = mergeToolkitIdentitySettings(set, cfg);
  const repository = getRepositoryFromSettings(merged) || getRepositoryFromSettings(set) || getRepositoryFromSettings(cfg);
  const parsed = parseRepositoryIdentity(repository);

  if (!parsed.repository) return null;

  return {
    repository: parsed.repository,
    branch: getBranchFromSettings(merged) || getBranchFromSettings(set) || getBranchFromSettings(cfg) || parsed.branch || null,
  };
}

function manifestMatchesRepo(manifest, configuredRepoIdentity) {
  if (!manifest || !configuredRepoIdentity) return false;

  const expectedInfo = getRepositoryMatchInfo(configuredRepoIdentity);
  const expectedPrefix = expectedInfo.prefix;
  const matchesPrefix = prefix => {
    if (!prefix) return false;
    const normalizedPrefix = prefix.toLowerCase();
    if (normalizedPrefix === expectedPrefix) return true;
    return !expectedInfo.hasBranch && normalizedPrefix.startsWith(`${expectedPrefix}--`);
  };

  const matchedByPrefix = [
    typeof manifest.wiki_id === 'string' ? manifest.wiki_id : null,
    normalizeRepoToWikiIdPrefix({ repository: manifest.repository, branch: manifest.branch }),
    normalizeRepoToWikiIdPrefix(manifest.canonical_repo_identifier),
  ].some(matchesPrefix);

  return matchedByPrefix;
}

function repositoryLeafAndBranchMatch(candidateRepoIdentity, expectedInfo) {
  const candidateInfo = getRepositoryMatchInfo(candidateRepoIdentity);
  if (!candidateInfo.repositoryLeaf || !expectedInfo.repositoryLeaf) return false;
  if (candidateInfo.repositoryLeaf !== expectedInfo.repositoryLeaf) return false;
  if (expectedInfo.branchPart && candidateInfo.branchPart !== expectedInfo.branchPart) return false;
  return true;
}

function manifestRepoMatchKey(manifest) {
  const candidates = [
    manifest?.canonical_repo_identifier,
    { repository: manifest?.repository, branch: manifest?.branch },
  ];

  for (const candidate of candidates) {
    const info = getRepositoryMatchInfo(candidate);
    if (info.parsed.repository && info.branchPart) {
      return `${info.parsed.repository.toLowerCase()}:${info.branchPart}`;
    }
  }

  return null;
}

function filterManifestsByRepo(manifests, configuredRepoIdentity) {
  const strictMatches = manifests.filter(manifest => manifestMatchesRepo(manifest, configuredRepoIdentity));
  if (strictMatches.length > 0) return strictMatches;

  const expectedInfo = getRepositoryMatchInfo(configuredRepoIdentity);
  const repoParts = expectedInfo.parsed.repository?.split('/').filter(Boolean) || [];
  if (repoParts.length !== 1 || !expectedInfo.leafBranchSuffix) return [];

  const leafMatches = manifests.filter(manifest => [
    { repository: manifest.repository, branch: manifest.branch },
    manifest.canonical_repo_identifier,
  ].some(candidate => repositoryLeafAndBranchMatch(candidate, expectedInfo)));

  const canonicalKeys = new Set(leafMatches.map(manifestRepoMatchKey).filter(Boolean));
  return canonicalKeys.size === 1 ? leafMatches : [];
}

function artifactMatchesRepo(artifactName, configuredRepoIdentity) {
  if (!artifactName || !configuredRepoIdentity) return false;

  const expectedInfo = getRepositoryMatchInfo(configuredRepoIdentity);
  const expectedPrefix = expectedInfo.prefix;
  const normalizedName = artifactName.toLowerCase();
  const artifactPrefix = normalizedName.split('/')[0];

  if (expectedPrefix) {
    if (artifactPrefix === expectedPrefix || normalizedName.startsWith(`${expectedPrefix}/`)) return true;
    return !expectedInfo.hasBranch && artifactPrefix.startsWith(`${expectedPrefix}--`);
  }

  return false;
}

// Copy of the getConfiguredRepo function for testing
function getConfiguredRepo(toolkit, settings, resolvedRepoName = null) {
  // If we have a resolved repository name from a toolkit reference, use it
  const resolved = parseRepositoryIdentity(resolvedRepoName);
  if (resolved.repository) {
    return resolved.repository;
  }

  return getConfiguredRepoIdentity(toolkit, settings, null)?.repository || null;
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

tests.push({
  name: 'Resolves legacy toolkit_configuration_code_repository reference field',
  fn: () => {
    const settings = {
      toolkit_configuration_code_repository: 321,
      toolkit_configuration_bucket: 'wiki-artifacts'
    };

    const result = getCodeToolkitReference(settings);
    if (result !== 321) {
      throw new Error(`Expected legacy code repository reference 321, got '${result}'`);
    }
  }
});

tests.push({
  name: 'Resolves legacy code_repository reference field',
  fn: () => {
    const settings = {
      code_repository: '654',
      bucket: 'wiki-artifacts'
    };

    const result = getCodeToolkitReference(settings);
    if (result !== '654') {
      throw new Error(`Expected legacy code repository reference '654', got '${result}'`);
    }
  }
});

tests.push({
  name: 'Builds ADO repository identifier from org/project/repository settings',
  fn: () => {
    const settings = {
      ado_configuration: {
        organization_url: 'https://dev.azure.com/epameliteatest/',
        project: 'TestProject'
      },
      repository_id: 'mb-java'
    };

    const result = getConfiguredRepo({}, settings, null);
    if (result !== 'epameliteatest/TestProject/mb-java') {
      throw new Error(`Expected 'epameliteatest/TestProject/mb-java', got '${result}'`);
    }
  }
});

tests.push({
  name: 'Builds ADO repository identifier from transformed toolkit_configuration fields',
  fn: () => {
    const settings = {
      toolkit_configuration_ado_configuration: {
        organization_url: 'https://dev.azure.com/epameliteatest/',
        project: 'TestProject'
      },
      toolkit_configuration_repository_id: 'mb-java'
    };

    const result = getConfiguredRepo({}, settings, null);
    if (result !== 'epameliteatest/TestProject/mb-java') {
      throw new Error(`Expected 'epameliteatest/TestProject/mb-java', got '${result}'`);
    }
  }
});

tests.push({
  name: 'Builds branch-aware ADO configured repo identity',
  fn: () => {
    const settings = {
      toolkit_configuration_ado_configuration: {
        organization_url: 'https://dev.azure.com/epameliteatest/',
        project: 'TestProject'
      },
      toolkit_configuration_repository_id: 'TestProject',
      toolkit_configuration_base_branch: 'mb-java'
    };

    const identity = getConfiguredRepoIdentity({}, settings, null);
    if (identity.repository !== 'epameliteatest/TestProject/TestProject') {
      throw new Error(`Expected repository 'epameliteatest/TestProject/TestProject', got '${identity.repository}'`);
    }
    if (identity.branch !== 'mb-java') {
      throw new Error(`Expected branch 'mb-java', got '${identity.branch}'`);
    }

    const prefix = normalizeRepoToWikiIdPrefix(identity);
    if (prefix !== 'epameliteatest--testproject--testproject--mb-java') {
      throw new Error(`Expected wiki_id prefix 'epameliteatest--testproject--testproject--mb-java', got '${prefix}'`);
    }
  }
});

tests.push({
  name: 'Builds ADO identity when repository and provider config are split across settings and toolkit_config',
  fn: () => {
    const toolkit = {
      toolkit_config: {
        ado_configuration: {
          organization_url: 'https://dev.azure.com/epameliteatest/',
          project: 'TestProject',
        },
      },
    };
    const settings = {
      repository_id: 'TestProject',
      active_branch: 'mb-java',
      ado_configuration: {
        elitea_title: 'Ado_test',
        private: true,
      },
    };

    const identity = getConfiguredRepoIdentity(toolkit, settings, null);
    if (identity.repository !== 'epameliteatest/TestProject/TestProject') {
      throw new Error(`Expected merged ADO repository identity, got '${identity.repository}'`);
    }
    if (normalizeRepoToWikiIdPrefix(identity) !== 'epameliteatest--testproject--testproject--mb-java') {
      throw new Error(`Expected full ADO wiki prefix, got '${normalizeRepoToWikiIdPrefix(identity)}'`);
    }
  }
});

tests.push({
  name: 'Matches ADO manifest by wiki_id/repository/branch',
  fn: () => {
    const manifest = {
      wiki_id: 'epameliteatest--testproject--testproject--mb-java',
      canonical_repo_identifier: 'epameliteatest/TestProject/TestProject:mb-java:6f785643',
      repository: 'epameliteatest/TestProject/TestProject',
      branch: 'mb-java'
    };
    const identity = {
      repository: 'epameliteatest/TestProject/TestProject',
      branch: 'mb-java'
    };

    if (!manifestMatchesRepo(manifest, identity)) {
      throw new Error('Expected manifest to match ADO configured repo identity');
    }
  }
});

tests.push({
  name: 'Matches ADO manifest by repository and branch when wiki_id is missing',
  fn: () => {
    const manifest = {
      repository: 'epameliteatest/TestProject/TestProject',
      branch: 'mb-java'
    };
    const identity = {
      repository: 'epameliteatest/TestProject/TestProject',
      branch: 'mb-java'
    };

    if (!manifestMatchesRepo(manifest, identity)) {
      throw new Error('Expected manifest repository/branch to match ADO configured repo identity');
    }
  }
});

tests.push({
  name: 'Does not match unrelated ADO manifest versions from shared bucket',
  fn: () => {
    const manifest = {
      wiki_id: 'eliteaai--alita-sdk--main',
      canonical_repo_identifier: 'EliteaAI/alita-sdk:main:923d6d27',
      repository: 'EliteaAI/alita-sdk',
      branch: 'main'
    };
    const identity = {
      repository: 'epameliteatest/TestProject/TestProject',
      branch: 'mb-java'
    };

    if (manifestMatchesRepo(manifest, identity)) {
      throw new Error('Expected unrelated manifest to be filtered out');
    }
  }
});

tests.push({
  name: 'Normalizes canonical repo identifier with branch and commit to wiki_id prefix',
  fn: () => {
    const prefix = normalizeRepoToWikiIdPrefix('epameliteatest/TestProject/TestProject:mb-java:6f785643');
    if (prefix !== 'epameliteatest--testproject--testproject--mb-java') {
      throw new Error(`Expected canonical prefix, got '${prefix}'`);
    }
  }
});

tests.push({
  name: 'Keeps single-segment ADO repo storage key without suffix-matching full manifests',
  fn: () => {
    const identity = getConfiguredRepoIdentity(
      {},
      {
        repository_id: 'TestProject',
        active_branch: 'mb-java',
        ado_configuration: {
          elitea_title: 'TestRepo',
          private: true,
        },
      },
      null
    );
    const manifest = {
      wiki_id: 'epameliteatest--testproject--testproject--mb-java',
      canonical_repo_identifier: 'epameliteatest/TestProject/TestProject:mb-java:6f785643',
      repository: 'epameliteatest/TestProject/TestProject',
      branch: 'mb-java',
    };

    const prefix = normalizeRepoToWikiIdPrefix(identity);
    if (prefix !== 'testproject--mb-java') {
      throw new Error(`Expected single-segment ADO storage prefix 'testproject--mb-java', got '${prefix}'`);
    }
    if (manifestMatchesRepo(manifest, identity)) {
      throw new Error('Expected leaf-only ADO identity not to match a full canonical manifest by suffix');
    }
  }
});

tests.push({
  name: 'Resolves a leaf-only ADO manifest only when the canonical match is unambiguous',
  fn: () => {
    const identity = {
      repository: 'TestProject',
      branch: 'mb-java',
    };
    const manifests = [
      {
        wiki_id: 'epameliteatest--testproject--testproject--mb-java',
        canonical_repo_identifier: 'epameliteatest/TestProject/TestProject:mb-java:6f785643',
        repository: 'epameliteatest/TestProject/TestProject',
        branch: 'mb-java',
      },
    ];

    const matches = filterManifestsByRepo(manifests, identity);
    if (matches.length !== 1 || matches[0] !== manifests[0]) {
      throw new Error(`Expected exactly one unambiguous leaf match, got ${matches.length}`);
    }
  }
});

tests.push({
  name: 'Rejects ambiguous leaf-only ADO manifest matches from shared buckets',
  fn: () => {
    const identity = {
      repository: 'service',
      branch: 'main',
    };
    const manifests = [
      {
        wiki_id: 'orga--proj1--service--main',
        canonical_repo_identifier: 'orgA/proj1/service:main:11111111',
        repository: 'orgA/proj1/service',
        branch: 'main',
      },
      {
        wiki_id: 'orgb--proj2--service--main',
        canonical_repo_identifier: 'orgB/proj2/service:main:22222222',
        repository: 'orgB/proj2/service',
        branch: 'main',
      },
    ];

    const matches = filterManifestsByRepo(manifests, identity);
    if (matches.length !== 0) {
      throw new Error(`Expected ambiguous leaf matches to be rejected, got ${matches.length}`);
    }
  }
});

tests.push({
  name: 'Does not suffix-match ADO artifact folders when full prefix is unavailable',
  fn: () => {
    const identity = {
      repository: 'TestProject',
      branch: 'mb-java',
    };
    const artifactName = 'epameliteatest--testproject--testproject--mb-java/wiki_pages/overview.md';

    if (artifactMatchesRepo(artifactName, identity)) {
      throw new Error('Expected full artifact path not to match a leaf-only repo by suffix');
    }
    if (!artifactMatchesRepo('testproject--mb-java/wiki_pages/overview.md', identity)) {
      throw new Error('Expected exact leaf-only artifact prefix to match');
    }
    if (artifactMatchesRepo('eliteaai--alita-sdk--main/wiki_pages/overview.md', identity)) {
      throw new Error('Expected unrelated artifact path to be filtered out');
    }
  }
});

tests.push({
  name: 'Does not leaf-match ADO manifest from another branch',
  fn: () => {
    const identity = {
      repository: 'TestProject',
      branch: 'mb-java',
    };
    const manifest = {
      wiki_id: 'epameliteatest--testproject--testproject--main',
      canonical_repo_identifier: 'epameliteatest/TestProject/TestProject:main:6f785643',
      repository: 'epameliteatest/TestProject/TestProject',
      branch: 'main',
    };

    if (manifestMatchesRepo(manifest, identity)) {
      throw new Error('Expected manifest from another branch to be filtered out');
    }
  }
});

tests.push({
  name: 'Preserves organization when parsing legacy visualstudio.com ADO repository URLs',
  fn: () => {
    const identity = parseRepositoryIdentity('https://epameliteatest.visualstudio.com/TestProject/_git/TestRepo');

    if (identity.repository !== 'epameliteatest/TestProject/TestRepo') {
      throw new Error(`Expected visualstudio repo path with org, got '${identity.repository}'`);
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
