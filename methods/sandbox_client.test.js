/**
 * Tests for EliteA Sandbox Client
 * Run with: node sandbox_client.test.js
 */

import { SandboxClient, SandboxArtifact, ApiDetailsRequestError } from './sandbox_client.js';

// Mock fetch for testing
let mockFetch;
let fetchCalls = [];

// Save original fetch
const originalFetch = global.fetch;

function setupMockFetch() {
    fetchCalls = [];
    mockFetch = async (url, options = {}) => {
        fetchCalls.push({ url, options });
        
        // Return mock responses based on URL patterns
        if (url.includes('/test_tool/') && options.method === 'POST') {
            const body = JSON.parse(options.body);
            if (url.includes('await_response=true')) {
                // Sync response
                return {
                    ok: true,
                    status: 200,
                    json: async () => ({ result: { success: true, output: 'test result' } })
                };
            } else {
                // Async response
                return {
                    ok: true,
                    status: 200,
                    json: async () => ({ task_id: 'test-task-123' })
                };
            }
        }
        
        if (url.includes('/task/') && options.method === 'GET') {
            // Task status response
            return {
                ok: true,
                status: 200,
                json: async () => ({
                    status: 'SUCCESS',
                    meta: { project_id: 1 },
                    result: { data: 'completed' }
                })
            };
        }
        
        if (url.includes('/task/') && options.method === 'DELETE') {
            // Task stop response
            return {
                ok: true,
                status: 204
            };
        }
        
        if (url.includes('/auth/user')) {
            return {
                ok: true,
                status: 200,
                json: async () => ({ id: 123, email: 'test@example.com' })
            };
        }
        
        if (url.includes('/mcp_sse/tools_list')) {
            return {
                ok: true,
                status: 200,
                json: async () => [
                    { name: 'tool1', type: 'function' },
                    { name: 'tool2', type: 'function' }
                ]
            };
        }
        
        if (url.includes('/mcp_sse/tools_call')) {
            return {
                ok: true,
                status: 200,
                json: async () => ({ result: 'mcp call result' })
            };
        }
        
        if (url.includes('/applications/applications')) {
            return {
                ok: true,
                status: 200,
                json: async () => ({
                    total: 2,
                    rows: [
                        { id: 1, name: 'App 1' },
                        { id: 2, name: 'App 2' }
                    ]
                })
            };
        }
        
        if (url.includes('/applications/application/')) {
            return {
                ok: true,
                status: 200,
                json: async () => ({ id: 1, name: 'Test App', versions: [] })
            };
        }
        
        if (url.includes('/buckets') && options.method === 'GET') {
            return {
                ok: true,
                status: 200,
                json: async () => ({
                    rows: [
                        { name: 'test-bucket' },
                        { name: 'other-bucket' }
                    ]
                })
            };
        }
        
        if (url.includes('/buckets') && options.method === 'POST') {
            return {
                ok: true,
                status: 200,
                json: async () => ({ success: true, name: 'test-bucket' })
            };
        }
        
        if (url.includes('/artifacts/artifacts/') && options.method === 'GET') {
            return {
                ok: true,
                status: 200,
                json: async () => ({
                    files: [
                        { name: 'file1.txt', size: 100 },
                        { name: 'file2.json', size: 200 }
                    ]
                })
            };
        }
        
        if (url.includes('/artifacts/artifacts/') && options.method === 'POST') {
            return {
                ok: true,
                status: 200,
                json: async () => ({ success: true, filename: 'test.txt' })
            };
        }
        
        if (url.includes('/artifacts/artifact/') && options.method === 'GET') {
            return {
                ok: true,
                status: 200,
                arrayBuffer: async () => new TextEncoder().encode('test file content').buffer
            };
        }
        
        if (url.includes('/artifacts/artifact/') && options.method === 'DELETE') {
            return {
                ok: true,
                status: 200,
                json: async () => ({ success: true })
            };
        }
        
        if (url.includes('/secrets/secret/')) {
            return {
                ok: true,
                status: 200,
                json: async () => ({ value: 'secret-value-123' })
            };
        }
        
        if (url.includes('/integrations/integrations/')) {
            return {
                ok: true,
                status: 200,
                json: async () => [
                    { id: 1, name: 'Config 1' },
                    { id: 2, name: 'Config 2' }
                ]
            };
        }
        
        // Default response
        return {
            ok: false,
            status: 404,
            json: async () => ({ error: 'Not found' })
        };
    };
    
    global.fetch = mockFetch;
}

function restoreFetch() {
    global.fetch = originalFetch;
}

// Test utilities
const colors = {
    reset: '\x1b[0m',
    green: '\x1b[32m',
    red: '\x1b[31m',
    yellow: '\x1b[33m',
    blue: '\x1b[34m'
};

function log(message, color = 'reset') {
    console.log(`${colors[color]}${message}${colors.reset}`);
}

function assert(condition, message) {
    if (!condition) {
        throw new Error(`Assertion failed: ${message}`);
    }
}

function assertEquals(actual, expected, message) {
    if (JSON.stringify(actual) !== JSON.stringify(expected)) {
        throw new Error(`${message}\nExpected: ${JSON.stringify(expected)}\nActual: ${JSON.stringify(actual)}`);
    }
}

function assertContains(str, substring, message) {
    if (!str.includes(substring)) {
        throw new Error(`${message}\nExpected "${str}" to contain "${substring}"`);
    }
}

// Test suite
const tests = [];
let passedTests = 0;
let failedTests = 0;

function test(name, fn) {
    tests.push({ name, fn });
}

async function runTests() {
    log('\n🧪 Running Sandbox Client Tests\n', 'blue');
    
    for (const { name, fn } of tests) {
        try {
            setupMockFetch();
            await fn();
            passedTests++;
            log(`✓ ${name}`, 'green');
        } catch (error) {
            failedTests++;
            log(`✗ ${name}`, 'red');
            log(`  ${error.message}`, 'red');
        } finally {
            restoreFetch();
        }
    }
    
    log(`\n${'='.repeat(50)}`, 'blue');
    log(`Tests: ${passedTests + failedTests}`, 'blue');
    log(`Passed: ${passedTests}`, 'green');
    log(`Failed: ${failedTests}`, failedTests > 0 ? 'red' : 'green');
    log('='.repeat(50) + '\n', 'blue');
    
    process.exit(failedTests > 0 ? 1 : 0);
}

// Tests

test('SandboxClient constructor initializes correctly', () => {
    const client = new SandboxClient({
        baseUrl: 'https://test.example.com',
        projectId: 123,
        authToken: 'test-token'
    });
    
    assert(client.baseUrl === 'https://test.example.com', 'baseUrl should be set');
    assert(client.projectId === 123, 'projectId should be set');
    assert(client.authToken === 'test-token', 'authToken should be set');
    assert(client.headers.Authorization === 'Bearer test-token', 'Authorization header should be set');
});

test('SandboxClient strips trailing slash from baseUrl', () => {
    const client = new SandboxClient({
        baseUrl: 'https://test.example.com/',
        projectId: 123,
        authToken: 'test-token'
    });
    
    assert(client.baseUrl === 'https://test.example.com', 'Trailing slash should be removed');
});

test('testToolSync sends correct request', async () => {
    const client = new SandboxClient({
        baseUrl: 'https://test.example.com',
        projectId: 123,
        authToken: 'test-token'
    });
    
    const result = await client.testToolSync(456, {
        tool: 'test_tool',
        input: { data: 'test' },
        user_input: 'Test input'
    });
    
    assert(result.result.success === true, 'Should return success result');
    assert(fetchCalls.length === 1, 'Should make one fetch call');
    assertContains(fetchCalls[0].url, 'await_response=true', 'URL should include await_response=true');
    assertContains(fetchCalls[0].url, '/test_tool/123/456', 'URL should include project and tool IDs');
});

test('testToolAsync without sid sends correct request', async () => {
    const client = new SandboxClient({
        baseUrl: 'https://test.example.com',
        projectId: 123,
        authToken: 'test-token'
    });
    
    const result = await client.testToolAsync(456, {
        tool: 'test_tool',
        input: { data: 'test' },
        user_input: 'Test input'
    });
    
    assert(result.task_id === 'test-task-123', 'Should return task_id');
    assert(fetchCalls.length === 1, 'Should make one fetch call');
    assertContains(fetchCalls[0].url, 'await_response=false', 'URL should include await_response=false');
    
    const body = JSON.parse(fetchCalls[0].options.body);
    assert(body.sid === null, 'sid should be null when not provided');
});

test('testToolAsync with sid sends correct request', async () => {
    const client = new SandboxClient({
        baseUrl: 'https://test.example.com',
        projectId: 123,
        authToken: 'test-token'
    });
    
    const result = await client.testToolAsync(456, {
        tool: 'test_tool',
        input: { data: 'test' },
        user_input: 'Test input'
    }, 'socket-id-123');
    
    assert(result.task_id === 'test-task-123', 'Should return task_id');
    
    const body = JSON.parse(fetchCalls[0].options.body);
    assert(body.sid === 'socket-id-123', 'sid should be included when provided');
});

test('getTaskStatus retrieves task status', async () => {
    const client = new SandboxClient({
        baseUrl: 'https://test.example.com',
        projectId: 123,
        authToken: 'test-token'
    });
    
    const status = await client.getTaskStatus('test-task-123', true, true);
    
    assert(status.status === 'SUCCESS', 'Should return status');
    assert(status.meta !== undefined, 'Should include meta when requested');
    assert(status.result !== undefined, 'Should include result when requested');
    assertContains(fetchCalls[0].url, 'meta=yes', 'URL should include meta=yes');
    assertContains(fetchCalls[0].url, 'result=yes', 'URL should include result=yes');
});

test('getTaskStatus without meta and result', async () => {
    const client = new SandboxClient({
        baseUrl: 'https://test.example.com',
        projectId: 123,
        authToken: 'test-token'
    });
    
    await client.getTaskStatus('test-task-123');
    
    assert(!fetchCalls[0].url.includes('meta=yes'), 'URL should not include meta=yes');
    assert(!fetchCalls[0].url.includes('result=yes'), 'URL should not include result=yes');
});

test('stopTaskById stops a task', async () => {
    const client = new SandboxClient({
        baseUrl: 'https://test.example.com',
        projectId: 123,
        authToken: 'test-token'
    });
    
    await client.stopTaskById('test-task-123');
    
    assert(fetchCalls.length === 1, 'Should make one fetch call');
    assert(fetchCalls[0].options.method === 'DELETE', 'Should use DELETE method');
    assertContains(fetchCalls[0].url, '/task/123/test-task-123', 'URL should include task ID');
});

test('getUserData retrieves user information', async () => {
    const client = new SandboxClient({
        baseUrl: 'https://test.example.com',
        projectId: 123,
        authToken: 'test-token'
    });
    
    const userData = await client.getUserData();
    
    assert(userData.id === 123, 'Should return user ID');
    assert(userData.email === 'test@example.com', 'Should return user email');
});

test('getMcpToolkits retrieves MCP toolkits', async () => {
    const client = new SandboxClient({
        baseUrl: 'https://test.example.com',
        projectId: 123,
        authToken: 'test-token'
    });
    
    const toolkits = await client.getMcpToolkits();
    
    assert(Array.isArray(toolkits), 'Should return an array');
    assert(toolkits.length === 2, 'Should return 2 toolkits');
    assert(toolkits[0].name === 'tool1', 'Should include tool names');
});

test('mcpToolCall executes MCP tool', async () => {
    const client = new SandboxClient({
        baseUrl: 'https://test.example.com',
        projectId: 123,
        authToken: 'test-token'
    });
    
    const result = await client.mcpToolCall({
        params: {
            name: 'test_tool',
            arguments: { arg1: 'value1' }
        }
    });
    
    assert(result.result === 'mcp call result', 'Should return MCP call result');
});

test('getListOfApps retrieves applications', async () => {
    const client = new SandboxClient({
        baseUrl: 'https://test.example.com',
        projectId: 123,
        authToken: 'test-token'
    });
    
    const apps = await client.getListOfApps();
    
    assert(Array.isArray(apps), 'Should return an array');
    assert(apps.length === 2, 'Should return 2 apps');
    assert(apps[0].name === 'App 1', 'Should include app names');
    assert(apps[0].id === 1, 'Should include app IDs');
});

test('getAppDetails retrieves application details', async () => {
    const client = new SandboxClient({
        baseUrl: 'https://test.example.com',
        projectId: 123,
        authToken: 'test-token'
    });
    
    const appDetails = await client.getAppDetails(1);
    
    assert(appDetails.id === 1, 'Should return app ID');
    assert(appDetails.name === 'Test App', 'Should return app name');
});

test('bucketExists checks bucket existence', async () => {
    const client = new SandboxClient({
        baseUrl: 'https://test.example.com',
        projectId: 123,
        authToken: 'test-token'
    });
    
    const exists = await client.bucketExists('test-bucket');
    
    assert(exists === true, 'Should return true for existing bucket');
});

test('createBucket creates a bucket', async () => {
    const client = new SandboxClient({
        baseUrl: 'https://test.example.com',
        projectId: 123,
        authToken: 'test-token'
    });
    
    const result = await client.createBucket('new-bucket', 'months', 3);
    
    assert(result.success === true, 'Should return success');
    
    const body = JSON.parse(fetchCalls[0].options.body);
    assert(body.name === 'new-bucket', 'Should include bucket name');
    assert(body.expiration_measure === 'months', 'Should include expiration measure');
    assert(body.expiration_value === 3, 'Should include expiration value');
});

test('listArtifacts lists bucket artifacts', async () => {
    const client = new SandboxClient({
        baseUrl: 'https://test.example.com',
        projectId: 123,
        authToken: 'test-token'
    });
    
    const artifacts = await client.listArtifacts('test-bucket');
    
    assert(artifacts.files !== undefined, 'Should return files');
    assertContains(fetchCalls[0].url, 'test-bucket', 'URL should include bucket name');
});

test('downloadArtifact downloads artifact', async () => {
    const client = new SandboxClient({
        baseUrl: 'https://test.example.com',
        projectId: 123,
        authToken: 'test-token'
    });
    
    const data = await client.downloadArtifact('test-bucket', 'test.txt');
    
    assert(data instanceof ArrayBuffer, 'Should return ArrayBuffer');
});

test('deleteArtifact deletes artifact', async () => {
    const client = new SandboxClient({
        baseUrl: 'https://test.example.com',
        projectId: 123,
        authToken: 'test-token'
    });
    
    await client.deleteArtifact('test-bucket', 'test.txt');
    
    assert(fetchCalls[0].options.method === 'DELETE', 'Should use DELETE method');
    assertContains(fetchCalls[0].url, 'filename=test.txt', 'URL should include filename');
});

test('unsecret retrieves secret value', async () => {
    const client = new SandboxClient({
        baseUrl: 'https://test.example.com',
        projectId: 123,
        authToken: 'test-token'
    });
    
    const secretValue = await client.unsecret('my-secret');
    
    assert(secretValue === 'secret-value-123', 'Should return secret value');
});

test('fetchAvailableConfigurations retrieves configurations', async () => {
    const client = new SandboxClient({
        baseUrl: 'https://test.example.com',
        projectId: 123,
        authToken: 'test-token'
    });
    
    const configs = await client.fetchAvailableConfigurations();
    
    assert(Array.isArray(configs), 'Should return an array');
    assert(configs.length === 2, 'Should return 2 configurations');
});

test('SandboxArtifact can be created', async () => {
    const client = new SandboxClient({
        baseUrl: 'https://test.example.com',
        projectId: 123,
        authToken: 'test-token'
    });
    
    const artifactManager = await client.artifact('test-bucket');
    
    assert(artifactManager instanceof SandboxArtifact, 'Should return SandboxArtifact instance');
    assert(artifactManager.bucketName === 'test-bucket', 'Should set bucket name');
});

test('Custom headers are included in requests', async () => {
    const client = new SandboxClient({
        baseUrl: 'https://test.example.com',
        projectId: 123,
        authToken: 'test-token',
        apiExtraHeaders: {
            'X-Custom-Header': 'custom-value'
        }
    });
    
    await client.getUserData();
    
    assert(client.headers['X-Custom-Header'] === 'custom-value', 'Custom header should be set');
});

test('XSECRET header is set', () => {
    const client = new SandboxClient({
        baseUrl: 'https://test.example.com',
        projectId: 123,
        authToken: 'test-token',
        XSECRET: 'my-secret'
    });
    
    assert(client.headers['X-SECRET'] === 'my-secret', 'X-SECRET header should be set');
});

test('Default XSECRET is "secret"', () => {
    const client = new SandboxClient({
        baseUrl: 'https://test.example.com',
        projectId: 123,
        authToken: 'test-token'
    });
    
    assert(client.headers['X-SECRET'] === 'secret', 'Default X-SECRET should be "secret"');
});

test('Model timeout is configurable', () => {
    const client = new SandboxClient({
        baseUrl: 'https://test.example.com',
        projectId: 123,
        authToken: 'test-token',
        modelTimeout: 300
    });
    
    assert(client.modelTimeout === 300, 'Model timeout should be set');
});

test('Error handling for failed requests', async () => {
    const client = new SandboxClient({
        baseUrl: 'https://test.example.com',
        projectId: 123,
        authToken: 'test-token'
    });
    
    // Mock a failed response
    global.fetch = async () => ({
        ok: false,
        status: 400,
        json: async () => ({ error: 'Bad request' })
    });
    
    try {
        await client.testToolSync(999, { tool: 'test' });
        assert(false, 'Should throw error');
    } catch (error) {
        assert(error.message.includes('Tool test failed'), 'Should throw appropriate error');
    }
});

test('ApiDetailsRequestError is thrown for API errors', async () => {
    const client = new SandboxClient({
        baseUrl: 'https://test.example.com',
        projectId: 123,
        authToken: 'test-token'
    });
    
    global.fetch = async () => ({
        ok: false,
        status: 500,
        text: async () => 'Server error'
    });
    
    try {
        await client.getUserData();
        assert(false, 'Should throw error');
    } catch (error) {
        assert(error instanceof ApiDetailsRequestError, 'Should throw ApiDetailsRequestError');
    }
});

// Run all tests
runTests();
