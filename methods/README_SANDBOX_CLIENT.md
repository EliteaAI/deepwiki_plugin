# EliteA Sandbox Client - JavaScript Implementation

A lightweight JavaScript client for interacting with the EliteA platform, providing comprehensive API access for applications, tools, MCP integration, artifacts management, and task execution.

## Features

- **Application Management**: Get app details, list applications, manage versions
- **Tool Testing**: Test tools synchronously and asynchronously with task tracking
- **MCP Integration**: Call MCP tools and list available toolkits
- **Artifact Management**: Create, read, update, delete artifacts with bucket support
- **Image Generation**: Generate images using configured AI models
- **Task Management**: Track async operations with task ID and status polling
- **Configuration**: Fetch and manage integrations and configurations
- **Secrets Management**: Unsecret values for secure credential handling

## Installation

```javascript
// ES Module
import { SandboxClient } from './methods/sandbox_client.js';

// CommonJS
const { SandboxClient } = require('./methods/sandbox_client.js');
```

## Basic Usage

### Initialize the Client

```javascript
const client = new SandboxClient({
    baseUrl: 'https://your-elitea-instance.com',
    projectId: 123,
    authToken: 'your-bearer-token',
    apiExtraHeaders: {
        // Optional additional headers
    },
    configurations: [], // Optional pre-loaded configurations
    XSECRET: 'secret', // Optional API secret
    modelTimeout: 120, // Request timeout in seconds
    modelImageGeneration: 'dall-e-3' // Optional image model
});
```

## Core Features

### 1. Application Management

```javascript
// Get application details
const appDetails = await client.getAppDetails(applicationId);

// List all applications
const apps = await client.getListOfApps();
// Returns: [{ name: 'App 1', id: 1 }, ...]

// Get version details
const versionDetails = await client.getAppVersionDetails(
    applicationId,
    versionId
);
```

### 2. Tool Testing (NEW)

#### Synchronous Tool Execution

```javascript
const result = await client.testToolSync(toolId, {
    tool: 'my_tool',
    testing_name: 'test_run_1',
    input: {
        param1: 'value1',
        param2: 'value2'
    },
    output: {
        // Expected output configuration
    },
    input_mapping: {
        // Input mappings
    },
    user_input: 'Test input data',
    call_type: 'tool' // or 'function'
});

console.log('Tool result:', result);
```

#### Asynchronous Tool Execution with Task Tracking

```javascript
// Start async tool test (sid is optional)
// Option A: Without sid - use polling to get results
const asyncResult = await client.testToolAsync(toolId, {
    tool: 'my_tool',
    testing_name: 'async_test',
    input: { /* ... */ },
    user_input: 'Test data'
});

// Option B: With sid - results stream via Socket.IO (if connected)
const asyncResultWithStreaming = await client.testToolAsync(toolId, {
    tool: 'my_tool',
    testing_name: 'async_test',
    input: { /* ... */ },
    user_input: 'Test data'
}, 'your-socket-id');

const taskId = asyncResult.task_id;
console.log('Task started:', taskId);

// Option 1: Poll for completion
const finalResult = await client.waitForTask(taskId, 300); // 300s timeout
console.log('Task completed:', finalResult);

// Option 2: Poll with progress callback
const result = await client.pollTaskStatus(
    taskId,
    (status) => {
        console.log('Progress:', status.status, 'Attempt:', status.attempts);
    },
    60, // max attempts
    1000 // interval in ms
);

// Option 3: Check status manually
const status = await client.getTaskStatus(taskId, true, false);
console.log('Current status:', status.status);

// Get status with result when complete
const statusWithResult = await client.getTaskStatus(taskId, true, true);
if (statusWithResult.status === 'SUCCESS') {
    console.log('Result:', statusWithResult.result);
}
```

### 3. Task Management

```javascript
// Get task status
const taskStatus = await client.getTaskStatus(
    taskId,
    true, // withMeta
    true  // withResult
);

// Stop a running task
await client.stopTaskById(taskId);

// Or stop by message group UUID (for chat tasks)
await client.stopTask(messageGroupUuid);

// Wait for task with timeout
try {
    const result = await client.waitForTask(taskId, 300);
    console.log('Task completed successfully:', result);
} catch (error) {
    console.error('Task failed or timed out:', error);
}
```

### 4. MCP (Model Context Protocol) Integration

```javascript
// Get available MCP toolkits
const toolkits = await client.getMcpToolkits();

// Call an MCP tool
const mcpResult = await client.mcpToolCall({
    params: {
        name: 'tool_name',
        arguments: {
            arg1: 'value1',
            arg2: 'value2'
        }
    }
});
```

### 5. Artifact Management

```javascript
// Create artifact manager for a bucket
const artifactManager = await client.artifact('my-bucket');

// Create artifact
await artifactManager.create('file.txt', 'Hello, World!');

// Get artifact
const content = await artifactManager.get('file.txt');

// List artifacts
const artifacts = await artifactManager.list();

// Append to artifact
await artifactManager.append('file.txt', '\nNew line');

// Overwrite artifact
await artifactManager.overwrite('file.txt', 'Completely new content');

// Delete artifact
await artifactManager.delete('file.txt');

// Get artifact as bytes
const bytes = await artifactManager.getContentBytes('file.txt');
```

### 6. Bucket Management

```javascript
// Check if bucket exists
const exists = await client.bucketExists('my-bucket');

// Create bucket with expiration
await client.createBucket(
    'my-bucket',
    'months', // expiration_measure: 'days', 'months', 'years'
    3         // expiration_value
);

// List artifacts in bucket
const artifacts = await client.listArtifacts('my-bucket');

// Create artifact directly
await client.createArtifact('my-bucket', 'file.txt', 'content');

// Download artifact
const data = await client.downloadArtifact('my-bucket', 'file.txt');

// Delete artifact
await client.deleteArtifact('my-bucket', 'file.txt');
```

### 7. Configuration & Integration Management

```javascript
// Fetch available configurations
const configs = await client.fetchAvailableConfigurations();

// Get all models and integrations
const integrations = await client.allModelsAndIntegrations();

// Get specific integration details
const integration = await client.getIntegrationDetails(
    integrationId,
    formatForModel = false
);
```

### 8. Secrets Management

```javascript
// Unsecret a value
const secretValue = await client.unsecret('my-secret-name');
console.log('Secret value:', secretValue);
```

### 9. Image Generation

```javascript
const imageResult = await client.generateImage({
    prompt: 'A beautiful sunset over mountains',
    n: 1,
    size: '1024x1024', // or 'auto'
    quality: 'hd', // or 'standard', 'auto'
    responseFormat: 'b64_json', // or 'url'
    style: 'vivid' // or 'natural'
});

// Access generated images
const images = imageResult.data;
```

### 10. User Management

```javascript
// Get current user data
const userData = await client.getUserData();
console.log('User ID:', userData.id);
console.log('Email:', userData.email);
```

## Complete Example: Tool Testing Workflow

```javascript
import { SandboxClient } from './methods/sandbox_client.js';

async function testToolWorkflow() {
    const client = new SandboxClient({
        baseUrl: 'https://elitea.example.com',
        projectId: 123,
        authToken: 'your-token'
    });

    try {
        // 1. Synchronous test
        console.log('Running synchronous tool test...');
        const syncResult = await client.testToolSync(456, {
            tool: 'data_processor',
            testing_name: 'sync_test',
            input: { data: [1, 2, 3, 4, 5] },
            user_input: 'Process these numbers'
        });
        console.log('Sync result:', syncResult);

        // 2. Asynchronous test with polling
        console.log('\nRunning asynchronous tool test...');
        const asyncResult = await client.testToolAsync(456, {
            tool: 'data_processor',
            testing_name: 'async_test',
            input: { data: Array.from({ length: 1000 }, (_, i) => i) },
            user_input: 'Process large dataset'
        });

        const taskId = asyncResult.task_id;
        console.log('Task ID:', taskId);

        // Poll with progress updates
        const result = await client.pollTaskStatus(
            taskId,
            (progress) => {
                console.log(`Status: ${progress.status}, Attempt: ${progress.attempts}`);
            },
            120, // 120 attempts
            1000 // 1 second interval
        );

        console.log('Final result:', result.result);

    } catch (error) {
        console.error('Error:', error.message);
    }
}

testToolWorkflow();
```

## Task Status Values

Task statuses follow Celery conventions:
- `PENDING`: Task is waiting to be executed
- `STARTED`: Task execution has begun
- `RETRY`: Task is being retried after failure
- `SUCCESS`: Task completed successfully
- `FAILURE`: Task failed with an error

## Error Handling

```javascript
try {
    const result = await client.testToolSync(toolId, params);
} catch (error) {
    if (error instanceof ApiDetailsRequestError) {
        console.error('API Error:', error.message);
    } else {
        console.error('Unexpected error:', error);
    }
}
```

## Advanced Usage

### Custom Headers

```javascript
const client = new SandboxClient({
    baseUrl: 'https://elitea.example.com',
    projectId: 123,
    authToken: 'token',
    apiExtraHeaders: {
        'X-Custom-Header': 'value',
        'X-Request-ID': uuid()
    }
});
```

### Timeout Configuration

```javascript
const client = new SandboxClient({
    baseUrl: 'https://elitea.example.com',
    projectId: 123,
    authToken: 'token',
    modelTimeout: 300 // 5 minutes for long-running operations
});
```

## Browser vs Node.js

This client works in both browser and Node.js environments:

**Browser:**
```html
<script type="module">
    import { SandboxClient } from './sandbox_client.js';
    const client = new SandboxClient({ /* ... */ });
</script>
```

**Node.js:**
```javascript
// Install node-fetch if using Node < 18
// npm install node-fetch

const { SandboxClient } = require('./sandbox_client.js');
const client = new SandboxClient({ /* ... */ });
```

## API Endpoints Reference

| Method | Endpoint | Purpose |
|--------|----------|---------|
| `testToolSync` | `POST /api/v1/applications/test_tool/{project_id}/{tool_id}` | Execute tool test synchronously |
| `testToolAsync` | `POST /api/v1/applications/test_tool/{project_id}/{tool_id}` | Execute tool test asynchronously |
| `getTaskStatus` | `GET /api/v1/applications/task/{project_id}/{task_id}` | Get task execution status |
| `stopTaskById` | `DELETE /api/v1/applications/task/{project_id}/{task_id}` | Stop running task |
| `getMcpToolkits` | `GET /api/v1/mcp_sse/tools_list/{project_id}/{user_id}` | List MCP toolkits |
| `mcpToolCall` | `POST /api/v1/mcp_sse/tools_call/{project_id}/{user_id}` | Execute MCP tool |

## TypeScript Support

For TypeScript users, here are the main types:

```typescript
interface SandboxClientConfig {
    baseUrl: string;
    projectId: number;
    authToken: string;
    apiExtraHeaders?: Record<string, string>;
    configurations?: any[];
    XSECRET?: string;
    modelTimeout?: number;
    modelImageGeneration?: string;
}

interface TestToolParams {
    tool: string;
    testing_name?: string;
    input?: Record<string, any>;
    output?: Record<string, any>;
    structured_output?: any;
    input_mapping?: Record<string, any>;
    transition?: Record<string, any>;
    input_variables?: any[];
    user_input?: string;
    call_type?: 'tool' | 'function';
}

interface TaskStatus {
    status: 'PENDING' | 'STARTED' | 'RETRY' | 'SUCCESS' | 'FAILURE';
    meta?: Record<string, any>;
    result?: any;
}
```

## License

See the LICENSE file in the elitea_core module.

## Support

For issues and questions, please refer to the main EliteA SDK documentation.
