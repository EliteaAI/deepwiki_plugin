# Integrating Sandbox Client in DeepWiki Plugin

This guide shows how to use the new EliteA Sandbox Client within the DeepWiki Plugin.

## Quick Start

### 1. Import the Client

```javascript
import { SandboxClient } from './methods/sandbox_client.js';
```

### 2. Initialize with Plugin Context

If you're using the client within a DeepWiki route or method that has access to the plugin context:

```javascript
// In a route handler or method
async function myPluginMethod(context) {
    // Get configuration from environment or context
    const client = new SandboxClient({
        baseUrl: process.env.ELITEA_BASE_URL || context.config.baseUrl,
        projectId: context.projectId,
        authToken: context.authToken,
        XSECRET: process.env.ELITEA_SECRET
    });
    
    // Use the client
    const apps = await client.getListOfApps();
    return apps;
}
```

### 3. Tool Testing in Plugin Routes

```javascript
// routes/tool_test.py equivalent in JavaScript
import { SandboxClient } from '../methods/sandbox_client.js';

export async function testToolRoute(req, res) {
    const { toolId, ...params } = req.body;
    const { projectId, authToken } = req.context;
    
    const client = new SandboxClient({
        baseUrl: req.context.baseUrl,
        projectId,
        authToken
    });
    
    try {
        if (req.query.async === 'true') {
            // Async execution
            // - Pass socketId (3rd param) if you want Socket.IO streaming
            // - Omit socketId if you'll poll for results instead
            const result = await client.testToolAsync(toolId, params, req.socketId);
            res.json({ task_id: result.task_id });
        } else {
            // Sync execution
            const result = await client.testToolSync(toolId, params);
            res.json({ result });
        }
    } catch (error) {
        res.status(400).json({ error: error.message });
    }
}
```

## Integration Patterns

### Pattern 1: DeepWiki Document Processing with Tool Testing

```javascript
import { SandboxClient } from '../methods/sandbox_client.js';

export class DocumentProcessor {
    constructor(context) {
        this.client = new SandboxClient({
            baseUrl: context.config.eliteaBaseUrl,
            projectId: context.projectId,
            authToken: context.authToken
        });
    }
    
    async processDocumentWithTool(documentId, toolId) {
        // Get document content
        const doc = await this.getDocument(documentId);
        
        // Test tool with document content
        const result = await this.client.testToolSync(toolId, {
            tool: 'document_analyzer',
            input: {
                content: doc.content,
                metadata: doc.metadata
            },
            user_input: 'Analyze this document'
        });
        
        // Store results as artifact
        const artifacts = await this.client.artifact('doc-analysis');
        await artifacts.create(
            `${documentId}_analysis.json`,
            JSON.stringify(result, null, 2)
        );
        
        return result;
    }
}
```

### Pattern 2: Background Processing with Task Tracking

```javascript
import { SandboxClient } from '../methods/sandbox_client.js';

export class BackgroundProcessor {
    constructor(context) {
        this.client = new SandboxClient({
            baseUrl: context.config.eliteaBaseUrl,
            projectId: context.projectId,
            authToken: context.authToken
        });
        this.activeTasks = new Map();
    }
    
    async startProcessing(jobId, toolId, params, socketId = null) {
        // Start async processing (socketId is optional)
        const result = await this.client.testToolAsync(toolId, params, socketId);
        
        // Track the task
        this.activeTasks.set(jobId, {
            taskId: result.task_id,
            startTime: Date.now(),
            params
        });
        
        // Monitor in background
        this._monitorTask(jobId, result.task_id);
        
        return result.task_id;
    }
    
    async _monitorTask(jobId, taskId) {
        try {
            const result = await this.client.pollTaskStatus(
                taskId,
                (status) => {
                    console.log(`Job ${jobId}: ${status.status}`);
                    // Could emit events here
                }
            );
            
            console.log(`Job ${jobId} completed:`, result);
            this.activeTasks.delete(jobId);
            
        } catch (error) {
            console.error(`Job ${jobId} failed:`, error);
            this.activeTasks.delete(jobId);
        }
    }
    
    async cancelJob(jobId) {
        const task = this.activeTasks.get(jobId);
        if (task) {
            await this.client.stopTaskById(task.taskId);
            this.activeTasks.delete(jobId);
            return true;
        }
        return false;
    }
    
    getActiveJobs() {
        return Array.from(this.activeTasks.entries()).map(([jobId, task]) => ({
            jobId,
            taskId: task.taskId,
            duration: Date.now() - task.startTime
        }));
    }
}
```

### Pattern 3: Wiki Generation with MCP Tools

```javascript
import { SandboxClient } from '../methods/sandbox_client.js';

export class WikiGenerator {
    constructor(context) {
        this.client = new SandboxClient({
            baseUrl: context.config.eliteaBaseUrl,
            projectId: context.projectId,
            authToken: context.authToken
        });
    }
    
    async generateWikiSection(codeSection, toolId) {
        // Use MCP tools for code analysis
        const toolkits = await this.client.getMcpToolkits();
        const codeAnalyzer = toolkits.find(t => t.name === 'code-analyzer');
        
        if (codeAnalyzer) {
            // Call MCP tool directly
            const analysis = await this.client.mcpToolCall({
                params: {
                    name: 'analyze-code',
                    arguments: {
                        code: codeSection.content,
                        language: codeSection.language
                    }
                }
            });
            
            // Then use regular tool to generate docs
            const docs = await this.client.testToolSync(toolId, {
                tool: 'doc_generator',
                input: {
                    analysis: analysis,
                    format: 'markdown'
                },
                user_input: 'Generate wiki section'
            });
            
            return docs;
        }
        
        return null;
    }
}
```

## Using with Plugin Events

```javascript
// plugin_implementation/wiki_agent.js
import { SandboxClient } from '../methods/sandbox_client.js';

export class WikiAgent {
    constructor(context) {
        this.context = context;
        this.client = new SandboxClient({
            baseUrl: context.config.eliteaBaseUrl,
            projectId: context.projectId,
            authToken: context.authToken
        });
    }
    
    async onRepositoryIndexed(event) {
        const { repositoryPath, modules } = event.data;
        
        // Process each module with appropriate tools
        const tasks = [];
        
        for (const module of modules) {
            // Start async tool test for each module
            const result = await this.client.testToolAsync(
                this.getToolIdForModule(module),
                {
                    tool: 'module_analyzer',
                    input: {
                        modulePath: module.path,
                        language: module.language
                    },
                    user_input: `Analyze ${module.name}`
                }
            );
            
            tasks.push({
                module: module.name,
                taskId: result.task_id
            });
        }
        
        // Wait for all to complete
        const results = await Promise.all(
            tasks.map(t => this.client.waitForTask(t.taskId, 600))
        );
        
        // Store aggregated results
        const artifacts = await this.client.artifact('wiki-analysis');
        await artifacts.create(
            'repository_analysis.json',
            JSON.stringify({ repository: repositoryPath, results }, null, 2)
        );
        
        return results;
    }
    
    getToolIdForModule(module) {
        // Logic to select appropriate tool based on module
        const toolMap = {
            'python': 123,
            'javascript': 456,
            'java': 789
        };
        return toolMap[module.language] || toolMap['python'];
    }
}
```

## Environment Configuration

Create a `.env` file for development:

```bash
# .env
ELITEA_BASE_URL=https://your-elitea-instance.com
ELITEA_PROJECT_ID=123
ELITEA_AUTH_TOKEN=your-bearer-token
ELITEA_SECRET=your-secret-key
```

Load it in your plugin initialization:

```javascript
import dotenv from 'dotenv';
dotenv.config();

// Now environment variables are available
const client = new SandboxClient({
    baseUrl: process.env.ELITEA_BASE_URL,
    projectId: parseInt(process.env.ELITEA_PROJECT_ID),
    authToken: process.env.ELITEA_AUTH_TOKEN,
    XSECRET: process.env.ELITEA_SECRET
});
```

## Error Handling in Plugin Context

```javascript
import { SandboxClient, ApiDetailsRequestError } from '../methods/sandbox_client.js';

export async function handleToolRequest(req, res, context) {
    const client = new SandboxClient({
        baseUrl: context.config.eliteaBaseUrl,
        projectId: context.projectId,
        authToken: context.authToken
    });
    
    try {
        const result = await client.testToolSync(
            req.params.toolId,
            req.body
        );
        
        res.json({ success: true, result });
        
    } catch (error) {
        if (error instanceof ApiDetailsRequestError) {
            // API-specific error
            res.status(400).json({
                error: 'API Error',
                message: error.message,
                type: 'api_error'
            });
        } else if (error.message.includes('timeout')) {
            // Timeout error
            res.status(504).json({
                error: 'Timeout',
                message: 'Tool execution timed out',
                type: 'timeout'
            });
        } else {
            // General error
            res.status(500).json({
                error: 'Internal Error',
                message: error.message,
                type: 'internal'
            });
        }
        
        // Log for debugging
        console.error('Tool test error:', error);
    }
}
```

## Testing Integration

```javascript
// tests/test_sandbox_client_integration.js
import { SandboxClient } from '../methods/sandbox_client.js';
import { expect } from 'chai';

describe('Sandbox Client Integration', () => {
    let client;
    
    before(() => {
        client = new SandboxClient({
            baseUrl: process.env.TEST_ELITEA_BASE_URL,
            projectId: process.env.TEST_PROJECT_ID,
            authToken: process.env.TEST_AUTH_TOKEN
        });
    });
    
    it('should list available applications', async () => {
        const apps = await client.getListOfApps();
        expect(apps).to.be.an('array');
    });
    
    it('should test tool synchronously', async () => {
        const result = await client.testToolSync(123, {
            tool: 'test_tool',
            input: { test: true },
            user_input: 'Test'
        });
        
        expect(result).to.have.property('result');
    });
    
    it('should test tool asynchronously with task tracking', async () => {
        const asyncResult = await client.testToolAsync(123, {
            tool: 'test_tool',
            input: { test: true },
            user_input: 'Test'
        });
        
        expect(asyncResult).to.have.property('task_id');
        
        const status = await client.getTaskStatus(asyncResult.task_id);
        expect(status).to.have.property('status');
    });
});
```

## Best Practices

1. **Reuse Client Instances**: Create one client per context/session
2. **Handle Timeouts**: Set appropriate timeouts for long operations
3. **Monitor Tasks**: Use polling for async operations or Socket.IO for real-time
4. **Store Results**: Use artifacts for persistent storage
5. **Error Handling**: Catch and handle specific error types
6. **Logging**: Log task IDs and operations for debugging
7. **Cleanup**: Stop orphaned tasks on plugin shutdown

## Next Steps

1. Add Socket.IO integration for real-time task updates
2. Create middleware for automatic client injection
3. Add request caching for frequently accessed data
4. Implement connection pooling for concurrent operations
5. Add metrics collection for monitoring
