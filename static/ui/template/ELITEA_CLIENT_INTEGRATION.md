# DeepWiki Integration with EliteA Sandbox Client

This guide shows how the DeepWiki plugin now uses the EliteA Sandbox Client for API interactions.

## Files Added

### 1. `/static/ui/template/src/utils/sandbox_client.js`
The core EliteA Sandbox Client (copied from `/methods/sandbox_client.js`)

### 2. `/static/ui/template/src/utils/eliteaClient.js`
Adapter that provides configured client instance and helper functions:
- `getEliteAClient()` - Get singleton client instance
- `testTool()` - Test tools synchronously
- `testToolAsync()` - Test tools asynchronously
- `getArtifacts()` - Get artifact manager
- `listMcpToolkits()` - List MCP toolkits
- `callMcpTool()` - Call MCP tools
- `listApplications()` - List applications
- `getApplication()` - Get application details

### 3. `/static/ui/template/src/utils/useEliteAHooks.js`
React hooks that integrate the client into React components:
- `useToolkit()` - Fetch and manage toolkit data
- `useArtifacts()` - Manage artifacts with CRUD operations
- `useToolInvocation()` - Invoke tools with task tracking
- `useMcpToolkits()` - Manage MCP toolkits
- `useApplications()` - Manage applications list

## Quick Start

### Using the Hooks (Recommended)

```jsx
import { useArtifacts } from './utils/useEliteAHooks';

function MyComponent({ bucketName }) {
  const {
    artifacts,
    loading,
    error,
    createArtifact,
    deleteArtifact,
    getArtifact,
    updateArtifact,
    refetch
  } = useArtifacts(bucketName);

  const handleCreate = async () => {
    const result = await createArtifact('test.md', '# Hello World');
    if (result.success) {
      console.log('Created successfully!');
    }
  };

  if (loading) return <div>Loading...</div>;
  if (error) return <div>Error: {error}</div>;

  return (
    <div>
      <button onClick={handleCreate}>Create Artifact</button>
      <ul>
        {artifacts.map(artifact => (
          <li key={artifact.name}>{artifact.name}</li>
        ))}
      </ul>
    </div>
  );
}
```

### Using the Client Directly

```jsx
import { getEliteAClient } from './utils/eliteaClient';

async function myFunction() {
  const client = getEliteAClient();
  
  // List artifacts
  const artifactManager = await client.artifact('my-bucket');
  const artifacts = await artifactManager.list();
  
  // Create artifact
  await artifactManager.create('file.txt', 'content');
  
  // Get artifact
  const content = await artifactManager.get('file.txt');
  
  // Delete artifact
  await artifactManager.delete('file.txt');
}
```

## Migration Examples

### Before: Direct Fetch

```jsx
// Old way - direct fetch
const loadArtifactsList = async (bucketName) => {
  const url = `/api/v1/artifacts/artifacts/default/${projectId}/${bucketName}`;
  const response = await fetch(url, { headers });
  const data = await response.json();
  setArtifacts(data.files || []);
};
```

### After: Using Hook

```jsx
// New way - using hook
const { artifacts, loading, error, refetch } = useArtifacts(bucketName);
// artifacts are automatically loaded and managed
```

### Before: Manual Artifact Management

```jsx
// Old way - manual artifact operations
const saveArtifact = async (fileName, content) => {
  const formData = new FormData();
  formData.append('file', new Blob([content]), fileName);
  
  const url = `/api/v1/artifacts/artifacts/default/${projectId}/${bucketName}`;
  const response = await fetch(url, {
    method: 'POST',
    body: formData,
    headers: {/* auth headers */}
  });
  
  if (!response.ok) {
    throw new Error('Failed to save');
  }
  
  // Reload list
  await loadArtifactsList(bucketName);
};
```

### After: Using Hook

```jsx
// New way - using hook
const { createArtifact } = useArtifacts(bucketName);

const saveArtifact = async (fileName, content) => {
  const result = await createArtifact(fileName, content);
  if (result.success) {
    console.log('Saved!');
    // List is automatically refreshed
  }
};
```

## Environment Configuration

The client reads configuration from multiple sources:

1. **window.deepwiki_ui_config** (production runtime)
2. **import.meta.env** (Vite development)
3. **process.env** (Node.js)

### Required Environment Variables

```env
# Base URL for API
VITE_API_BASE_URL=https://your-instance.com

# Project ID
VITE_PROJECT_ID=123

# Authentication token
VITE_AUTH_TOKEN=your-bearer-token

# API secret (optional)
VITE_API_SECRET=secret
```

### Runtime Configuration

In production, set config in your HTML template:

```html
<script>
  window.deepwiki_ui_config = {
    base_url: '{{ base_url }}',
    project_id: {{ project_id }},
    auth_token: '{{ auth_token }}'
  };
</script>
```

## Available Hooks

### useArtifacts(bucketName)

Manages artifacts in a bucket.

```jsx
const {
  artifacts,      // Array of artifacts
  loading,        // Loading state
  error,          // Error message
  createArtifact, // (fileName, content) => Promise
  deleteArtifact, // (fileName) => Promise
  getArtifact,    // (fileName) => Promise
  updateArtifact, // (fileName, content) => Promise
  refetch         // () => Promise - Reload list
} = useArtifacts('my-bucket');
```

### useToolInvocation(toolkitName, toolName)

Invokes tools with optional async task tracking.

```jsx
const {
  invoking,  // Boolean - invocation in progress
  result,    // Invocation result
  error,     // Error message
  taskId,    // Task ID for async invocations
  progress,  // Progress information
  invoke,    // (parameters, useAsync) => Promise
  cancel     // (invocationId) => Promise
} = useToolInvocation('deepwiki_plugin', 'generate_wiki');
```

### useMcpToolkits()

Lists and manages MCP toolkits.

```jsx
const {
  toolkits,  // Array of MCP toolkits
  loading,   // Loading state
  error,     // Error message
  callTool,  // (params) => Promise
  refetch    // () => Promise
} = useMcpToolkits();
```

### useApplications()

Lists and manages applications.

```jsx
const {
  applications,  // Array of applications
  loading,       // Loading state
  error,         // Error message
  getAppDetails, // (appId) => Promise
  refetch        // () => Promise
} = useApplications();
```

## Advanced Usage

### Custom Client Configuration

```jsx
import { createEliteAClient } from './utils/eliteaClient';

const customClient = createEliteAClient({
  baseUrl: 'https://custom-instance.com',
  projectId: 456,
  authToken: 'custom-token',
  headers: {
    'X-Custom-Header': 'value'
  },
  modelTimeout: 600
});

// Use custom client
const apps = await customClient.getListOfApps();
```

### Task Tracking with Progress

```jsx
import { getEliteAClient } from './utils/eliteaClient';

async function longRunningTask() {
  const client = getEliteAClient();
  
  // Start async task
  const { task_id } = await client.testToolAsync(toolId, {
    tool: 'data_processor',
    input: { large_dataset: [...] },
    user_input: 'Process data'
  });
  
  // Poll with progress callback
  const result = await client.pollTaskStatus(
    task_id,
    (status) => {
      console.log(`Status: ${status.status}, Attempt: ${status.attempts}`);
      // Update UI with progress
      setProgress(status);
    },
    120, // max attempts
    1000 // interval ms
  );
  
  console.log('Completed:', result);
}
```

### Error Handling

```jsx
import { ApiDetailsRequestError } from './utils/sandbox_client';

try {
  const result = await createArtifact('file.txt', 'content');
  if (!result.success) {
    throw new Error(result.error);
  }
} catch (error) {
  if (error instanceof ApiDetailsRequestError) {
    // Handle API-specific errors
    console.error('API Error:', error.message);
  } else {
    // Handle other errors
    console.error('Error:', error);
  }
}
```

## Integration Checklist

- [x] Copy `sandbox_client.js` to UI utils folder
- [x] Create `eliteaClient.js` adapter
- [x] Create `useEliteAHooks.js` React hooks
- [ ] Update `DeepWikiApp.jsx` to use hooks
- [ ] Replace direct fetch calls with client methods
- [ ] Update environment configuration
- [ ] Test artifact operations
- [ ] Test tool invocations
- [ ] Add error boundaries for error handling
- [ ] Update documentation

## Benefits

1. **Unified API Client**: Single client for all EliteA API interactions
2. **Type Safety**: Better code completion and type checking
3. **Error Handling**: Consistent error handling across the app
4. **Task Tracking**: Built-in support for async operations
5. **State Management**: React hooks handle loading and error states
6. **Reusability**: Client and hooks can be reused across components
7. **Testing**: Easier to mock and test
8. **Maintainability**: Centralized API logic

## Next Steps

1. **Migrate DeepWikiApp.jsx**: Replace fetch calls with hooks
2. **Add Error Boundaries**: Wrap components with error boundaries
3. **Add Loading States**: Use hook loading states consistently
4. **Add Tests**: Write tests for hooks and client
5. **Performance**: Add caching and request deduplication
6. **WebSocket**: Integrate Socket.IO for real-time updates

## Example: Full Component Migration

```jsx
// Before
function WikiGenerator() {
  const [artifacts, setArtifacts] = useState([]);
  const [loading, setLoading] = useState(false);
  
  useEffect(() => {
    const load = async () => {
      setLoading(true);
      const response = await fetch(`/api/v1/artifacts...`);
      const data = await response.json();
      setArtifacts(data.files);
      setLoading(false);
    };
    load();
  }, []);
  
  const saveWiki = async (content) => {
    const formData = new FormData();
    formData.append('file', new Blob([content]), 'wiki.md');
    await fetch('/api/v1/artifacts...', { method: 'POST', body: formData });
    // reload...
  };
  
  return <div>{/* render */}</div>;
}

// After
import { useArtifacts } from './utils/useEliteAHooks';

function WikiGenerator() {
  const { artifacts, loading, createArtifact } = useArtifacts('wiki-bucket');
  
  const saveWiki = async (content) => {
    await createArtifact('wiki.md', content);
    // List is auto-refreshed!
  };
  
  return <div>{/* render */}</div>;
}
```

## Troubleshooting

### Client not initialized
```
Error: Cannot read property 'testToolSync' of undefined
```
**Solution**: Make sure environment variables are set correctly.

### CORS errors
```
Access to fetch at '...' has been blocked by CORS policy
```
**Solution**: Check baseUrl configuration and ensure API allows CORS.

### Authentication errors
```
401 Unauthorized
```
**Solution**: Verify authToken is set and valid.

## Support

For issues or questions:
1. Check environment configuration
2. Verify API endpoints are accessible
3. Check browser console for detailed errors
4. Review network tab for request/response details
