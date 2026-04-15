/**
 * Example: Using EliteA Client with Iframe Session Sharing
 * 
 * This file demonstrates how to use the EliteA Sandbox Client in React components
 * when running inside an iframe with automatic session sharing.
 */

import React, { useState, useEffect } from 'react';
import { 
  getEliteAClient, 
  updateClientAuthToken,
  updateClientConfig 
} from './utils/eliteaClient';
import { 
  useArtifacts, 
  useToolInvocation, 
  useMcpToolkits 
} from './utils/useEliteAHooks';

// ============================================================================
// EXAMPLE 1: Basic Usage with Automatic Session Sharing
// ============================================================================

/**
 * Simple component that uses artifacts with automatic session sharing
 */
function ArtifactViewer({ bucketName = 'wiki-docs' }) {
  // Hook automatically uses shared session from parent window
  const { artifacts, loading, error, createArtifact, deleteArtifact } = useArtifacts(bucketName);

  const handleCreateFile = async () => {
    // This call will automatically include session cookies
    const result = await createArtifact('example.md', '# Hello World\n\nThis is a test.');
    if (result.success) {
      console.log('File created successfully!');
    } else {
      console.error('Failed to create file:', result.error);
    }
  };

  if (loading) return <div>Loading artifacts...</div>;
  if (error) return <div>Error: {error}</div>;

  return (
    <div>
      <h2>Artifacts in {bucketName}</h2>
      <button onClick={handleCreateFile}>Create Example File</button>
      <ul>
        {artifacts.map(artifact => (
          <li key={artifact.name}>
            {artifact.name}
            <button onClick={() => deleteArtifact(artifact.name)}>Delete</button>
          </li>
        ))}
      </ul>
    </div>
  );
}

// ============================================================================
// EXAMPLE 2: Using Direct Client with Session Sharing
// ============================================================================

/**
 * Component that uses the client directly instead of hooks
 */
function DirectClientUsage() {
  const [toolkits, setToolkits] = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    // Get singleton client - automatically configured for iframe session sharing
    const client = getEliteAClient();
    
    // All client methods will include session cookies automatically
    async function loadData() {
      try {
        setLoading(true);
        
        // This call includes credentials: 'include' automatically
        const mcpToolkits = await client.getMcpToolkits();
        setToolkits(mcpToolkits);
        
        console.log('[DirectClient] Session cookies shared automatically');
      } catch (error) {
        console.error('Error loading toolkits:', error);
      } finally {
        setLoading(false);
      }
    }

    loadData();
  }, []);

  if (loading) return <div>Loading MCP toolkits...</div>;

  return (
    <div>
      <h2>MCP Toolkits (via Direct Client)</h2>
      <ul>
        {toolkits.map(toolkit => (
          <li key={toolkit.id}>{toolkit.name}</li>
        ))}
      </ul>
    </div>
  );
}

// ============================================================================
// EXAMPLE 3: Iframe Detection and Logging
// ============================================================================

/**
 * Component that detects iframe mode and logs session info
 */
function IframeDetector() {
  const [iframeInfo, setIframeInfo] = useState({});

  useEffect(() => {
    const isIframe = window.self !== window.top;
    const parentOrigin = document.referrer;
    const currentOrigin = window.location.origin;
    
    setIframeInfo({
      isIframe,
      parentOrigin,
      currentOrigin,
      sameOrigin: parentOrigin.startsWith(currentOrigin)
    });

    if (isIframe) {
      console.log('[IframeDetector] Running in iframe mode');
      console.log('[IframeDetector] Parent origin:', parentOrigin);
      console.log('[IframeDetector] Session cookies will be shared automatically');
    } else {
      console.log('[IframeDetector] Running in standalone mode');
    }
  }, []);

  return (
    <div>
      <h2>Iframe Session Info</h2>
      <dl>
        <dt>Is Iframe:</dt>
        <dd>{iframeInfo.isIframe ? 'Yes ✓' : 'No'}</dd>
        
        <dt>Current Origin:</dt>
        <dd>{iframeInfo.currentOrigin}</dd>
        
        <dt>Parent Origin:</dt>
        <dd>{iframeInfo.parentOrigin || 'N/A'}</dd>
        
        <dt>Same Origin:</dt>
        <dd>{iframeInfo.sameOrigin ? 'Yes ✓' : 'No'}</dd>
        
        <dt>Session Sharing:</dt>
        <dd>{iframeInfo.isIframe && iframeInfo.sameOrigin ? 'Enabled ✓' : 'N/A'}</dd>
      </dl>
    </div>
  );
}

// ============================================================================
// EXAMPLE 4: Dynamic Token Update (Advanced)
// ============================================================================

/**
 * Component that listens for token updates from parent window
 * (Only needed for special scenarios - normal session sharing doesn't require this)
 */
function TokenUpdateListener() {
  const [tokenStatus, setTokenStatus] = useState('Using session cookies');

  useEffect(() => {
    // Listen for explicit token updates from parent (optional, not usually needed)
    const handleMessage = (event) => {
      // Verify origin for security
      if (event.origin !== window.location.origin) {
        console.warn('[TokenUpdate] Ignoring message from different origin:', event.origin);
        return;
      }

      if (event.data.type === 'AUTH_TOKEN_UPDATE') {
        console.log('[TokenUpdate] Received token update from parent');
        updateClientAuthToken(event.data.token);
        setTokenStatus('Token updated from parent');
      }
    };

    window.addEventListener('message', handleMessage);
    return () => window.removeEventListener('message', handleMessage);
  }, []);

  return (
    <div>
      <h2>Token Update Status</h2>
      <p>{tokenStatus}</p>
      <small>
        Note: In most cases, session cookies are shared automatically and
        explicit token updates are not needed.
      </small>
    </div>
  );
}

// ============================================================================
// EXAMPLE 5: Tool Invocation with Session Sharing
// ============================================================================

/**
 * Component that invokes a tool with automatic session authentication
 */
function ToolInvoker({ toolkitName = 'deepwiki_plugin', toolName = 'generate_wiki' }) {
  const { 
    invoking, 
    result, 
    error, 
    invoke 
  } = useToolInvocation(toolkitName, toolName);

  const handleInvoke = async () => {
    // Session cookies included automatically
    const result = await invoke({
      repository_path: '/path/to/repo',
      output_format: 'markdown'
    });

    if (result.success) {
      console.log('Tool invoked successfully!');
      console.log('Result:', result.data);
    } else {
      console.error('Tool invocation failed:', result.error);
    }
  };

  return (
    <div>
      <h2>Tool Invoker: {toolName}</h2>
      <button onClick={handleInvoke} disabled={invoking}>
        {invoking ? 'Running...' : 'Run Tool'}
      </button>
      
      {error && <div style={{ color: 'red' }}>Error: {error}</div>}
      
      {result && (
        <div>
          <h3>Result:</h3>
          <pre>{JSON.stringify(result, null, 2)}</pre>
        </div>
      )}
    </div>
  );
}

// ============================================================================
// EXAMPLE 6: Configuration Update (Edge Case)
// ============================================================================

/**
 * Component that dynamically updates client configuration
 * (Rare scenario - usually config is set once at startup)
 */
function ConfigUpdater() {
  const [projectId, setProjectId] = useState(1);

  const handleProjectChange = (newProjectId) => {
    setProjectId(newProjectId);
    
    // Update client configuration dynamically
    updateClientConfig({
      projectId: newProjectId
    });
    
    console.log('[ConfigUpdater] Updated project ID to:', newProjectId);
  };

  return (
    <div>
      <h2>Dynamic Configuration</h2>
      <p>Current Project ID: {projectId}</p>
      <button onClick={() => handleProjectChange(1)}>Project 1</button>
      <button onClick={() => handleProjectChange(2)}>Project 2</button>
      <button onClick={() => handleProjectChange(3)}>Project 3</button>
      <small>
        Note: Changing project ID requires appropriate permissions.
        Session cookies are still shared for authentication.
      </small>
    </div>
  );
}

// ============================================================================
// EXAMPLE 7: Error Handling with Session Issues
// ============================================================================

/**
 * Component that handles session-related errors gracefully
 */
function SessionErrorHandler() {
  const { artifacts, loading, error } = useArtifacts('test-bucket');
  const [sessionError, setSessionError] = useState(null);

  useEffect(() => {
    if (error) {
      // Check if error is session-related
      if (error.includes('401') || error.includes('Unauthorized')) {
        setSessionError('Session expired or not authenticated');
        console.error('[SessionError] Authentication failed - session may be expired');
        
        // In iframe, this might mean parent session expired
        // Could notify parent window to refresh
        if (window.self !== window.top) {
          window.parent.postMessage({ 
            type: 'SESSION_EXPIRED',
            source: 'deepwiki_plugin'
          }, window.location.origin);
        }
      } else if (error.includes('403') || error.includes('Forbidden')) {
        setSessionError('No permission to access this resource');
      } else {
        setSessionError('An error occurred');
      }
    } else {
      setSessionError(null);
    }
  }, [error]);

  if (sessionError) {
    return (
      <div style={{ 
        padding: '20px', 
        backgroundColor: '#ffebee', 
        border: '1px solid #f44336',
        borderRadius: '4px'
      }}>
        <h3>Session Error</h3>
        <p>{sessionError}</p>
        <button onClick={() => window.location.reload()}>
          Reload Page
        </button>
      </div>
    );
  }

  if (loading) return <div>Loading...</div>;

  return (
    <div>
      <h2>Artifacts (Session Active)</h2>
      <ul>
        {artifacts.map(artifact => (
          <li key={artifact.name}>{artifact.name}</li>
        ))}
      </ul>
    </div>
  );
}

// ============================================================================
// EXAMPLE 8: Complete Integration Example
// ============================================================================

/**
 * Main app component showing complete integration
 */
function DeepWikiPluginApp() {
  const [activeTab, setActiveTab] = useState('artifacts');

  // Detect iframe mode on mount
  useEffect(() => {
    const isIframe = window.self !== window.top;
    console.log('[DeepWikiApp] Iframe mode:', isIframe);
    
    if (isIframe) {
      console.log('[DeepWikiApp] Session cookies will be shared from parent automatically');
      console.log('[DeepWikiApp] No manual authentication required');
    }
  }, []);

  return (
    <div style={{ padding: '20px' }}>
      <h1>DeepWiki Plugin (Iframe Session Sharing)</h1>
      
      {/* Tab Navigation */}
      <div style={{ marginBottom: '20px' }}>
        <button onClick={() => setActiveTab('artifacts')}>Artifacts</button>
        <button onClick={() => setActiveTab('tools')}>Tools</button>
        <button onClick={() => setActiveTab('info')}>Iframe Info</button>
        <button onClick={() => setActiveTab('error')}>Error Handling</button>
      </div>

      {/* Tab Content */}
      {activeTab === 'artifacts' && <ArtifactViewer />}
      {activeTab === 'tools' && <ToolInvoker />}
      {activeTab === 'info' && <IframeDetector />}
      {activeTab === 'error' && <SessionErrorHandler />}
    </div>
  );
}

// ============================================================================
// Summary of Key Points
// ============================================================================

/*
KEY TAKEAWAYS:

1. **Automatic Session Sharing**
   - Session cookies are shared automatically when in iframe
   - No manual token passing needed
   - Works because parent and iframe are same origin

2. **Zero Configuration**
   - Just use getEliteAClient() or React hooks
   - Session authentication happens automatically
   - No postMessage communication required

3. **Error Handling**
   - Watch for 401/403 errors (session expired/no permission)
   - Can notify parent window if needed
   - Graceful degradation for session issues

4. **Development Testing**
   - Test in both iframe and standalone modes
   - Check browser console for iframe detection logs
   - Verify cookies in Network tab

5. **Best Practices**
   - Always use singleton client (getEliteAClient)
   - Prefer React hooks over direct client usage
   - Handle session errors gracefully
   - Log iframe mode in development

6. **Advanced Scenarios**
   - Use updateClientAuthToken for explicit token updates (rare)
   - Use updateClientConfig for dynamic configuration (rare)
   - Listen for postMessage for parent-child communication (optional)
*/

export {
  ArtifactViewer,
  DirectClientUsage,
  IframeDetector,
  TokenUpdateListener,
  ToolInvoker,
  ConfigUpdater,
  SessionErrorHandler,
  DeepWikiPluginApp
};
