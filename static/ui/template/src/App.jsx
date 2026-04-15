import { useEffect, useState, useMemo } from 'react';
import { ThemeProvider, createTheme, CssBaseline } from '@mui/material';
import { Box, Typography, Paper, CircularProgress, Alert } from '@mui/material';
import io from 'socket.io-client';

function App() {
  const [mode, setMode] = useState('light');
  const [toolkit, setToolkit] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [projectId, setProjectId] = useState(null);
  const [toolkitId, setToolkitId] = useState(null);
  const [socket, setSocket] = useState(null);
  const [socketStatus, setSocketStatus] = useState('disconnected');
  const [isDevelopment, setIsDevelopment] = useState(false);

  // Extract URL parameters and fetch toolkit data
  useEffect(() => {
    // Get theme from URL parameter
    const urlParams = new URLSearchParams(window.location.search);
    const themeParam = urlParams.get('theme');
    const toolkitIdParam = urlParams.get('toolkit_id');
    const projectIdParam = urlParams.get('project_id');
    
    if (themeParam === 'dark' || themeParam === 'light') {
      setMode(themeParam);
    }

    // Extract project_id from URL path
    // URL format: /ui_host/{provider}/{ui_name}/{project_id}/...
    const pathParts = window.location.pathname.split('/');
    const uiHostIndex = pathParts.indexOf('ui_host');
    
    let extractedProjectId = null;
    let extractedToolkitId = toolkitIdParam;
    let devMode = false;
    
    if (uiHostIndex !== -1 && pathParts.length > uiHostIndex + 3) {
      // Production mode: running through ui_host proxy
      extractedProjectId = pathParts[uiHostIndex + 3];
      devMode = false;
    } else {
      // Development mode: use URL params or env defaults
      devMode = true;
      extractedProjectId = projectIdParam || import.meta.env.VITE_DEFAULT_PROJECT_ID;
      extractedToolkitId = extractedToolkitId || import.meta.env.VITE_DEFAULT_TOOLKIT_ID;
    }

    setProjectId(extractedProjectId);
    setToolkitId(extractedToolkitId);
    setIsDevelopment(devMode);

    // Fetch toolkit details if we have both IDs
    if (extractedProjectId && extractedToolkitId) {
      fetchToolkitDetails(extractedProjectId, extractedToolkitId);
    } else {
      setLoading(false);
      
      // Only show error if we're missing IDs
      if (!extractedProjectId || !extractedToolkitId) {
        setError(
          `Missing required parameters. Add to URL: ?project_id=X&toolkit_id=Y or set in .env file`
        );
      }
    }
  }, []);

  // Initialize Socket.IO connection
  useEffect(() => {
    const newSocket = io({
      path: '/socket.io',
      transports: ['websocket', 'polling'],
      withCredentials: true, // Important: include session cookies
    });

    newSocket.on('connect', () => {
      console.log('Socket.IO connected');
      setSocketStatus('connected');
    });

    newSocket.on('disconnect', () => {
      console.log('Socket.IO disconnected');
      setSocketStatus('disconnected');
    });

    newSocket.on('connect_error', (error) => {
      console.error('Socket.IO connection error:', error);
      setSocketStatus('error');
    });

    setSocket(newSocket);

    return () => {
      newSocket.close();
    };
  }, []);

  const fetchToolkitDetails = async (pid, tid) => {
    try {
      setLoading(true);
      
      // Prepare headers - add Authorization in dev mode like EliteAUI
      const headers = {};
      const devToken = import.meta.env.VITE_DEV_TOKEN;
      if (import.meta.env.DEV && devToken) {
        headers['Authorization'] = `Bearer ${devToken}`;
        headers['Cache-Control'] = 'no-cache';
      }
      
      const response = await fetch(
        `/api/v2/elitea_core/tool/prompt_lib/${pid}/${tid}`,
        {
          method: 'GET',
          credentials: 'include',
          headers,
        }
      );

      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(`HTTP ${response.status}: ${response.statusText} - ${errorText}`);
      }

      const data = await response.json();
      setToolkit(data);
      setError(null);
    } catch (err) {
      console.error('Error fetching toolkit:', err);
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  // Create MUI theme based on mode
  const theme = useMemo(
    () =>
      createTheme({
        palette: {
          mode,
        },
      }),
    [mode],
  );

  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <Box sx={{ p: 3, minHeight: '100vh' }}>
        <Typography variant="h4" gutterBottom>
          DeepWiki Plugin Custom UI
        </Typography>

        <Box sx={{ mb: 3 }}>
          <Typography variant="body2" color="text.secondary">
            Theme: {mode} | Project ID: {projectId || 'N/A'} | Toolkit ID: {toolkitId || 'N/A'}
          </Typography>
          <Typography variant="body2" color="text.secondary">
            Socket.IO Status: {socketStatus}
          </Typography>
          {isDevelopment && (
            <Alert severity="info" sx={{ mt: 1 }}>
              <strong>Development Mode:</strong> Running locally. 
              Use URL params: <code>?project_id=X&toolkit_id=Y</code> or configure .env with VITE_DEFAULT_PROJECT_ID and VITE_DEFAULT_TOOLKIT_ID.
            </Alert>
          )}
        </Box>

        {loading && (
          <Box sx={{ display: 'flex', justifyContent: 'center', p: 3 }}>
            <CircularProgress />
          </Box>
        )}

        {error && (
          <Alert severity="warning" sx={{ mb: 2 }}>
            Error: {error}
          </Alert>
        )}

        {!loading && !error && toolkit && (
          <Paper sx={{ p: 3 }}>
            <Typography variant="h6" gutterBottom>
              Toolkit Details
            </Typography>
            <Typography variant="body1">
              <strong>Name:</strong> {toolkit.name || 'N/A'}
            </Typography>
            <Typography variant="body1">
              <strong>Description:</strong> {toolkit.description || 'N/A'}
            </Typography>
            <Typography variant="body1">
              <strong>Type:</strong> {toolkit.type || 'N/A'}
            </Typography>
            <Box sx={{ mt: 2 }}>
              <Typography variant="body2" color="text.secondary">
                Settings:
              </Typography>
              <pre style={{ 
                background: mode === 'dark' ? '#1e1e1e' : '#f5f5f5',
                padding: '12px',
                borderRadius: '4px',
                overflow: 'auto',
                fontSize: '12px'
              }}>
                {JSON.stringify(toolkit.settings, null, 2)}
              </pre>
            </Box>
          </Paper>
        )}

        {!loading && !error && !toolkit && (
          <Alert severity="info">
            No toolkit data available. This is a template UI.
          </Alert>
        )}

        <Box sx={{ mt: 3 }}>
          <Typography variant="h6" gutterBottom>
            Example Features
          </Typography>
          <Typography variant="body2" paragraph>
            This is a starter template demonstrating:
          </Typography>
          <ul>
            <li>Automatic theme switching (light/dark) based on URL parameter</li>
            <li>REST API calls with session authentication</li>
            <li>Socket.IO connection with session authentication</li>
            <li>URL parameter extraction (project_id, toolkit_id, theme)</li>
            <li>MUI component integration</li>
            <li>Error handling and loading states</li>
          </ul>
          <Typography variant="body2" color="text.secondary">
            Modify src/App.jsx to build your custom UI. See README.md for detailed documentation.
          </Typography>
        </Box>
      </Box>
    </ThemeProvider>
  );
}

export default App;
