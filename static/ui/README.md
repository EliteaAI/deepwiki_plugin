# Plugin Custom UI Development Guide

This guide explains how to create custom user interfaces for your plugin that integrate with the Elitea platform.

## Overview

Custom plugin UIs are React applications served from your plugin that run in an iframe within the main Elitea UI. They have full access to Elitea's REST APIs and Socket.IO connections using the user's authenticated session.

## Directory Structure

```
deepwiki_plugin/
├── static/
│   └── ui/
│       ├── dist/              # Build output (served to users)
│       ├── template/          # Development source code
│       │   ├── src/           # React source code
│       │   ├── .env.example   # Environment variables template
│       │   ├── package.json
│       │   └── vite.config.js
│       └── README.md         # This file
├── routes/
│   ├── descriptor.py         # Plugin descriptor
│   └── ui.py                # UI route handler
```

## Local Development Setup

### 1. Environment Configuration

Create a `.env` file in the `template/` directory:

```bash
cd static/ui/template
cp .env.example .env
```

Edit `.env` and configure your remote Elitea server:

```env
# Point to your Elitea instance
VITE_DEV_SERVER=https://dev.elitea.ai

# API and Socket.IO paths (usually don't need to change)
VITE_SERVER_URL=/api/v2
VITE_SOCKET_PATH=/socket.io/

# Optional: Add auth token if required
# VITE_DEV_TOKEN=your-jwt-token-here
```

### 2. Install Dependencies

```bash
cd static/ui/template
npm install
```

### 3. Run Development Server

```bash
npm run dev
```

The UI will start on `http://localhost:5174` and automatically proxy:
- All `/api/*` requests to your remote server
- All `/socket.io/*` WebSocket connections to your remote server
- Session authentication will be forwarded automatically

### 4. Local Testing

Open `http://localhost:5174` in your browser. The app will:
- Connect to the remote Elitea server for API calls
- Establish WebSocket connection via Socket.IO
- Use session authentication from the remote server

**Note**: When running standalone (not in iframe), you may need to authenticate with the remote server first by logging in to the main Elitea UI at the configured `VITE_DEV_SERVER`.

## Production Deployment

### 1. Plugin Descriptor Setup

In `routes/descriptor.py`, add:

```python
descriptor = {
    "name": "YourServiceProvider",
    "provided_toolkits": [
        {
            "name": "YourToolkit",
            "toolkit_metadata": {
                "application": True,              # Mark as application
                "custom_ui_route": "your_ui"      # UI route name
            }
        }
    ],
    "provided_ui": [
        {
            "name": "your_ui",                    # Must match custom_ui_route
            "path": "/ui",                        # Path to UI route (relative to service_location_url)
            "headers": {                          # Optional: Custom headers to inject
                "X-User-Id": {"type": "user_id"},
                "X-User-Name": {"type": "user_name"},
                "X-Project-Id": {"type": "project_id"},
                "X-Project-Name": {"type": "project_name"}
            }
        }
    ]
}
```

### 2. UI Route Handler

Create `routes/ui.py`:

```python
from flask import send_from_directory
from pylon.core.tools import web
import os

class Route:
    @web.route("/ui/", defaults={"path": "index.html"})
    @web.route("/ui/<path:path>")
    def ui_route(self, path):
        plugin_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        static_dir = os.path.join(plugin_dir, "static", "ui", "dist")
        
        full_path = os.path.join(static_dir, path)
        if os.path.isdir(full_path) or not os.path.exists(full_path):
            path = "index.html"
        
        return send_from_directory(static_dir, path)
```

### 3. React Application Setup

Copy the starter template:

```bash
cd deepwiki_plugin/static/ui
cp -r template/* .
npm install
```

## URL Parameters

Your custom UI receives these parameters in the URL:

- `theme` - Current theme mode (`light` or `dark`)
- `toolkit_id` - ID of the current toolkit instance
- `project_id` - Available in the URL path: `/ui_host/{provider}/{ui}/{project_id}/`

### Extracting Parameters

```javascript
// Get URL parameters
const urlParams = new URLSearchParams(window.location.search);
const theme = urlParams.get('theme') || 'light';
const toolkitId = urlParams.get('toolkit_id');

// Get project_id from URL path
const pathParts = window.location.pathname.split('/');
const uiHostIndex = pathParts.indexOf('ui_host');

if (uiHostIndex !== -1 && pathParts.length > uiHostIndex + 3) {
  // Production mode: running through ui_host proxy
  const projectId = pathParts[uiHostIndex + 3];
} else {
  // Development mode: running standalone
  // Use mock data or connect to remote server via .env configuration
  const projectId = 'dev-project-123';  // Mock value
}
```

### Development vs Production Mode

The template automatically detects whether it's running in development or production:

**Production Mode** (through ui_host proxy):
- URL contains `/ui_host/{provider}/{ui}/{project_id}/`
- Session authentication works automatically
- All API calls go to the same server
- Socket.IO connects to the same server

**Development Mode** (npm run dev):
- URL is typically `http://localhost:5173/`
- Mock data is displayed by default
- To connect to a remote server, configure `.env` (see Local Development section)
- Socket.IO may not connect unless proxy is configured

The starter template includes logic to handle both modes gracefully.

## Authentication

Your custom UI automatically inherits the user's session. No additional authentication is required.

### REST API Calls

Use `fetch` with `credentials: 'include'` to make authenticated requests:

```javascript
// Example: Fetch toolkit details
const response = await fetch(
  `/api/v2/elitea_core/tool/prompt_lib/${projectId}/${toolkitId}`,
  {
    credentials: 'include',  // Include session cookies
    headers: {
      'Content-Type': 'application/json',
    },
  }
);

const toolkit = await response.json();
```

### Common API Endpoints

```javascript
// Get toolkit details
GET /api/v2/elitea_core/tool/prompt_lib/{project_id}/{toolkit_id}

// Update toolkit
PUT /api/v2/elitea_core/tool/prompt_lib/{project_id}/{toolkit_id}
Body: { name, description, settings, ... }

// List toolkits
GET /api/v2/tools/prompt_lib/{project_id}?application=true

// Get toolkit types
GET /api/v2/toolkit_types/prompt_lib/{project_id}?application=true
```

Refer to Postman documentation for complete API reference.

### Socket.IO Connection

```javascript
import io from 'socket.io-client';

const socket = io({
  path: '/socket.io',
  transports: ['websocket', 'polling'],
  withCredentials: true,  // Include session cookies
});

socket.on('connect', () => {
  console.log('Connected to Socket.IO');
});

socket.on('your_event', (data) => {
  console.log('Received:', data);
});

// Emit events
socket.emit('your_event', { data: 'value' });
```

## MUI Theming

Match the main Elitea UI theme for consistency:

```javascript
import { createTheme, ThemeProvider } from '@mui/material/styles';
import { useMemo, useEffect, useState } from 'react';

function App() {
  const [mode, setMode] = useState('light');

  // Read theme from URL parameter
  useEffect(() => {
    const urlParams = new URLSearchParams(window.location.search);
    const themeParam = urlParams.get('theme');
    if (themeParam === 'dark' || themeParam === 'light') {
      setMode(themeParam);
    }
  }, []);

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
      {/* Your app components */}
    </ThemeProvider>
  );
}
```

For full Elitea theme compatibility, copy theme definitions from `EliteAUI/src/lightPalette.js` and `EliteAUI/src/darkPalette.js`.

## Local Development

### Development Server with API Proxy

Configure Vite to proxy API requests during development:

```javascript
// vite.config.js
export default {
  server: {
    proxy: {
      '/api': {
        target: 'https://your-elitea-instance.com',
        changeOrigin: true,
        secure: false,
      },
      '/socket.io': {
        target: 'https://your-elitea-instance.com',
        changeOrigin: true,
        ws: true,
      },
    },
  },
};
```

### Run Development Server

```bash
npm run dev
```

Access at `http://localhost:5173/?theme=light&toolkit_id=5`

### Hot Reload

Vite provides instant hot module replacement (HMR). Changes appear immediately without full page reload.

## Building for Production

```bash
npm run build
```

Output goes to `dist/` directory, which is served by the plugin's UI route.

### Build Checklist

- ✅ All API calls use `credentials: 'include'`
- ✅ Socket.IO uses `withCredentials: true`
- ✅ Theme switching based on URL parameter works
- ✅ No hardcoded URLs (use relative paths)
- ✅ Build output is in `static/ui/dist/`

## Debugging

### Browser DevTools

1. Open Elitea UI in browser
2. Navigate to your custom app
3. Right-click iframe → Inspect Element
4. Use Console, Network, and Elements tabs as normal

### Common Issues

**API calls return 401 Unauthorized:**
- Ensure `credentials: 'include'` is set
- Check that cookies are not blocked

**Theme not updating:**
- Verify URL parameter is being read correctly
- Check theme mode is passed to ThemeProvider

**Resources not loading:**
- Check build output in `dist/` directory
- Verify file paths are relative, not absolute
- Check browser console for 404 errors

## Example: Complete Minimal App

```jsx
// src/App.jsx
import { useEffect, useState, useMemo } from 'react';
import { ThemeProvider, createTheme, CssBaseline, Box, Typography } from '@mui/material';

function App() {
  const [mode, setMode] = useState('light');
  const [toolkit, setToolkit] = useState(null);

  // Extract parameters
  useEffect(() => {
    const urlParams = new URLSearchParams(window.location.search);
    const theme = urlParams.get('theme') || 'light';
    const toolkitId = urlParams.get('toolkit_id');
    const pathParts = window.location.pathname.split('/');
    const projectId = pathParts[pathParts.indexOf('ui_host') + 3];

    setMode(theme);

    // Fetch toolkit details
    if (projectId && toolkitId) {
      fetch(`/api/v2/elitea_core/tool/prompt_lib/${projectId}/${toolkitId}`, {
        credentials: 'include',
      })
        .then(res => res.json())
        .then(data => setToolkit(data))
        .catch(console.error);
    }
  }, []);

  const theme = useMemo(() => createTheme({ palette: { mode } }), [mode]);

  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <Box p={3}>
        <Typography variant="h4">My Custom App</Typography>
        {toolkit && (
          <Typography variant="body1">
            Toolkit: {toolkit.name}
          </Typography>
        )}
      </Box>
    </ThemeProvider>
  );
}

export default App;
```

## Best Practices

1. **Keep UIs lightweight** - Users may have many tabs open
2. **Handle errors gracefully** - Show user-friendly messages
3. **Use MUI components** - Maintain visual consistency
4. **Optimize bundle size** - Tree-shake unused code
5. **Test in both themes** - Ensure readability in light and dark modes
6. **Respect session state** - Custom UI manages its own state independently
7. **Use relative URLs** - Never hardcode domain names

## Deployment

1. Build your UI: `npm run build`
2. Commit `dist/` folder to plugin repository
3. Deploy plugin to Elitea instance
4. UI is automatically available at configured route

## Support

- **API Documentation:** Check Postman collection
- **MUI Documentation:** https://mui.com/
- **Vite Documentation:** https://vitejs.dev/
- **Elitea Issues:** File issues in main repository

## Migration Notes

### From Older Versions

If you're updating an existing plugin:

1. Add `toolkit_metadata.application = True` to descriptor
2. Add `toolkit_metadata.custom_ui_route` to descriptor
3. Add `provided_ui` section to descriptor root
4. Create UI route handler in `routes/ui.py`
5. Build UI to `static/ui/dist/`

### Recommended MUI Version

Current EliteAUI uses MUI v5. Match the major version for best compatibility:

```json
{
  "dependencies": {
    "@mui/material": "^5.0.0",
    "@mui/icons-material": "^5.0.0",
    "@emotion/react": "^11.0.0",
    "@emotion/styled": "^11.0.0"
  }
}
```

---

**Last Updated:** December 2025
