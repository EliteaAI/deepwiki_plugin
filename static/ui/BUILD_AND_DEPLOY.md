# DeepWiki Plugin UI - Build and Deployment Guide

## Overview

The DeepWiki plugin includes a custom UI that must be built before deployment. The build process compiles the React application into static files that are served by the plugin's `/ui` route.

## Build Process

### 1. Local Build

Build the UI locally for testing or manual deployment:

```bash
cd /path/to/deepwiki/static/ui/template
npm install
npm run build
```

**Output**: Built files will be in `static/ui/dist/`

**Build configuration** (from `vite.config.js`):
- Output directory: `../dist` (relative to template/)
- Source maps: enabled
- Empty output dir before build: yes

### 2. Automated Build (GitHub Actions)

The `.github/workflows/build-ui.yml` workflow automatically builds the UI when:
- Push to `main` or `develop` branches (if UI files changed)
- Pull request to `main` or `develop` (if UI files changed)
- Manual trigger via `workflow_dispatch`

**Workflow details**:
- Tests on Node.js 18.x and 20.x
- Builds on Ubuntu latest
- Validates build output (checks for `dist/index.html`)
- Uploads build artifacts (retention: 7 days)
- Auto-commits built files to `main` branch (with `[skip ci]`)

### 3. Verify Build Output

After building, verify the output structure:

```bash
ls -la static/ui/dist/

# Expected files:
# - index.html (main HTML file)
# - assets/ (CSS, JS, and other assets with hashed names)
# - vite.svg (favicon)
```

## Deployment Options

### Option 1: Deploy with Built Files (Recommended for Production)

1. **Build the UI**:
   ```bash
   cd static/ui/template
   npm install
   npm run build
   ```

2. **Commit built files** (if not using GitHub Actions):
   ```bash
   git add static/ui/dist/
   git commit -m "chore: update UI build artifacts"
   git push
   ```

3. **Deploy plugin** with built files included in repository

4. **Plugin serves UI** via `/ui` route from `static/ui/dist/`

**Advantages**:
- No build step required on deployment server
- Faster deployment
- No Node.js required on production server
- Works with standard Python plugin deployment

### Option 2: Build on Deployment Server

1. **Deploy plugin** without built files

2. **On deployment server**:
   ```bash
   cd /path/to/deepwiki/static/ui/template
   npm install --production=false
   npm run build
   ```

3. **Restart plugin/service** to load new UI

**Advantages**:
- Smaller repository size (no built files in git)
- Build uses server's environment

**Disadvantages**:
- Requires Node.js on deployment server
- Longer deployment time
- More complex deployment process

### Option 3: CI/CD Pipeline Build

1. **CI/CD pipeline** builds UI as part of deployment:
   ```yaml
   # Example GitLab CI / GitHub Actions step
   - name: Build Plugin UI
     run: |
       cd static/ui/template
       npm ci
       npm run build
   ```

2. **Package plugin** with built UI

3. **Deploy packaged plugin** to server

## Production Configuration

### Environment Variables

For production deployment, the UI uses **session-based authentication** via the `ui_host.py` proxy route. No environment variables are needed in production.

**Development only** (`.env` file):
```bash
VITE_DEV_SERVER=https://dev.elitea.ai
VITE_DEV_TOKEN=<your-bearer-token>
VITE_DEFAULT_PROJECT_ID=5
VITE_DEFAULT_TOOLKIT_ID=42
```

### Build Settings

The build is configured via `vite.config.js`:

```javascript
build: {
  outDir: '../dist',        // Output to static/ui/dist/
  emptyOutDir: true,        // Clean before build
  sourcemap: true,          // Include source maps for debugging
}
```

**For production builds**, you may want to disable source maps:

```javascript
build: {
  outDir: '../dist',
  emptyOutDir: true,
  sourcemap: false,  // Disable for production
  minify: 'terser',  // Use terser for better minification
}
```

## Plugin Route Configuration

The plugin serves the built UI via `routes/ui.py`:

```python
@bp.route('/ui', defaults={'path': ''})
@bp.route('/ui/<path:path>')
def serve_ui(path):
    """Serve the custom UI"""
    ui_dir = os.path.join(os.path.dirname(__file__), '..', 'static', 'ui', 'dist')
    
    if path == '':
        return send_from_directory(ui_dir, 'index.html')
    
    # Serve static files
    if os.path.exists(os.path.join(ui_dir, path)):
        return send_from_directory(ui_dir, path)
    
    # Fallback to index.html for SPA routing
    return send_from_directory(ui_dir, 'index.html')
```

**Important**: The UI is accessed via the `ui_host.py` proxy route:
```
https://your-domain.com/app/ui_host/deepwiki/ui/{project_id}/
```

This route automatically injects custom headers (X-User-Id, X-Project-Id, etc.) for authentication.

## Deployment Checklist

### Pre-Deployment

- [ ] Build UI locally and test
- [ ] Verify all API calls work with production backend
- [ ] Check theme switching (light/dark)
- [ ] Test authentication flow
- [ ] Verify Socket.IO connection
- [ ] Check console for errors

### Deployment

- [ ] Build production version (sourcemaps disabled)
- [ ] Commit built files to repository OR
- [ ] Configure CI/CD to build UI
- [ ] Deploy plugin with built UI
- [ ] Verify `static/ui/dist/` exists on server
- [ ] Restart plugin/service

### Post-Deployment

- [ ] Access UI via ui_host route
- [ ] Verify theme parameter works
- [ ] Check API calls with session auth
- [ ] Test Socket.IO connection
- [ ] Monitor browser console for errors
- [ ] Verify custom headers are injected

## Troubleshooting

### Build Fails

**Issue**: `npm run build` fails with dependency errors

**Solution**:
```bash
rm -rf node_modules package-lock.json
npm install
npm run build
```

### UI Not Loading

**Issue**: Accessing `/app/ui_host/deepwiki/ui/{project_id}/` returns 404

**Solutions**:
1. Verify built files exist: `ls static/ui/dist/index.html`
2. Check plugin descriptor has correct UI configuration
3. Verify ui_host.py route is working
4. Check plugin service logs

### Blank Page

**Issue**: UI loads but shows blank page

**Solutions**:
1. Check browser console for errors
2. Verify vite.config.js `base` path is correct (should be `/`)
3. Check network tab for 404s on asset files
4. Verify index.html exists in dist/

### API Calls Failing

**Issue**: API calls return 401/403 errors

**Solutions**:
1. Verify accessing via ui_host route (not direct plugin URL)
2. Check custom headers are injected (X-User-Id, etc.)
3. Verify session authentication is working
4. Check CORS settings if calling external APIs

### Theme Not Switching

**Issue**: Theme parameter in URL doesn't change UI theme

**Solutions**:
1. Check URL has `?theme=dark` or `?theme=light`
2. Verify App.jsx extracts theme from URL params
3. Check MUI ThemeProvider uses extracted theme
4. Look for JavaScript errors in console

## Build Optimization

### Reduce Bundle Size

1. **Code splitting**:
   ```javascript
   // vite.config.js
   build: {
     rollupOptions: {
       output: {
         manualChunks: {
           'react-vendor': ['react', 'react-dom'],
           'mui-vendor': ['@mui/material', '@mui/icons-material'],
         }
       }
     }
   }
   ```

2. **Remove unused dependencies**:
   ```bash
   npm prune
   ```

3. **Analyze bundle**:
   ```bash
   npm install --save-dev rollup-plugin-visualizer
   npm run build
   # View stats.html
   ```

### Performance Tuning

1. **Enable compression** (if serving directly from plugin):
   ```python
   from flask import send_from_directory, make_response
   import gzip
   
   # In ui.py route
   response = make_response(send_from_directory(ui_dir, path))
   response.headers['Content-Encoding'] = 'gzip'
   ```

2. **Cache static assets**:
   ```python
   response.headers['Cache-Control'] = 'public, max-age=31536000'
   ```

## CI/CD Integration Examples

### GitHub Actions

Already configured in `.github/workflows/build-ui.yml`

### GitLab CI

```yaml
build-ui:
  stage: build
  image: node:20
  script:
    - cd static/ui/template
    - npm ci
    - npm run build
  artifacts:
    paths:
      - static/ui/dist/
    expire_in: 1 week
```

### Manual Deployment Script

```bash
#!/bin/bash
# deploy_ui.sh

set -e

echo "Building DeepWiki Plugin UI..."
cd static/ui/template
npm install
npm run build

echo "Build complete! Output in static/ui/dist/"
ls -la ../dist/

echo "Ready to deploy!"
```

## Summary

**Recommended production workflow**:
1. Use GitHub Actions to auto-build UI on push to main
2. Include built files in repository
3. Deploy plugin with pre-built UI files
4. UI is served via ui_host.py proxy route
5. Session-based authentication (no tokens needed)

This approach minimizes deployment complexity and ensures the UI is always production-ready.
