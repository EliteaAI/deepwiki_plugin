# Quick Start Guide - Testing with Real Backend Data

## Current Configuration

Your `.env` file is already configured to connect to the backend:
- Server: `https://dev.elitea.ai`
- Auth Token: ✅ Configured
- Default Project ID: `5`
- Default Toolkit ID: `42`

## How to Run

### Option 1: Using Default IDs from .env
Simply start the dev server:
```bash
npm run dev
```
Then open: http://localhost:5174/

The UI will automatically use:
- project_id=5
- toolkit_id=42

### Option 2: Using URL Parameters
Start the dev server:
```bash
npm run dev
```
Then open with specific IDs:
```
http://localhost:5174/?project_id=5&toolkit_id=123
```

### Option 3: Testing Different Themes
```
http://localhost:5174/?project_id=5&toolkit_id=123&theme=dark
```

## What Will Happen

1. ✅ App will connect to `https://dev.elitea.ai`
2. ✅ API call will be made to `/api/v2/applications/tool/prompt_lib/5/42`
3. ✅ Auth token will be included in the request
4. ✅ Real toolkit data will be fetched and displayed
5. ✅ Socket.IO will connect to the backend

## Troubleshooting

### If you see "401 Unauthorized"
Your auth token may have expired. To get a new one:
1. Open https://dev.elitea.ai in your browser
2. Login to Elitea
3. Open DevTools (F12) > Application > Cookies
4. Copy the value of the `token` cookie
5. Update `VITE_DEV_TOKEN` in `.env`
6. Restart the dev server

### If you see "404 Not Found"
The project_id or toolkit_id doesn't exist. Try different IDs:
```
http://localhost:5174/?project_id=1&toolkit_id=1
```

### If you see CORS errors
The proxy should handle this. Check the terminal for proxy logs.

## Next Steps

Once you verify real data is loading:
1. Modify `src/App.jsx` to build your custom UI
2. Use the `toolkit` state object which contains real backend data
3. Make API calls using the `fetchToolkitDetails` pattern
4. Deploy by running `npm run build` (outputs to `../dist/`)

## Checking if it works

Look for these signs of success:
- ✅ No "Development Mode" alert (or it shows real data)
- ✅ "Toolkit Details" section shows real toolkit name/description
- ✅ "Socket.IO Status" shows "connected"
- ✅ Browser DevTools Network tab shows successful API calls

## Example: Finding Valid IDs

To find valid project and toolkit IDs from your backend:

```bash
# Login to your Elitea instance and run these API calls in browser console:

// Get all projects
fetch('/api/v2/projects', {credentials: 'include'})
  .then(r => r.json())
  .then(data => console.table(data.map(p => ({id: p.id, name: p.name}))))

// Get toolkits for project 5
fetch('/api/v2/tools/prompt_lib/5?application=true', {credentials: 'include'})
  .then(r => r.json())
  .then(data => console.table(data.map(t => ({id: t.id, name: t.name}))))
```
