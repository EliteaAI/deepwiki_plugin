import { defineConfig, loadEnv } from 'vite'
import react from '@vitejs/plugin-react'

// https://vitejs.dev/config/
export default ({ mode }) => {
  // Load env file based on mode
  const env = loadEnv(mode, process.cwd(), '')
  
  const {
    VITE_DEV_SERVER = 'https://dev.elitea.ai',
    VITE_SERVER_URL = '/api/v2',
    VITE_SOCKET_PATH = '/socket.io/',
    VITE_DEV_TOKEN = '',
  } = env

  return defineConfig({
    plugins: [react()],
    base: './',
    build: {
      outDir: '../dist',
      emptyOutDir: true,
      sourcemap: true,
    },
    server: {
      port: 5174, // Different port from main EliteAUI to avoid conflicts
      proxy: {
        // Proxy API requests to remote server
        [VITE_SERVER_URL]: {
          target: VITE_DEV_SERVER,
          changeOrigin: true,
          secure: false,
          ws: true,
          configure: (proxy, _options) => {
            proxy.on('error', (err, _req, _res) => {
              console.error('Proxy error:', err);
            });
            proxy.on('proxyReq', (proxyReq, req, _res) => {
              console.log('Proxying request:', req.method, req.url);
            });
            proxy.on('proxyRes', (proxyRes, req, _res) => {
              console.log('Received response:', proxyRes.statusCode, req.url);
            });
          },
        },
        // Proxy Socket.IO requests
        [VITE_SOCKET_PATH]: {
          target: VITE_DEV_SERVER,
          changeOrigin: true,
          secure: false,
          ws: true, // Enable WebSocket proxying
          configure: (proxy, _options) => {
            proxy.on('error', (err, _req, _res) => {
              console.error('Socket.IO proxy error:', err);
            });
            proxy.on('proxyReq', (proxyReq, req, _res) => {
              if (VITE_DEV_TOKEN) {
                proxyReq.setHeader('Authorization', `Bearer ${VITE_DEV_TOKEN}`);
              }
              console.log('Proxying socket request:', req.method, req.url);
            });
          },
        },
      },
    },
  })
}
