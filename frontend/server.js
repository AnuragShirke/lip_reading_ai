const express = require('express');
const { createProxyMiddleware } = require('http-proxy-middleware');
const path = require('path');

const app = express();
const PORT = process.env.PORT || 3000;

// Define the backend URL
const BACKEND_URL = process.env.BACKEND_URL || 'https://lip-reading-backend.onrender.com';

// Serve static files from the dist directory
app.use(express.static(path.join(__dirname, 'dist')));

// Set up proxy for API requests
app.use('/api', createProxyMiddleware({
  target: BACKEND_URL,
  changeOrigin: true,
  pathRewrite: {
    '^/api': '', // Remove /api prefix when forwarding to backend
  },
  onProxyReq: (proxyReq, req, res) => {
    // Log proxy requests
    console.log(`Proxying ${req.method} ${req.url} to ${BACKEND_URL}`);
  },
  onError: (err, req, res) => {
    console.error('Proxy error:', err);
    res.status(500).send('Proxy error: ' + err.message);
  }
}));

// Handle SPA routing - serve index.html for all other routes
app.get('*', (req, res) => {
  res.sendFile(path.join(__dirname, 'dist', 'index.html'));
});

// Start the server
app.listen(PORT, () => {
  console.log(`Frontend server running on port ${PORT}`);
  console.log(`Proxying API requests to ${BACKEND_URL}`);
});
