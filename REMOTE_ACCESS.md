# Remote Access Configuration

## Server Information

**Server IP**: Check with `hostname -I`
**Backend Port**: 5000
**Frontend Port**: 5173

## Quick Start

### 1. Start the Backend (Terminal 1)

```bash
./run_backend.sh
```

The backend will be accessible at:
- `http://localhost:5000`
- `http://<SERVER_IP>:5000`

### 2. Configure Frontend for Remote Access

Edit `stego-game/src/App.jsx` and update line 21:

```javascript
// Change this:
const response = await fetch('http://localhost:5000/api/generate', {

// To this (replace with your server IP):
const response = await fetch('http://172.17.0.3:5000/api/generate', {
```

### 3. Start the Frontend (Terminal 2)

```bash
cd stego-game
./run_frontend.sh
```

The frontend will be accessible at:
- `http://localhost:5173`
- `http://<SERVER_IP>:5173`

### 4. Access from Remote Machine

Open your browser to:
```
http://<SERVER_IP>:5173
```

Replace `<SERVER_IP>` with your actual server IP address.

## Testing the Backend

### Health Check
```bash
curl http://localhost:5000/health
```

Expected response:
```json
{
  "model": "google/gemma-3-1b-it",
  "status": "ok"
}
```

### Test Generation
```bash
curl -X POST http://localhost:5000/api/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "what is a mitochondria", "temperature": 1.3}'
```

## Port Forwarding (if needed)

If accessing from outside the local network, you may need to set up port forwarding or open firewall ports:

```bash
# Ubuntu/Debian with ufw
sudo ufw allow 5000/tcp  # Backend
sudo ufw allow 5173/tcp  # Frontend
```

## Current Server Status

Backend is running on:
- http://127.0.0.1:5000
- http://172.17.0.3:5000

To check backend logs:
```bash
# View logs from the background process
tail -f logs/backend.log  # if logging to file
```

## Architecture

```
┌─────────────────┐
│  Remote Client  │
│   (Browser)     │
└────────┬────────┘
         │
         │ HTTP Request to :5173
         ▼
┌─────────────────┐
│  Vite Dev       │
│  Server         │
│  (Frontend)     │
└────────┬────────┘
         │
         │ API Call to :5000/api/generate
         ▼
┌─────────────────┐
│  Flask Server   │
│  (Backend)      │
│                 │
│  ┌───────────┐  │
│  │  Vanilla  │  │
│  │  (Trans-  │  │
│  │  formers) │  │
│  └───────────┘  │
│                 │
│  ┌───────────┐  │
│  │   Stego   │  │
│  │ (Encoder) │  │
│  └───────────┘  │
└─────────────────┘
```

## Troubleshooting

### Cannot connect from remote machine

1. **Check if server is listening on 0.0.0.0**:
   ```bash
   netstat -tulpn | grep -E '5000|5173'
   ```

2. **Check firewall**:
   ```bash
   sudo ufw status
   ```

3. **Test from server itself**:
   ```bash
   curl http://localhost:5000/health
   ```

### CORS errors in browser

- Make sure Flask backend has CORS enabled (already configured)
- Check browser console for specific error messages

### Frontend can't reach backend

- Verify backend is running: `curl http://localhost:5000/health`
- Check API URL in `App.jsx` matches your server IP
- Look for network errors in browser console

## Performance Tips

- **First request takes 30-60 seconds** (model loading)
- Subsequent requests are faster (10-30 seconds)
- Run backend on a machine with GPU for best performance
- Consider adding a loading indicator for better UX
