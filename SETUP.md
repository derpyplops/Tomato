# Steganography Guessing Game - Setup Guide

## Overview

This project consists of:
- **Frontend**: React app with Vite (port 5173)
- **Backend**: Flask API server (port 5000)

Both servers are configured to allow remote access (bound to 0.0.0.0).

## Prerequisites

- Python 3.10+
- Node.js and pnpm
- uv (Python package manager)
- CUDA-capable GPU (for model inference)

## Installation

### 1. Install Python Dependencies

```bash
uv sync
```

This will install:
- Flask and flask-cors
- PyTorch with CUDA support
- Transformers
- All other dependencies

### 2. Install Frontend Dependencies

```bash
cd stego-game
pnpm install
```

## Running the Application

### Option 1: Run Backend and Frontend Separately (Recommended)

**Terminal 1 - Backend:**
```bash
./run_backend.sh
```
This starts the Flask server on `http://0.0.0.0:5000`

**Terminal 2 - Frontend:**
```bash
cd stego-game
./run_frontend.sh
```
This starts the React dev server on `http://0.0.0.0:5173`

### Option 2: Run Manually

**Backend:**
```bash
uv run python backend/app.py
```

**Frontend:**
```bash
cd stego-game
pnpm run dev
```

## Remote Access Configuration

### For Testing from Another Machine:

1. **Find your server's IP address:**
   ```bash
   hostname -I
   ```
   Or:
   ```bash
   ip addr show
   ```

2. **Make sure ports are accessible:**
   - Port 5000 (Backend API)
   - Port 5173 (Frontend)

3. **Update Frontend API URL:**

   Edit `stego-game/src/App.jsx` line 21:
   ```javascript
   const response = await fetch('http://YOUR_SERVER_IP:5000/api/generate', {
   ```
   Replace `localhost` with your server's IP address.

4. **Access the frontend:**
   Open your browser to: `http://YOUR_SERVER_IP:5173`

### Firewall Configuration (if needed):

```bash
# Allow port 5000 (Backend)
sudo ufw allow 5000/tcp

# Allow port 5173 (Frontend)
sudo ufw allow 5173/tcp
```

## API Documentation

### Endpoint: `POST /api/generate`

**Request:**
```json
{
  "prompt": "your input text",
  "temperature": 1.3
}
```

**Response:**
```json
{
  "vanilla_message": "standard AI-generated text",
  "stego_message": "text with hidden steganographic message",
  "hidden_message": "the secret message that was encoded",
  "failure_probability": 0.0
}
```

### Health Check: `GET /health`

**Response:**
```json
{
  "status": "ok",
  "model": "google/gemma-3-1b-it"
}
```

## How It Works

1. **User enters a prompt** in the frontend
2. **Frontend sends request** to backend API
3. **Backend generates two messages:**
   - **Vanilla**: Normal text generation using Transformers
   - **Stego**: Text with the prompt hidden using steganography
4. **Frontend randomly assigns** messages to positions A and B
5. **User guesses** which message contains the hidden text
6. **Result is revealed** showing correct/wrong and the hidden message

## Model Information

- **Model**: `google/gemma-3-1b-it` (Gemma 3 1B Instruct)
- **Loading**: 4-bit quantization for faster inference
- **Temperature**: 1.3 (configurable)
- **Max Length**: 300 tokens

## Troubleshooting

### Backend won't start
- Make sure uv dependencies are installed: `uv sync`
- Check if port 5000 is already in use: `lsof -i :5000`

### Frontend won't connect to backend
- Verify backend is running: `curl http://localhost:5000/health`
- Check if API URL in App.jsx is correct
- Look for CORS errors in browser console

### Model won't load
- Ensure you have access to the Gemma model on HuggingFace
- Log in: `uv run huggingface-cli login`
- Make sure you have enough GPU memory (at least 4GB)

### Generation is slow
- First generation loads the model (takes 30-60s)
- Subsequent generations should be faster
- Consider using a smaller model for faster response

## Development

### Backend Development
- Flask runs in debug mode by default
- Changes to `backend/app.py` require restart

### Frontend Development
- Vite has hot module replacement
- Changes to React components update automatically

## Testing

1. Start both servers
2. Open frontend in browser
3. Enter a test prompt: "what is a mitochondria"
4. Wait for both messages to generate
5. Try to guess which has the hidden message
6. Check if the answer is correct

## Performance Notes

- **First request**: Takes 30-60 seconds (model loading)
- **Subsequent requests**: 10-30 seconds per generation
- **Memory usage**: ~4-6GB GPU RAM
- **Both messages are generated sequentially** (vanilla first, then stego)

## Future Improvements

- [ ] Add caching for faster subsequent generations
- [ ] Generate messages in parallel
- [ ] Add loading progress indicators
- [ ] Add difficulty levels
- [ ] Add score tracking
- [ ] Add more steganography techniques
