# Tomato API Server - Docker Setup

This guide explains how to run the Tomato encoder API server using Docker.

## Prerequisites

- Docker installed
- Docker Compose installed
- NVIDIA Docker runtime (for GPU support)

## Quick Start

### Using Docker Compose (Recommended)

```bash
# Build and start the server
docker-compose -f docker-compose.server.yml up --build

# Run in detached mode
docker-compose -f docker-compose.server.yml up -d

# Stop the server
docker-compose -f docker-compose.server.yml down
```

### Using Docker directly

```bash
# Build the image
docker build -f Dockerfile.server -t tomato-api .

# Run the container
docker run --gpus all -p 8000:8000 -v $(pwd)/logs:/app/logs tomato-api
```

## API Endpoints

Once running, the server will be available at `http://localhost:8000`

### Health Check
```bash
curl http://localhost:8000/health
```

### Encode Message
```bash
curl -X POST http://localhost:8000/encode \
  -H "Content-Type: application/json" \
  -d '{
    "plaintext": "hello world",
    "prompt": "what is the capital of france",
    "cipher_len": 15,
    "max_len": 100
  }'
```

## Configuration

- **Port**: 8000 (can be changed in docker-compose.server.yml)
- **Logs**: Stored in `./logs` directory (mounted as volume)
- **Model**: google/gemma-3-1b-it (default)
- **Temperature**: 1.3 (default)

## API Documentation

When the server is running, visit:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## Request Format

```json
{
  "plaintext": "secret message to encode",
  "prompt": "cover story prompt for natural text generation",
  "cipher_len": 20,
  "max_len": 150
}
```

## Response Format

```json
{
  "stegotext": [5, 22, 8, 16, 9, 21, ...],
  "formatted_stegotext": "The capital of France is Paris. ..."
}
```

## Notes

- The server processes one request at a time to prevent RAM exhaustion
- Returns HTTP 503 if another request is being processed
- First request will take longer as the model loads
- GPU is required for acceptable performance
