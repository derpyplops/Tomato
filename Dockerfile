FROM nvidia/cuda:11.8.0-devel-ubuntu22.04

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies including Node.js
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    git \
    wget \
    curl \
    ca-certificates \
    gnupg \
    unzip \
    && rm -rf /var/lib/apt/lists/*

# Install Node.js 18.x
RUN curl -fsSL https://deb.nodesource.com/setup_18.x | bash - \
    && apt-get install -y nodejs \
    && rm -rf /var/lib/apt/lists/*

# Install pnpm
RUN npm install -g pnpm

# Install ngrok
RUN wget https://bin.equinox.io/c/bNyj1mQVY4c/ngrok-v3-stable-linux-amd64.tgz \
    && tar -xvzf ngrok-v3-stable-linux-amd64.tgz -C /usr/local/bin \
    && rm ngrok-v3-stable-linux-amd64.tgz

# Install uv (fast Python package installer)
RUN curl -LsSf https://astral.sh/uv/install.sh | sh \
    && ln -s /root/.cargo/bin/uv /usr/local/bin/uv

# Set working directory
WORKDIR /app

# Copy the local Tomato repository
COPY . /app

# Upgrade pip
RUN pip3 install --upgrade pip

# Install mec dependency first
RUN pip3 install --no-cache-dir git+https://github.com/user1342/mec

# Install PyTorch with CUDA 11.8 support
RUN pip3 install --no-cache-dir \
    torch==2.0.1+cu118 \
    torchvision==0.15.2+cu118 \
    torchaudio==2.0.2+cu118 \
    --index-url https://download.pytorch.org/whl/cu118

# Install other dependencies from requirements.txt
RUN pip3 install --no-cache-dir \
    tqdm \
    setuptools \
    huggingface_hub \
    numpy \
    accelerate \
    pyyaml \
    "bitsandbytes>=0.39.0" \
    "transformers[torch]>=4.28.1" \
    flask \
    flask-cors

# Install Tomato package
RUN pip3 install --no-cache-dir -e .

# Install frontend dependencies
WORKDIR /app/stego-game
RUN pnpm install

# Return to app directory
WORKDIR /app

# Expose ports
EXPOSE 5000 5173

# Set entrypoint
ENTRYPOINT ["python3"]
CMD ["--help"]