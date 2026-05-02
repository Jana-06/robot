FROM python:3.11-slim

# System libs required by opencv + espeak TTS
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libgl1-mesa-glx \
    libsm6 \
    libxext6 \
    libxrender-dev \
    espeak \
    espeak-data \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first (layer-cached unless requirements.txt changes)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the project
COPY . .

# Render injects $PORT at runtime; 8765 is the local default
EXPOSE 8765

CMD ["python", "maini.py"]
