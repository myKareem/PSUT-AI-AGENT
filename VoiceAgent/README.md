# PSUT Voice Agent - Deployment Guide

This directory contains everything needed to run the PSUT Voice Agent on any machine (specifically targeting machines with GPUs like the RTX 4080).

## Option 1: Native Windows Setup (Recommended for Desktop use with `main.py`)

1. **Install Python 3.10+**.
2. **Install Ollama**: Download and install from [ollama.com](https://ollama.com).
3. **Install Dependencies**:
   Open a terminal in this directory and run:
   ```bash
   pip install -r requirements.txt
   ```
4. **Handling the Fine-Tuned Model (`qwen2.5-7b-instruct.Q4_K_M-001.gguf`)**:
   Because the GGUF model is very large (~4.6GB), it is not included in GitHub. You have two options to transfer it:
   
   **A. Manual Copy (USB/Network Drive)**
   Copy the `qwen2.5-7b-instruct.Q4_K_M-001.gguf` file manually to this folder.
   
   **B. Google Drive / HuggingFace**
   - Upload your `qwen2.5-7b-instruct.Q4_K_M-001.gguf` to a Hugging Face model repository (Free and optimized for large files).
   - Alternatively, upload it to Google Drive and use `gdown` to download it to your new PC.
   - Example to download from HF: `pip install huggingface_hub` then run:
     `huggingface-cli download your-username/your-repo-name qwen2.5-7b-instruct.Q4_K_M-001.gguf --local-dir .`

5. **Load the Model into Ollama**:
   Once the `.gguf` file is in this folder, open a terminal and run:
   ```bash
   ollama create jordanian-uni-bot -f Modelfile
   ```
6. **Run the Project**:
   - To run the web version: `python server.py` (then open `http://localhost:8000` in your browser).
   - To run the desktop version: `python main.py` (requires a microphone connected).

---

## Option 2: Docker Compose (Recommended for isolated Server deployment)

We have provided a `docker-compose.yml` to run both Ollama and the Voice Agent API Server smoothly.

### Prerequisites
- Install [Docker Desktop](https://www.docker.com/products/docker-desktop/).
- Ensure the NVIDIA Container Toolkit is installed so Docker can access your RTX 4080 GPU.

### Setup Steps
1. Place your `qwen2.5-7b-instruct.Q4_K_M-001.gguf` file into this `VoiceAgent` folder. (See Option 1 on how to transfer it).
2. Open a terminal in this folder and run:
   ```bash
   docker-compose build
   docker-compose up -d
   ```
3. The first time you run this, you need to tell the Ollama container to create the model using the `Modelfile`. Run:
   ```bash
   docker exec -it ollama-service ollama create jordanian-uni-bot -f /Modelfile
   ```
4. Open your browser to `http://localhost:8000` to interact with the Voice Agent.

> **Note**: Whisper and SentenceTransformers models will automatically download on the first run, so the first request might take a bit longer.
