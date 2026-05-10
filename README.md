PSUT Voice Agent - Deployment Guide

This directory contains everything needed to run the PSUT Voice Agent on any machine (specifically targeting machines with GPUs like the RTX 4080).
Option 1: Native Windows Setup (Recommended for Desktop use with main.py)

    Install Python 3.10+.

    Install Ollama: Download and install from ollama.com.

    Install Dependencies: Open a terminal in this directory and run:

    pip install -r requirements.txt

    Handling the Fine-Tuned Model (qwen2.5-7b-instruct.Q4_K_M-001.gguf): Because the GGUF model is very large (~4.6GB), it is not included in GitHub. You have two options to transfer it:

    A. Manual Copy (USB/Network Drive) Copy the qwen2.5-7b-instruct.Q4_K_M-001.gguf file manually to this folder.

    B. Google Drive / HuggingFace
        Upload your qwen2.5-7b-instruct.Q4_K_M-001.gguf to a Hugging Face model repository (Free and optimized for large files).
        Alternatively, upload it to Google Drive and use gdown to download it to your new PC.
        Example to download from HF: pip install huggingface_hub then run: huggingface-cli download your-username/your-repo-name qwen2.5-7b-instruct.Q4_K_M-001.gguf --local-dir .

    Load the Model into Ollama: Once the .gguf file is in this folder, open a terminal and run:

    ollama create jordanian-uni-bot -f Modelfile

    Run the Project:
        To run the web version: python server.py (then open http://localhost:8000 in your browser).
        To run the desktop version: python main.py (requires a microphone connected).
