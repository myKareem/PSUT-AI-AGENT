# 🎙️ PSUT-AI-AGENT: S.A.D.A. TTS Module Setup

This directory contains the finalized, lightweight FastPitch TTS module for the S.A.D.A. system (trained on Jordanian dialects). Follow these exact steps to run it locally so your RAG and LLM components can communicate with it.

---

## 1. Get the Code & Set Up Environment
First, make sure you are on the correct branch and create a clean virtual environment to avoid dependency conflicts with the RAG system.

```bash
# Pull the latest clean code
git fetch
git checkout MuradUpdated

# Create and activate a virtual environment (Windows)
python -m venv sada_tts_env
sada_tts_env\Scripts\activate
2. Download the Acoustic Weights
Because the model brain is too large for GitHub, you need to download the trained checkpoint manually.

Download the FastPitch Checkpoint Here

Placement:
Once downloaded, place the .pth file directly into this exact folder path:

checkpoints/exp_fp_adv/states_800000.pth

3. Install Dependencies
TTS models are hardware-dependent, so we must install PyTorch separately from the standard requirements.

A. Install PyTorch
If you have an NVIDIA GPU on Windows, run this command to enable instant hardware acceleration:

Bash
pip install torch torchaudio --index-url [https://download.pytorch.org/whl/cu118](https://download.pytorch.org/whl/cu118)
Note: If you do not have an NVIDIA GPU, this command might fail. If so, get your specific installation command from the PyTorch website.

B. Install Core API Packages
Once PyTorch is installed, install the remaining lightweight server requirements:

Bash
pip install -r requirements.txt
4. Booting the API Engine
Do not run inference.py directly in the terminal. Instead, boot up the FastAPI server. This keeps the model loaded in your memory so the RAG system can instantly synthesize speech without a loading delay.

Bash
uvicorn app:app --reload
5. Integration (For RAG/LLM)
Once the server is running, it will actively listen on http://127.0.0.1:8000.
