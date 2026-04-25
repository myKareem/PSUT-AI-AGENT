# 🎤 PSUT-AI-AGENT: S.A.D.A. TTS Module Setup

This directory contains the finalized, lightweight FastPitch TTS module for the S.A.D.A. system (trained on Jordanian dialects). Follow these exact steps to run it locally so your RAG and LLM components can communicate with it.

---

## **1. Get the Code & Set Up Environment**

Make sure you're on the correct branch and using a clean virtual environment.

### Pull the latest code
```bash
git fetch
git checkout MuradUpdated
```
### Create and activate a virtual environment (Windows)
```bash
python -m venv sada_tts_env
sada_tts_env\Scripts\activate
```

## **2. Download the Acoustic Weights**

Because the model is too large for GitHub, you need to download the trained checkpoint manually.

Download the FastPitch Checkpoint Here: https://drive.google.com/drive/folders/1W_ZoTUwgvVNmpWYyTyY05aIm3gcGzhKy?usp=drive_link


Once downloaded, place the .pth file into this exact path:

```bash
checkpoints/exp_fp_adv/ --> checkpoints/exp_fp_adv/states_800000.pth
```

## **3. Install Dependencies**

TTS models are hardware-dependent, so PyTorch must be installed separately.

### Install PyTorch (NVIDIA GPU - Windows)
```bash
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118
```
⚠️ If you do NOT have an NVIDIA GPU, this may fail.
Get your correct installation command from the official PyTorch website.

### Install Core API Packages
```bash
pip install -r requirements.txt
```

## **4. Booting the API Engine**

Do NOT run inference.py directly.

Start the FastAPI server instead:
```bash
uvicorn app:app --reload
```
This keeps the model loaded in memory so your system can generate speech instantly.

## **5. Integration (For RAG / LLM)**

Once the server is running, it will listen on:
```bash
http://127.0.0.1:8000
```
The RAG or LLM system can now send requests for speech synthesis.
