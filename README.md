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

## **2. Download the Acoustic Weights**

The model weights are too large for GitHub, so you must download them manually.

**Download the FastPitch Checkpoint** https://drive.google.com/drive/folders/1W_ZoTUwgvVNmpWYyTyY05aIm3gcGzhKy?usp=drive_link

After downloading, place the `.pth` file in the following directory:
checkpoints/exp_fp_adv/ --> checkpoints/exp_fp_adv/states_800000.pth

## **3. Install Dependencies**

TTS models depend on hardware acceleration, so PyTorch must be installed separately.

### A. Install PyTorch

If you have an **NVIDIA GPU (Windows)**, use:

```bash
pip install torch torchaudio --index-url [https://download.pytorch.org/whl/cu118](https://download.pytorch.org/whl/cu118)

NOTE: If you do not have an NVIDIA GPU, this command may fail.
Visit the official PyTorch website to get the correct installation command for your system.

### B. Install Core API Packages

Once PyTorch is installed, install the remaining dependencies:

```bash
pip install -r requirements.txt

## **4. Booting the API Engine**
Do not run inference.py directly in the terminal. Instead, boot up the FastAPI server. This keeps the model loaded in your memory so the RAG system can instantly synthesize speech without a loading delay.

```bash
uvicorn app:app --reload


✅ You're Ready
Once the server is running, your TTS pipeline is ready to accept requests.
