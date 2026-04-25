# PSUT-AI-AGENT
## 🎙️ S.A.D.A. TTS Module Setup

To run this TTS module locally alongside the RAG and LLM components, you need to download the trained acoustic model weights.

**1. Download the Weights:**
* [Download the FastPitch Checkpoint Here](INSERT_YOUR_GOOGLE_DRIVE_LINK_HERE)

**2. Placement:**
Once downloaded, place the `.pth` file directly into this exact folder path:
`checkpoints/exp_fp_adv/states_800000.pth`

**3. Running the API:**
First, install the GPU-accelerated version of PyTorch by running:
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118
(Note: If you do not have an NVIDIA GPU, this command might fail. If so, get your specific installation command from the PyTorch website.)

Once PyTorch is installed, run: pip install -r requirements.txt

Do not run `inference.py` directly. Instead, boot up the FastAPI server by running:
`uvicorn app:app --reload`
This will keep the TTS model loaded in your GPU/RAM so the RAG system can instantly synthesize speech!
