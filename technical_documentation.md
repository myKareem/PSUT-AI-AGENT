# PSUT Voice Agent Technical Documentation

## 1. Overview
The PSUT Voice Agent is a real-time, interactive conversational AI pipeline specifically designed for Princess Sumaya University for Technology (PSUT). It supports Jordanian Arabic speech, utilizing a local setup integrating Speech-to-Text (STT), Large Language Models (LLM) with RAG, and Text-to-Speech (TTS).

The system features two main entry points:
- **`server.py`**: A WebSocket-based FastAPI server to handle remote front-end clients, offering full-duplex communication with real-time transcription and generation.
- **`main.py`**: A local desktop execution loop utilizing PyAudio/SoundDevice and WebRTC Voice Activity Detection (VAD) for standalone real-time interaction.

## 2. System Architecture

The pipeline consists of four major sequential steps:
1. **Audio Capture & VAD**: Detecting human speech thresholds and transmitting valid audio sequences.
2. **Speech-to-Text (STT)**: Converting audio chunks into text.
3. **Dialogue & Retrieval (LLM + RAG)**: Generating responses based on university knowledge bases.
4. **Text-to-Speech (TTS)**: Synthesizing spoken Arabic audio back to the user.

---

### 2.1 Audio Capture & STT Module (`stt_engine.py`)
- **Engine**: `faster-whisper`
- **Model**: `HebArabNlpProject/WhisperLevantine` (optimized for Levantine/Jordanian Arabic).
- **Execution**: Runs on `float16` precision via CUDA for low-latency inference.
- **Mechanics**: Accepts 16kHz float32 NumPy audio buffers and processes transcription synchronously. Implements internal VAD parameters to filter out silence and non-speech noise.

### 2.2 Orchestration & RAG Chatbot (`chatbot.py`)
This is the core "brain" of the agent, handling query routing, context retrieval, and response generation via `LangChain` and `Ollama`.

#### A. Embedding & Vector/Graph Databases
- **Embedding Model**: `Omartificial-Intelligence-Space/Arabic-Triplet-Matryoshka-V2` runs locally via `SentenceTransformers`.
- **Databases**:
  - **In-Memory CAG (Context-Augmented Generation)**: High-priority, fast cosine similarity checks against `general_faq.md`. If a high confidence hit occurs, RAG is bypassed.
  - **Qdrant (Vector DB)**: Used for robust semantic retrieval over document collections (`student_guide`, `major_overview`).
  - **NetworkX Graph DB**: Custom semantic graph index for the `staff_directory`. Embeds node descriptions and relationships, utilizing an overlap boost algorithm to heavily weigh exact Arabic name matches.

#### B. Intent Router Model
- A lightweight router model (`qwen2.5:1.5b`) evaluates the last 3 conversational turns.
- Dynamically classifies user intent into categories: `major_overview`, `student_guide`, `staff_directory`, or `chitchat`.
- Short-circuits queries that are purely chitchat directly to the LLM to save retrieval latency.

#### C. LLM Generation
- **Main Model**: A fine-tuned `Qwen2.5-7B` model running locally via Ollama (`jordanian-uni-bot`).
- **Prompt Engineering**: System instructions restrict output to plain Arabic sentences, forbid markdown and emojis (which break TTS), force number-to-word conversions, and mandate high factual grounding (STRICT GROUNDING).
- **TTS Cleaner**: A custom Regex pipeline extracts, protects, and restores phone numbers and emails while completely stripping non-speakable characters to prevent acoustic model crashes.

### 2.3 Text-to-Speech Module (`sada_tts.py`)
- **Engine**: Wrapper around `tts-arabic-pytorch-master` (SadaTTS).
- **Inference**: Accepts cleaned Arabic strings and generates `.wav` audio files.
- **Pre-Processing**: Implements an aggressive regex filter (`_ARABIC_ONLY`) restricting input strictly to Arabic blocks and whitespace, avoiding inference failures on latin characters or punctuation.

---

## 3. Fine-Tuning Process (`finetune.py`)

To achieve the localized Jordanian dialect and high adherence to PSUT contexts, the foundational LLM (`Qwen2.5-7B-Instruct`) was fine-tuned using `Unsloth` and `TRL` (Transformer Reinforcement Learning).

### 3.1 Setup & Configuration
- **Base Model**: `unsloth/Qwen2.5-7B-Instruct`
- **Precision**: 4-bit quantization during training for VRAM efficiency on consumer hardware.
- **Max Sequence Length**: 2048 tokens.

### 3.2 LoRA (Low-Rank Adaptation)
Parameter-Efficient Fine-Tuning was applied to specific target modules (`q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`).
- **Rank (r)**: 16 (Kept relatively low for stability on a smaller custom dataset).
- **Alpha**: 32
- **Dropout**: 0.05 (To mitigate overfitting).

### 3.3 Dataset Formulation
The model was fine-tuned on a custom JSON dataset (`data.json`) structured into pairs of `instruction`, `context`, and `response`.
- **System Prompt**: Enforced a polite, Jordanian dialect ("بقدر، عشان، اه، تمام، ولا يهمك") and instructed the model to apologize when context is insufficient.
- **Formatting**: Prompts were combined using a Chat Template that explicitly demarcates user queries and `[المعلومات المتاحة]` (Available Information/Context).

### 3.4 Training Parameters
- **Batch Size**: 2 (per device) with 4 gradient accumulation steps.
- **Learning Rate**: $5 \times 10^{-5}$ (Lower learning rate utilized for stability).
- **Scheduler**: Cosine learning rate scheduler.
- **Optimizer**: `adamw_8bit`
- **Epochs**: 1
- **Weight Decay**: 0.01

### 3.5 Export & Deployment
After successful training via `SFTTrainer`, the model weights were saved and natively exported into **GGUF** format utilizing `q4_k_m` quantization (`qwen2.5-7b-instruct.Q4_K_M-001.gguf`). 
This `.gguf` file was subsequently imported into `Ollama` via the custom `Modelfile` blueprint to serve as the live inference backend.

## 4. Execution Pipelines

### Local Execution (`main.py`)
1. Instantiates the microphone stream.
2. `WebRTCVAD` identifies speech. Once 600ms of silence follows speech, the frame queue is batched.
3. Audio buffer processed by `transcribe_audio`.
4. Output text sent to `generate_response()`, yielding cleaned sentences.
5. Sentences concurrently generated into WAV via `SadaTTS` and played aloud using PyGame.

### Server Execution (`server.py`)
1. Starts FastAPI on port `8000`.
2. Serves a pre-generated greeting via WebSocket connection.
3. Accumulates incoming binary audio from the front-end.
4. Skips buffers $<100$ms. Transcribes valid audio.
5. Dispatches sentence-by-sentence text and binary audio blobs back through the WebSocket for low-latency playback.
