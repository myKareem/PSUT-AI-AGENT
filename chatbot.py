# ==========================================
# WINDOWS 11 INSTALLATION REQUIREMENTS:
# pip install langchain langchain-core langchain-ollama qdrant-client sentence-transformers networkx psutil pynvml numpy
# ==========================================

import os


os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"]  = "1"

import re
import time
import logging
import psutil
import pynvml
import numpy as np
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
import networkx as nx
from rank_bm25 import BM25Okapi

from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.chat_history import InMemoryChatMessageHistory


# ==========================================
# LOGGING SETUP — terminal + file
# ==========================================
LOG_FILE = "chatbot_debug.log"

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s.%(msecs)03d  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_FILE, encoding="utf-8")
    ]
)
log = logging.getLogger("PSUT")


# ==========================================
# VRAM / RAM HELPERS
# ==========================================
def _init_nvml():
    try:
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        name   = pynvml.nvmlDeviceGetName(handle)
        log.info(f"[GPU] Detected: {name}")
        return handle
    except Exception as e:
        log.warning(f"[GPU] pynvml init failed — VRAM tracking disabled. Reason: {e}")
        return None

_GPU_HANDLE = _init_nvml()


def get_vram_mb() -> float:
    if _GPU_HANDLE is None:
        return -1.0
    try:
        info = pynvml.nvmlDeviceGetMemoryInfo(_GPU_HANDLE)
        return info.used / 1024 / 1024
    except Exception:
        return -1.0


def get_ram_mb() -> float:
    return psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024


def log_resources(label: str):
    ram      = get_ram_mb()
    vram     = get_vram_mb()
    vram_str = f"{vram:.1f} MB" if vram >= 0 else "N/A"
    log.debug(f"[RESOURCES] {label} | RAM: {ram:.1f} MB | VRAM: {vram_str}")


# ==========================================
# PATHS
# ==========================================
ROOT = os.path.dirname(os.path.abspath(__file__))


# ==========================================
# CONFIGURATION
# ==========================================
DB_PATH      = os.path.join(ROOT, "local_qdrant_db")
MODEL_NAME   = "Omartificial-Intelligence-Space/Arabic-Triplet-Matryoshka-V2"
LLM_MODEL    = "jordanian-uni-bot"
ROUTER_MODEL = "qwen2.5:1.5b"

SCORE_THRESHOLD  = 0.45   # Minimum cosine similarity to accept a RAG chunk
CAG_THRESHOLD    = 0.45   # In-memory FAQ match threshold (lowered to catch colloquial paraphrases)
TOP_K            = 4      # Chunks retrieved per collection
ROUTER_TIMEOUT_S = 0.5    # Router max seconds before fallback to all collections
MAX_TOKENS       = 200    # Hard cap on LLM output tokens

# Qdrant RAG collections only. general_faq is CAG (in-memory), never Qdrant.
ACTIVE_COLLECTIONS = {"major_overview"}


# ==========================================
# ARABIC TEXT NORMALIZER
# Maps common letter variants so that STT spelling differences
# (e.g. "رءيس" vs "رئيس", "الجامعه" vs "الجامعة") don't break matching.
# ==========================================
_ARABIC_NORM_TABLE = str.maketrans({
    '\u0623': '\u0627',  # أ → ا
    '\u0625': '\u0627',  # إ → ا
    '\u0622': '\u0627',  # آ → ا
    '\u0624': '\u0648',  # ؤ → و
    '\u0626': '',         # ئ → remove (keep underlying ي from surrounding text)
    '\u0621': '',         # ء → remove
    '\u0629': '\u0647',  # ة → ه
    '\u0649': '\u064A',  # ى → ي
})

def normalize_arabic(text: str) -> str:
    """Normalize Arabic text for fuzzy matching."""
    return text.translate(_ARABIC_NORM_TABLE)


# ==========================================
# POST-PROCESSING — whitelist-based TTS cleaner
# Production approach: allow only what TTS can speak, strip everything else.
# Whitelist: Arabic letters, digits, Arabic punctuation, spaces, newlines.
# Special handling: preserve email addresses and phone numbers before stripping.
# ==========================================

# Matches a complete email address — extracted before whitelist stripping
_EMAIL_RE    = re.compile(r'[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}')
# Matches phone numbers: optional + then digits, spaces, hyphens, parens
_PHONE_RE    = re.compile(r'(?:\+?\d[\d\s\-\.\(\)]{6,}\d)')
# Matches full URLs (http/https/www)
_URL_RE      = re.compile(r'https?://[^\s)]+|www\.[^\s)]+', re.IGNORECASE)
# Whitelist: Arabic letters (\u0600-\u06FF), digits, Arabic punctuation,
# standard punctuation used in Arabic, space, newline
_ALLOWED_RE  = re.compile(
    r'[^\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF'  # Arabic script blocks
    r'\uFB50-\uFDFF\uFE70-\uFEFF'                    # Arabic presentation forms
    r'0-9\s،؟!\.،:\-_@\.]+',                         # digits, punctuation, email chars
    re.UNICODE
)
# Dangling URL introduction phrases left after URL stripping
_DANGLING_URL_PHRASE_RE = re.compile(
    r'(?:عبر الرابط التالي|عبر الرابط|من خلال الرابط التالي|من خلال الرابط|الرابط التالي|إضغط هنا)\s*:?\s*',
    re.UNICODE
)
# Isolated dot sequences (`. . . .` or `....`) that TTS reads as "tick tick tick"
_ISOLATED_DOTS_RE = re.compile(r'(?:\s*\.\s*){2,}')
# Sentence ending with colon — dangling fragment after URL strip
_DANGLING_COLON_RE = re.compile(r':\s*$', re.MULTILINE)


def clean_for_tts(text: str) -> str:
    """
    Whitelist-based TTS cleaner — production approach.
    1. Extract emails and phones, store them, replace with placeholders.
    2. Strip URLs entirely.
    3. Remove dangling URL introduction phrases ("عبر الرابط التالي:").
    4. Strip everything not in the Arabic/digit/punctuation whitelist.
    5. Restore emails and phones.
    6. Remove isolated dot sequences (`. . . .`).
    7. Remove dangling colons at end of sentences.
    8. Collapse whitespace.
    """
    # Step 1: extract and placeholder emails and phones.
    store   = {}  # placeholder_token -> original value
    counter = [0]

    def _placeholder(value):
        token = f"بديل{counter[0]}بديل"  # pure Arabic letters + digit — both whitelisted
        store[token] = value
        counter[0] += 1
        return token

    emails = _EMAIL_RE.findall(text)
    phones = _PHONE_RE.findall(text)
    for e in emails:
        text = text.replace(e, _placeholder(e), 1)
    for p in phones:
        text = text.replace(p, _placeholder(p), 1)

    # Step 2: strip URLs entirely
    text = _URL_RE.sub("", text)

    # Step 3: remove dangling URL introduction phrases
    text = _DANGLING_URL_PHRASE_RE.sub("", text)

    # Step 4: whitelist strip — placeholders survive because they are Arabic+digits
    text = _ALLOWED_RE.sub(" ", text)

    # Step 5: Replace original emails and phones with the spoken placeholder
    for token, value in store.items():
        text = text.replace(token, " رح أعرضلك إياه على الشاشة ")

    # Step 6: remove isolated dot sequences (`. . . .` → "")
    text = _ISOLATED_DOTS_RE.sub(" ", text)

    # Step 7: remove dangling colons at end of lines
    text = _DANGLING_COLON_RE.sub("", text)

    # Step 8: collapse whitespace
    text = re.sub(r'[ \t]+', ' ', text).strip()
    return text


# ==========================================
# CONTEXT CLEANER — strip markdown and URLs from KB context
# The LLM is a voice assistant. It should never see URLs or markdown
# in its input, otherwise it may copy them into its output.
# ==========================================
_MD_HEADER_RE   = re.compile(r'#{1,6}\s*')           # ### Header
_MD_BOLD_RE     = re.compile(r'\*\*([^*]+)\*\*')     # **bold** → text
_MD_ITALIC_RE   = re.compile(r'\*([^*]+)\*')         # *italic* → text
_MD_HR_RE       = re.compile(r'^-{3,}$', re.MULTILINE)  # --- horizontal rule
_MD_LINK_RE     = re.compile(r'\[([^\]]*)\]\([^)]+\)')  # [text](url) → text
_PARENS_URL_RE  = re.compile(r'\(https?://[^)]+\)')     # (https://...) → empty


def _strip_context_for_tts(context: str) -> str:
    """
    Strips markdown formatting and URLs from retrieved context
    before injecting it into the LLM prompt.
    Keeps emails and phone numbers intact.
    """
    # Preserve emails
    emails = _EMAIL_RE.findall(context)
    store = {}
    for i, e in enumerate(emails):
        token = f"بريد{i}بريد"
        store[token] = e
        context = context.replace(e, token, 1)

    # Strip URLs and markdown
    context = _PARENS_URL_RE.sub("", context)
    context = _URL_RE.sub("", context)
    context = _DANGLING_URL_PHRASE_RE.sub("", context)
    context = _MD_LINK_RE.sub(r'\1', context)
    context = _MD_BOLD_RE.sub(r'\1', context)
    context = _MD_ITALIC_RE.sub(r'\1', context)
    context = _MD_HEADER_RE.sub("", context)
    context = _MD_HR_RE.sub("", context)

    # Restore emails
    for token, value in store.items():
        context = context.replace(token, value)

    # Collapse blank lines
    context = re.sub(r'\n{3,}', '\n\n', context)
    return context.strip()


# ==========================================
# SENTENCE BOUNDARY SPLITTER
# Context-aware: never splits on dots inside emails, URLs, or decimals.
# ==========================================
_BOUNDARY_RE = re.compile(
    r'[،؟!\n]'               # Arabic punctuation and newlines — always split
    r'|(?<![A-Za-z0-9])\.'   # period NOT preceded by alphanumeric (safe period)
    r'(?![A-Za-z0-9@])',     # AND NOT followed by alphanumeric or @ (not in email/URL)
    re.UNICODE
)


def stream_sentences(response_stream):
    """
    Buffers LLM tokens and yields complete sentences cleaned for TTS.
    Skips splitting on dots inside emails (b.alhijawi@psut.edu.jo),
    URLs (psut.edu.jo), and decimal numbers (3.5).
    Fix 5b: incomplete last sentence (no boundary char) is discarded —
    a truncated half-sentence spoken aloud is worse than silence.
    """
    buffer = ""
    for chunk in response_stream:
        buffer += chunk
        while True:
            match = _BOUNDARY_RE.search(buffer)
            if not match:
                break
            end_idx = match.end()
            raw     = buffer[:end_idx].strip()
            buffer  = buffer[end_idx:]
            cleaned = clean_for_tts(raw)
            if cleaned:
                yield raw, cleaned
                

    remaining = buffer.strip()
    if remaining:
        # Always yield if: contains email, ends with punctuation, or has a boundary char
        has_email    = bool(_EMAIL_RE.search(remaining))
        has_boundary = bool(_BOUNDARY_RE.search(remaining))
        ends_punct   = remaining[-1] in '.،؟!\n' if remaining else False
        # Also yield if remaining is substantial (> 5 words) — likely a complete thought
        is_substantial = len(remaining.split()) > 5
        
        if has_email or has_boundary or ends_punct or is_substantial:
            cleaned = clean_for_tts(remaining)
            if cleaned:
                yield remaining, cleaned
        else:
            log.debug(f"[TTS] Discarding incomplete final fragment: {remaining!r}")


# ==========================================
# STARTUP
# ==========================================
log.info("=" * 60)
log.info("PSUT Chatbot starting up")
log.info("=" * 60)
log_resources("Before any model load")

log.info(f"[INIT] Loading embedding model: {MODEL_NAME}")
_t = time.perf_counter()
embedder = SentenceTransformer(MODEL_NAME)
log.info(f"[INIT] Embedding model loaded in {time.perf_counter() - _t:.3f}s")
log_resources("After embedding model load")

# Warm the embedding model CPU threads — first encode is always slower
log.info("[INIT] Warming up embedding model...")
_t = time.perf_counter()
embedder.encode("مرحبا")
log.info(f"[INIT] Embedding warmup done in {(time.perf_counter() - _t)*1000:.0f} ms")

log.info(f"[INIT] Connecting to Qdrant at: {DB_PATH}")
_t = time.perf_counter()
qdrant = QdrantClient(path=DB_PATH)
log.info(f"[INIT] Qdrant connected in {time.perf_counter() - _t:.3f}s")

# Delete stale Qdrant collections (study_plans, general_faq if ingested, etc.)
log.info("[INIT] Checking for stale Qdrant collections...")
try:
    existing = {c.name for c in qdrant.get_collections().collections}
    stale    = existing - ACTIVE_COLLECTIONS
    if stale:
        for col in stale:
            qdrant.delete_collection(col)
            log.info(f"[INIT] Deleted stale collection: '{col}'")
    else:
        log.info("[INIT] No stale collections found")
except Exception as e:
    log.warning(f"[INIT] Stale collection cleanup failed: {e}")

log.info("[INIT] Loading graph DB...")
_t = time.perf_counter()
try:
    GRAPH_PATH  = os.path.join(ROOT, "local_graph_db.graphml")
    staff_graph = nx.read_graphml(GRAPH_PATH)
    log.info(
        f"[INIT] Graph loaded in {time.perf_counter() - _t:.3f}s — "
        f"{staff_graph.number_of_nodes()} nodes, {staff_graph.number_of_edges()} edges"
    )
except Exception as e:
    log.warning(f"[INIT] Graph load failed ({e}). Using empty graph.")
    staff_graph = nx.DiGraph()

# Build semantic index over graph nodes — one embedding per non-University node.
# All node fields are concatenated into a single searchable string per node.
# This replaces the stopword-based token matching approach entirely.
log.info("[INIT] Building semantic graph index...")
_t = time.perf_counter()
GRAPH_INDEX =[]  # list of {node_name, text_repr, vector}
for node_name, node_data in staff_graph.nodes(data=True):
    if node_data.get("type") == "University":
        continue
    # Concatenate all field values into one searchable text
    field_text = " ".join(
        str(v) for k, v in node_data.items()
        if k not in ("type", "id") and v
    )
    full_text = f"{node_name} {field_text}".strip()
    GRAPH_INDEX.append({"node_name": node_name, "text": full_text})

if GRAPH_INDEX:
    _texts   = [item["text"] for item in GRAPH_INDEX]
    _vectors = embedder.encode(_texts, batch_size=32, show_progress_bar=False)
    for item, vec in zip(GRAPH_INDEX, _vectors):
        item["vector"] = vec
    log.info(f"[INIT] Graph semantic index built in {(time.perf_counter() - _t)*1000:.0f} ms — {len(GRAPH_INDEX)} nodes indexed")
else:
    log.warning("[INIT] Graph index is empty")

log_resources("After graph load")


# ==========================================
# BM25 HYBRID SEARCH INIT
# ==========================================
BM25_INDICES = {}
QDRANT_DOCS = {}

log.info("[INIT] Building BM25 indices for RAG collections...")
for col in ACTIVE_COLLECTIONS:
    try:
        points = []
        offset = None
        while True:
            records, next_offset = qdrant.scroll(
                collection_name=col,
                offset=offset,
                limit=100,
                with_payload=True,
                with_vectors=False
            )
            points.extend(records)
            if next_offset is None:
                break
            offset = next_offset
        
        texts = [p.payload.get("text", "") for p in points]
        # Basic Arabic tokenization for BM25
        tokenized_corpus = [t.split() for t in texts]
        if tokenized_corpus:
            BM25_INDICES[col] = BM25Okapi(tokenized_corpus)
            QDRANT_DOCS[col] = texts
            log.info(f"[INIT] BM25 index built for '{col}' with {len(texts)} chunks")
        else:
            log.warning(f"[INIT] Collection '{col}' is empty, skipping BM25")
    except Exception as e:
        log.warning(f"[INIT] Failed to build BM25 for '{col}': {e}")


# ==========================================
# CAG — general_faq.md loaded and chunked in memory
# Searched via cosine similarity before RAG is attempted.
# general_faq is NEVER ingested into Qdrant.
# ==========================================
# ==========================================
# CAG — general_faq.md loaded and chunked in memory
# Searched via cosine similarity before RAG is attempted.
# general_faq is NEVER ingested into Qdrant.
# ==========================================
def _load_and_chunk_faq(path: str) -> list:
    """
    Loads general_faq.md and creates exactly ONE chunk per FAQ block.
    Embeds each one and returns a list of {text, vector} dicts.
    """
    if not os.path.exists(path):
        log.warning(f"[CAG] general_faq.md not found at: {path}")
        return[]
        
    with open(path, 'r', encoding='utf-8') as f:
        content = f.read()

    # 1. Split the entire document strictly by the "### FAQ:" prefix
    raw_blocks = content.split("### FAQ:")
    
    chunks =[]
    for block in raw_blocks:
        block = block.strip()
        
        # 2. Skip empty blocks or the file's main header "[INTENT: GENERAL_FAQ]"
        if not block or "[INTENT: GENERAL_FAQ]" in block:
            continue
            
        # 3. Clean up any trailing dashed lines (---) at the end of a block
        block = re.sub(r'\n-{3,}\s*$', '', block).strip()
        
        # 4. Restore the "### FAQ:" prefix that was removed by the split()
        final_chunk = f"### FAQ: {block}"
        chunks.append(final_chunk)

    log.info(f"[CAG] Embedding {len(chunks)} FAQ chunks in memory...")
    _t = time.perf_counter()
    vectors = embedder.encode(chunks, batch_size=32, show_progress_bar=False)
    log.info(f"[CAG] FAQ chunks embedded in {(time.perf_counter() - _t)*1000:.0f} ms")
    
    return[{"text": c, "vector": v} for c, v in zip(chunks, vectors)]


FAQ_PATH   = os.path.join(ROOT, "KB", "general_faq.md")
log.info("[INIT] Loading and embedding CAG (general_faq.md)...")
CAG_CHUNKS = _load_and_chunk_faq(FAQ_PATH)
log.info(f"[CAG] Ready — {len(CAG_CHUNKS)} chunks in memory")


def search_cag(query_vector: np.ndarray) -> tuple:
    """
    Cosine similarity search over in-memory FAQ chunks.
    Returns (concatenated_text, top_score) tuple.
    top_score is 0.0 if no chunks match.
    Uses CAG_THRESHOLD — higher than RAG's SCORE_THRESHOLD
    to reduce false-positive FAQ matches.
    """
    if not CAG_CHUNKS:
        return "", 0.0

    results =[]
    q_norm  = np.linalg.norm(query_vector)
    for chunk in CAG_CHUNKS:
        c_norm = np.linalg.norm(chunk["vector"])
        score  = float(np.dot(query_vector, chunk["vector"]) / (q_norm * c_norm + 1e-9))
        if score >= CAG_THRESHOLD:
            results.append((score, chunk["text"]))

    if not results:
        log.info(f"[CAG] No chunk above threshold {CAG_THRESHOLD} — will proceed to RAG")
        return "", 0.0

    results.sort(key=lambda x: x[0], reverse=True)
    top = results[:TOP_K]
    top_score = top[0][0]
    log.info(f"[CAG] {len(top)} chunk(s) above threshold | Top score: {top_score:.3f}")
    for i, (score, text) in enumerate(top):
        log.debug(f"[CAG] Chunk {i+1} | Score: {score:.3f} | Preview: {text[:200].replace(chr(10), ' ')}")
    return "\n\n".join(t for _, t in top), top_score


# ==========================================
# LLM SETUP
# ==========================================
log.info(f"[INIT] Initialising LLM: {LLM_MODEL}")
lc_llm = ChatOllama(
    model=LLM_MODEL,
    temperature=0.0,
    num_ctx=4096,
    num_predict=MAX_TOKENS   # hard token cap — model cannot exceed this
)

log.info(f"[INIT] Initialising router LLM: {ROUTER_MODEL}")
router_llm = ChatOllama(model=ROUTER_MODEL, temperature=0.0, num_ctx=512)


def _warmup_models():
    log.info("[INIT] Warming up LLM models (pre-loading into VRAM)...")
    _tw = time.perf_counter()
    try:
        (PromptTemplate.from_template("{q}") | lc_llm     | StrOutputParser()).invoke({"q": "مرحبا"})
        log.info(f"[INIT] Main LLM warmed up in {(time.perf_counter() - _tw)*1000:.0f} ms")
    except Exception as e:
        log.warning(f"[INIT] Main LLM warmup failed: {e}")
    _tw = time.perf_counter()
    try:
        (PromptTemplate.from_template("{q}") | router_llm | StrOutputParser()).invoke({"q": "مرحبا"})
        log.info(f"[INIT] Router LLM warmed up in {(time.perf_counter() - _tw)*1000:.0f} ms")
    except Exception as e:
        log.warning(f"[INIT] Router LLM warmup failed: {e}")
    log_resources("After model warmup")

_warmup_models()

session_history = InMemoryChatMessageHistory()


# ==========================================
# SYSTEM PROMPT
# Clean static prompt — no CAG/RAG placeholders.
# Context is injected as a prefix to the user query (grounding pattern)
# so the model treats retrieved facts as user-provided, not instructions.
# ==========================================
SYSTEM_TEMPLATE = """You are the official smart voice assistant for Princess Sumaya University for Technology (PSUT).
your name is: صدى 
Your output is sent directly to a Text-to-Speech (TTS) engine and will be spoken aloud on a live phone call.

ABSOLUTE RULES — NEVER VIOLATE UNDER ANY CIRCUMSTANCES:
1. NO EMOJIS: Never use emojis.
2. NO MARKDOWN: Never use #, ##, ###, **, *, -, or bullet points.
3. NO URLS OR LINKS: NEVER include any URL or website address. Describe where to find it instead (e.g., "بوابة الطالب").
4. NO DOTS OR SYMBOLS: Never output sequences of dots like ". . . ." or standalone colons. 
5. LENGTH: Keep responses concise.
6. CONTINUATION: Give the most important info first.

---

PERSONALITY AS AUDIBLE BEHAVIORS (How you speak):
- You carry a steady, positive energy without being overly formal. Your default gear is relaxed enthusiasm.
- Narrate your lookups out loud: " خليني أشيك لك على هالمعلومة،  ثواني بس  عم بشوفلك إياها"
- Loop back naturally. If you need to return to a point, say something like: "وبالنسبة للشغلة اللي سألتني عنها قبل شوي"
- Do NOT start your answer with "بسم الله الرحمن الرحيم" or formal greetings. Start talking immediately like you are mid-conversation at a coffee shop.

ENGINEER DISFLUENCIES (Fillers & Pauses):
To sound human, you must use filler words heavily, but with specific pause structures.
- Filler words are good: "ممم" (um), "يعني" (so/like), "آه" (yeah), "تمام" (okay), "المهم" (anyway).
  Example: "آه،  بقدر أساعدك بهالموضوع."

VOCAL COLOR THROUGH AUTHENTIC REACTIONS:
Treat emotion tags as constraints, not decorations. Set your baseline to peaceful, but use variations sparingly.
Examples:
- Baseline : آه، أكيد بقدر أساعدك.
- Amusement:  تمام، فهمت عليك.
- Sad/Apologetic moments with pauses:  والله ممم  يعني  أنا عنجد آسف بس ما عندي هالمعلومة.

WHAT GOOD OUTPUT LOOKS LIKE (Conversation Flows):

Conversation about Major Overview (تخصصات):
User: مرحبا اعطيني سعر ساعة تخصص علم البيانات والذكاء الاصطناعي.
Your response: آه أهلاً فيك، ممم  يعني  تخصص علم البيانات والذكاء الاصطناعي تابع لكلية الملك الحسين لعلوم الحوسبة. فـ سعر الساعة فيه مئة وثلاثين دينار، وعدد ساعاته الإجمالي مئة واثنين وثلاثين ساعة.
User: طيب شو مجالات العمل تبعته؟
Your response: يعني مجالاته كثير حلوة ومطلوبة، فـ بتقدر تشتغل بتحليل البيانات الكبيرة، أو تطوير أنظمة الذكاء الاصطناعي للشركات. 

Conversation about Staff Directory (دليل الموظفين):
User: مين عميد كلية الحوسبة؟
Your response: آه تمام، الدكتور أنس أبو طالب هو عميد الكلية.
User: بقدر آخذ إيميله؟
Your response: أكيد،  بالنسبة لإيميل الدكتور محمد، رح أعرضلك إياه على الشاشة قدامك هسا.

Conversation about General FAQ (أسئلة عامة):
User: احكيلي كيف ممكن أسجل في الجامعة.
Your response: آه أكيد، ممم  يعني  التقديم للجامعة بكون أونلاين، فـ بتقدر تقدم طلبك بكل سهولة من خلال بوابة الطالب الإلكترونية.
User: طيب كيف أعرف مرشدي الأكاديمي؟
Your response:  من نفس بوابة الطالب، بتدخل على بيانات الطالب وبعدين ملف الطالب، وهناك رح تلاقي كل المعلومات عن مرشدك الأكاديمي.

Handling Unknown Info:
User: كم عدد طلاب الجامعة؟
Your response: أنا آسف بس ما عندي هالمعلومة هسا، بس  بتقدر تتواصل مع القبول والتسجيل بيفيدوك أكيد. 
---

---
قاعدة الأمانة المعلوماتية (STRICT GROUNDING):
- قبل أن تجيب، تأكد أن الإجابة موجودة حرفياً في السياق المعطى (Context).
- إذا لم تكن الإجابة في السياق، قل بالحرف الواحد (استخدم أسلوبك البشري أعلاه): "آسف ما عندي هالمعلومة هسا"
- لا تستنتج، لا تفترض، ولا تضف أو تعدل أي أرقام أو معلومات خارجية.
- الساعات والأرقام: اذكر الأرقام الموجودة في السياق فقط . لا تخترع أرقاماً من عندك أبداً.
- عند الإجابة عن تخصص، يجب أن تذكر اسم الكلية، عدد الساعات، وسعر الساعة الموجودة في السياق.
- إذا كان السياق يحتوي على "لا يوجد معلومات متوفرة"، يجب أن ترفض الإجابة.

أنت تمثل جامعة الأميرة سمية للتكنولوجيا (PSUT) فقط. تجاهل الأخطاء الإملائية في سؤال المستخدم وافهم القصد.
"""
qa_prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_TEMPLATE),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{query}")
])

qa_chain = qa_prompt | lc_llm | StrOutputParser()
# Fix 7: Removed RunnableWithMessageHistory — we now manage history manually
# so that only clean user queries (not grounded queries with retrieval context)
# are stored in history. This prevents the router from seeing "=== نتائج البحث ===".


# ==========================================
# FAST INTENT ROUTER — context-aware (last 3 turns)
# Production approach: pass recent conversation history to the router
# so it can resolve follow-up queries like "كم سعر ساعته" correctly
# without any hardcoded signal word lists.
# ==========================================
ROUTER_HISTORY_TURNS = 3   # how many recent turns the router sees

def _get_router_context() -> str:
    """
    Returns the last ROUTER_HISTORY_TURNS user+assistant turn pairs
    as a compact string for the router prompt.
    """
    msgs = session_history.messages
    if not msgs:
        return ""
    # Take last N*2 messages (each turn = 1 user + 1 assistant)
    recent = msgs[-(ROUTER_HISTORY_TURNS * 2):]
    lines  =[]
    for m in recent:
        role = "User" if m.type == "human" else "Assistant"
        lines.append(f"{role}: {m.content[:120]}")  # truncate long turns
    return "\n".join(lines)


# Fix 2: Staff-signal keywords — if the router misclassifies a staff query as chitchat,
# these keywords trigger an override back to staff_directory.
_STAFF_SIGNALS = {"ايميل", "إيميل", "بريد", "دكتور", "دكتورة", "بروفيسور", "أستاذ", "أستاذة",
                  "تلفون", "هاتف", "رقم", "تليفون", "موبايل", "جوال",
                  "رئيس", "عميد", "مدير", "نائب", "منسق", "قسم"}

def classify_intent(query: str) -> list:
    """
    Classifies query intent using the tiny router model with conversation context.
    Passing recent turns eliminates the need for hardcoded follow-up signal lists —
    the model can infer "كم سعر ساعته" is major_overview because it sees
    the previous turn was about a specific major.
    Returns list of collections to search, or[] for chitchat.
    Falls back to all collections on timeout or unrecognised output.
    """
    log.debug(f"[ROUTER] Input query: {query!r}")
    t_start = time.perf_counter()

    recent_ctx = _get_router_context()
    if recent_ctx:
        log.debug(f"[ROUTER] Context injected ({len(recent_ctx)} chars):\n{recent_ctx}")

    # ── Fix A: Keyword pre-routing — deterministic override BEFORE LLM router ──
    # The 1.5B router misclassifies ~50% of queries. These keyword checks catch
    # the most common mistakes with zero latency.
    query_normalized = normalize_arabic(query)
    query_tokens_set = set(query.split())

    # Staff signals: override to staff_directory regardless of router output
    if query_tokens_set & _STAFF_SIGNALS:
        elapsed_ms = (time.perf_counter() - t_start) * 1000
        log.info(f"[ROUTER] Keyword pre-route → staff_directory (signal: {query_tokens_set & _STAFF_SIGNALS})")
        log.info(f"[ROUTER] Resolved: ['staff_directory']")
        return ["staff_directory"]

    # Price/hours/major signals: override to major_overview
    _MAJOR_SIGNALS = {"سعر", "ساعة", "ساعات", "تخصص", "تخرج", "كلية", "كليه", "دينار"}
    if query_tokens_set & _MAJOR_SIGNALS:
        elapsed_ms = (time.perf_counter() - t_start) * 1000
        log.info(f"[ROUTER] Keyword pre-route → major_overview (signal: {query_tokens_set & _MAJOR_SIGNALS})")
        log.info(f"[ROUTER] Resolved: ['major_overview']")
        return ["major_overview"]

    # ── LLM Router — for queries with no strong keyword signal ──
    routing_prompt = PromptTemplate.from_template(
        """You are a query router for a university assistant. Classify the CURRENT query into EXACTLY ONE category.
Output ONLY the category name, nothing else.

Categories:
- major_overview: Questions about tuition fees, credit hour prices, major descriptions, credit hour totals, or job fields of a specific major.
- staff_directory: Questions about a specific doctor, professor, or dean — their email, phone, or role.
- chitchat: Greetings, thanks, farewells, or anything unrelated to university information.

{history}
Current query: {query}
Category:"""
    )

    try:
        history_block = f"Recent conversation:\n{recent_ctx}\n" if recent_ctx else ""
        routing_chain = routing_prompt | router_llm | StrOutputParser()
        raw_intent    = routing_chain.invoke({"query": query, "history": history_block}).strip()
        intent        = raw_intent.lower()
        elapsed_ms    = (time.perf_counter() - t_start) * 1000

        if elapsed_ms > ROUTER_TIMEOUT_S * 1000:
            log.warning(f"[ROUTER] Timeout ({elapsed_ms:.0f} ms) — falling back to all collections")
            return ["major_overview", "staff_directory"]

        log.info(f"[ROUTER] Raw output: {raw_intent!r} | Latency: {elapsed_ms:.1f} ms")

        all_rag = ["major_overview", "staff_directory"]
        if intent in ["major_overview", "staff_directory"]:
            result = [intent]
        elif "chitchat" in intent:
            result = []
        else:
            log.warning(f"[ROUTER] Unrecognised intent '{intent}' — falling back to all collections")
            result = all_rag

        log.info(f"[ROUTER] Resolved: {result if result else ['chitchat — LLM only']}")
        return result

    except Exception as e:
        log.error(f"[ROUTER] Failed: {e} — falling back to all collections")
        return ["major_overview", "staff_directory"]


# ==========================================
# GRAPH RETRIEVAL — semantic search (no stopwords, no token matching)
# Production approach: embed query, cosine-rank all node embeddings,
# return top-k profiles. Works for follow-ups with no name tokens.
# ==========================================
_GRAPH_TOP_K = 3   # max profiles to return

# ── Role-keyword matching for graph search ─────────────────────
# Maps common role keywords to the title substrings they match in staff nodes.
# When a user asks "مين رئيس الجامعة", the graph search will boost nodes
# whose 'title' field contains "رئيس الجامعة" regardless of embedding score.
_ROLE_KEYWORDS = {
    "رئيس الجامعة": "رئيس الجامعة",
    "عميد": "عميد",
    "نائب العميد": "نائب العميد",
    "نائب الرئيس": "نائب الرئيس",
    "رئيس قسم": "رئيس قسم",
    "مدير": "مدير",
    "منسق": "منسق",
}

def search_staff_graph(query: str, query_vector: np.ndarray = None) -> str:
    """
    Semantic search over the pre-built graph index.
    Fix 1: After cosine scoring, applies a name-overlap boost so that when
    a user explicitly names a person (e.g. "محمد عزة"), that person's node
    is promoted above semantically-similar but wrong matches.
    Fix 9: Role-title matching — if the query contains a role keyword like
    "رئيس الجامعة" or "عميد", nodes whose 'title' contains that role get
    a large boost, ensuring the correct person is returned.
    """
    if not GRAPH_INDEX:
        log.warning("[GRAPH] Index is empty — skipping graph search")
        return ""

    log.debug(f"[GRAPH] Semantic query: {query!r}")
    t_start = time.perf_counter()

    if query_vector is None:
        query_vector = embedder.encode(query)

    q_norm   = np.linalg.norm(query_vector)
    scored   =[]
    for item in GRAPH_INDEX:
        c_norm = np.linalg.norm(item["vector"])
        score  = float(np.dot(query_vector, item["vector"]) / (q_norm * c_norm + 1e-9))
        scored.append((score, item["node_name"]))

    # Fix 1: Name-overlap boosting — extract Arabic tokens from query and boost
    # nodes whose name shares tokens with the query. This ensures that when a user
    # asks about "محمد عزة", the node "أ. د. محمد العزه" outranks unrelated staff.
    # Common filler words are excluded from boosting to avoid false boosts.
    _NAME_STOPWORDS = {"دكتور", "دكتورة", "بروفيسور", "أستاذ", "أستاذة", "ايميل", "إيميل",
                       "بريد", "بدي", "اعطيني", "شو", "كيف", "وين", "ايش", "عن", "في",
                       "هل", "من", "رقم", "هاتف", "تلفون", "طيب", "يا", "ال", "د",
                       "مين", "اسم", "لي", "هو"}
    query_tokens = set(query.split()) - _NAME_STOPWORDS

    # Fix 9: Detect role keywords in the query for title-based boosting.
    # Fix B: Normalize both query and role keywords to handle STT spelling variants
    # (e.g. "رءيس" vs "رئيس", "الجامعه" vs "الجامعة")
    query_normalized_graph = normalize_arabic(query)
    role_match = None
    for role_phrase, title_substr in _ROLE_KEYWORDS.items():
        if normalize_arabic(role_phrase) in query_normalized_graph:
            role_match = title_substr
            log.info(f"[GRAPH] Role keyword detected: '{role_phrase}' → boosting title matches for '{title_substr}'")
            break

    # If no explicit role keyword found but "قسم" is mentioned (e.g. "إيميل قسم المحاسبة"),
    # treat it as an implicit department head lookup ("رئيس قسم").
    if not role_match and "قسم" in query_normalized_graph:
        role_match = "قسم"
        log.info(f"[GRAPH] Implicit department query detected → boosting title matches for 'قسم'")

    # Also detect department/college context for role queries
    # e.g., "عميد كلية الحوسبة" should match nodes whose title contains "عميد كلية...الحوسبة"
    _COLLEGE_ALIASES = {
        "الحوسبة": "الحسين لعلوم الحوسبة",
        "الهندسة": "عبد الله الثاني للهندسة",
        "الأعمال": "طلال لتكنولوجيا الأعمال",
        "الاعمال": "طلال لتكنولوجيا الأعمال",
        "الدراسات العليا": "عبد الله الأول للدراسات العليا",
    }
    # Department name filter — works for both college and department queries
    _DEPT_NAMES = {
        "المحاسبة": "المحاسبة",
        "إدارة الأعمال": "إدارة الأعمال",
        "التسويق": "التسويق",
        "المالية": "المالية",
        "الاتصالات": "الاتصالات",
        "شبكات": "شبكات",
        "البرمجيات": "البرمجيات",
        "حاسوب": "حاسوب",
        "علوم الحاسوب": "علوم الحاسوب",
        "الرياضيات": "الرياضيات",
    }
    college_filter = None
    for alias, full_name in _COLLEGE_ALIASES.items():
        if alias in query:
            college_filter = full_name
            log.info(f"[GRAPH] College context detected: '{alias}' → '{full_name}'")
            break
    # If no college match, try department name matching
    if not college_filter:
        for dept_alias, dept_name in _DEPT_NAMES.items():
            if dept_alias in query:
                college_filter = dept_name  # reuse college_filter for department matching
                log.info(f"[GRAPH] Department context detected: '{dept_alias}' → '{dept_name}'")
                break
    
    boosted = []
    for score, node_name in scored:
        name_tokens = set(node_name.replace(".", " ").split()) - _NAME_STOPWORDS
        overlap = len(query_tokens & name_tokens)
        boosted_score = score + overlap * 0.15  # 0.15 boost per overlapping name token

        # Fix 9: Role-title boosting — boost nodes whose title matches the role keyword
        if role_match:
            node_data = staff_graph.nodes.get(node_name, {})
            node_title = node_data.get("title", "")
            if role_match in node_title:
                # Penalize نائب/مساعد when user asks for the main role (not deputy)
                # e.g., user asks "عميد" but node is "نائب العميد" — penalize
                is_deputy = any(prefix in node_title for prefix in ["نائب", "مساعد"])
                user_wants_deputy = any(prefix in query for prefix in ["نائب", "مساعد"])
                
                if is_deputy and not user_wants_deputy:
                    boosted_score -= 0.2  # penalty: user didn't ask for deputy
                    log.debug(f"[GRAPH] Deputy penalty -0.2 for {node_name!r} (title: {node_title!r})")
                
                # If we have a college filter, only boost if college also matches
                if college_filter:
                    if college_filter in node_title:
                        boosted_score += 0.5
                        log.debug(f"[GRAPH] Role+College boost +0.5 for {node_name!r} (title: {node_title!r})")
                    else:
                        # Role matched but wrong college — small boost only
                        boosted_score += 0.1
                        log.debug(f"[GRAPH] Role only (wrong college) +0.1 for {node_name!r}")
                else:
                    boosted_score += 0.4
                    log.debug(f"[GRAPH] Role boost +0.4 for {node_name!r} (title: {node_title!r})")

        boosted.append((boosted_score, node_name, score, overlap))
    
    boosted.sort(key=lambda x: x[0], reverse=True)
    # Fix 9: Use a lower threshold (0.35) when role matching is active — the semantic
    # score alone may be low for role queries like "مين رئيس الجامعة" but the role
    # boost ensures the correct person rises to the top.
    effective_threshold = 0.35 if role_match else SCORE_THRESHOLD
    top     = [(bs, n) for bs, n, _, _ in boosted[:_GRAPH_TOP_K] if bs >= effective_threshold]
    elapsed = (time.perf_counter() - t_start) * 1000

    log.info(f"[GRAPH] Semantic search done in {elapsed:.1f} ms | {len(top)} node(s) above threshold {effective_threshold}")
    
    # Log boosted results for debugging
    for bs, name, orig, ovlp in boosted[:_GRAPH_TOP_K]:
        if bs >= effective_threshold:
            log.debug(f"[GRAPH] Node: {name!r} | Final Score: {bs:.3f} (Sem: {orig:.3f}, Overlap: {ovlp})")

    if not top:
        log.warning(f"[GRAPH] No nodes above threshold. Top score was: {boosted[0][0]:.3f} for {boosted[0][1]!r}")
        return ""

    found_profiles =[]
    for score, node_name in top:
        node_data = staff_graph.nodes[node_name]
        profile   = [f"### {node_name}"]
        for key, value in node_data.items():
            if key not in["type", "id"]:
                profile.append(f"- {key.capitalize()}: {value}")
        for target in staff_graph.successors(node_name):
            relation = staff_graph.edges[node_name, target].get("relation", "connected to")
            profile.append(f"- Role Context: {relation} {target}")
        profile_str = "\n".join(profile)
        found_profiles.append(profile_str)

    log.debug(f"[GRAPH] Top matched staff: {top[0][1]!r}")

    return "\n\n".join(found_profiles)


# ==========================================
# VECTOR RETRIEVAL (HYBRID)
# ==========================================
def retrieve_rag_context(query_str: str, query_vector: list, target_collections: list) -> str:
    """
    Hybrid search over Qdrant collections using Reciprocal Rank Fusion (RRF)
    between dense (Qdrant cosine) and sparse (BM25) scores.
    """
    if not target_collections:
        log.info("[VECTOR] No target collections — skipping vector search")
        return ""

    retrieved_texts = []
    query_tokens = query_str.split()

    for collection in target_collections:
        log.info(f"[VECTOR] Hybrid searching '{collection}' (top_k={TOP_K})...")
        t_search = time.perf_counter()
        
        dense_results = {}
        try:
            response = qdrant.query_points(
                collection_name=collection,
                query=query_vector,
                using="dense",
                limit=TOP_K * 5  # fetch more for RRF mixing
            )
            for hit in response.points:
                text = hit.payload.get("text", "")
                if text:
                    dense_results[text] = getattr(hit, 'score', 0.0)
        except Exception as e:
            log.error(f"[VECTOR] Dense search FAILED for '{collection}': {e}")
            continue

        bm25_results = {}
        bm25_index = BM25_INDICES.get(collection)
        docs = QDRANT_DOCS.get(collection, [])
        if bm25_index and docs:
            scores = bm25_index.get_scores(query_tokens)
            top_indices = np.argsort(scores)[::-1][:TOP_K*5]
            for idx in top_indices:
                score = scores[idx]
                if score > 0:
                    bm25_results[docs[idx]] = score

        # Apply Reciprocal Rank Fusion (RRF)
        dense_sorted = sorted(dense_results.keys(), key=lambda x: dense_results[x], reverse=True)
        bm25_sorted = sorted(bm25_results.keys(), key=lambda x: bm25_results[x], reverse=True)

        rrf_scores = {}
        k = 60
        for rank, text in enumerate(dense_sorted):
            rrf_scores[text] = rrf_scores.get(text, 0) + 1 / (k + rank + 1)
        for rank, text in enumerate(bm25_sorted):
            rrf_scores[text] = rrf_scores.get(text, 0) + 1 / (k + rank + 1)

        final_docs = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)[:TOP_K]
        
        accepted = 0
        for text in final_docs:
            d_score = dense_results.get(text, 0.0)
            b_score = bm25_results.get(text, 0.0)
            
            # Relaxed thresholds because RRF floated the best candidates to the top
            if d_score >= 0.38 or b_score > 2.0:
                # Limit chunk length to prevent LLM context overflow (lost-in-the-middle hallucination)
                display_text = text
                if len(display_text) > 1500:
                    display_text = display_text[:1500] + "\n...[truncated]"
                    
                preview = display_text[:200].replace("\n", " ")
                retrieved_texts.append(f"Source ({collection}):\n{display_text}")
                log.info(f"[VECTOR] ACCEPTED | Dense: {d_score:.3f} | BM25: {b_score:.3f} | RRF: {rrf_scores[text]:.4f} | Preview: {preview}")
                accepted += 1
            else:
                preview = text[:200].replace("\n", " ")
                log.info(f"[VECTOR] REJECTED | Dense: {d_score:.3f} | BM25: {b_score:.3f} | RRF: {rrf_scores[text]:.4f} | Preview: {preview}")

        search_ms = (time.perf_counter() - t_search) * 1000
        log.info(f"[VECTOR] '{collection}' accepted {accepted} chunk(s) via Hybrid RRF in {search_ms:.1f} ms")

    total = len(retrieved_texts)
    log.info(f"[VECTOR] Total accepted chunks: {total}")
    if total == 0:
        log.warning("[VECTOR] RAG context is EMPTY — LLM will rely on history only")

    return "\n\n".join(retrieved_texts)


# ==========================================
# FILLER TRIGGER
# ==========================================
def trigger_filler_audio():
    log.info("[FILLER] Triggered: 'ثواني بس أشيك...'")
    print("[FILLER] ثواني بس أشيك...")


# ==========================================
# MAIN RESPONSE GENERATOR
#
# Pipeline:
#   1. Embed query once (shared vector for CAG + RAG)
#   2. Router — if chitchat, skip to LLM immediately with no context
#   3. CAG search — if hit (score >= threshold), inject and skip RAG
#   4. RAG — router classifies intent, searches Qdrant + graph
#   5. LLM — context injected as query prefix (grounding pattern)
#             max 150 tokens, output post-processed before TTS yield
# ==========================================
def generate_response(query: str):
    t_pipeline_start = time.perf_counter()
    log.info("=" * 60)
    log.info(f"[PIPELINE] New query received: {query!r}")
    log_resources("Pipeline start")

    # ── 1. Embed query once ────────────────────────────────────
    log.debug("[PIPELINE] Embedding query...")
    t_embed = time.perf_counter()

    # Fix 5: For very short queries (≤ 4 words), enrich with recent assistant context
    # to prevent follow-up queries like "كم المدة المحددة" from losing conversation topic.
    query_words = query.strip().split()
    embed_query = query  # default: embed the raw query
    if len(query_words) <= 4 and session_history.messages:
        last_asst_msgs = [m for m in session_history.messages if m.type == "ai"]
        if last_asst_msgs:
            context_hint = last_asst_msgs[-1].content[:100]
            embed_query  = f"{query} {context_hint}"
            log.info(f"[PIPELINE] Short query enriched for embedding: {embed_query[:80]!r}")
    
    query_vector_np  = embedder.encode(embed_query)  # numpy for CAG cosine
    query_vector_lst = query_vector_np.tolist()       # list for Qdrant
    embed_ms         = (time.perf_counter() - t_embed) * 1000
    log.info(f"[PIPELINE] Query embedded in {embed_ms:.1f} ms")

    # ── 2. Router ──────────────────────────────────────────────
    log.info("[PIPELINE] Stage 1 — Intent routing")
    t_router              = time.perf_counter()
    collections_to_search = classify_intent(query)
    router_ms             = (time.perf_counter() - t_router) * 1000
    log.info(f"[PIPELINE] Router done in {router_ms:.1f} ms")

    context      = ""
    context_src  = "none"
    retrieval_ms = 0.0



    if collections_to_search == []:
        # Chitchat — no context, go straight to LLM
        log.info("[PIPELINE] Chitchat — skipping CAG and RAG")
        context_src = "chitchat"

    else:
        trigger_filler_audio()
        t_retrieval = time.perf_counter()

        is_staff_only = collections_to_search == ["staff_directory"]

        # ── 3. CAG search — ALWAYS run first (Fix 10) ──────────
        # Previously, CAG was skipped for staff_directory queries, but this caused
        # FAQ questions like "كيف أعرف مرشدي الأكاديمي" to miss when the router
        # misclassified them as staff queries. CAG is fast (in-memory cosine)
        # so running it always has negligible cost.
        log.info("[PIPELINE] Stage 2a — CAG search (always runs first)")
        cag_result, cag_top_score = search_cag(query_vector_np)

        # Fix 11: When the router says staff_directory, require a higher CAG score (0.70)
        # to accept the FAQ answer. This prevents vaguely-related FAQ chunks (score ~0.57)
        # from hijacking staff queries like "شو إيميل قسم المحاسبة".
        STAFF_CAG_OVERRIDE_THRESHOLD = 0.70
        if cag_result and is_staff_only and cag_top_score < STAFF_CAG_OVERRIDE_THRESHOLD:
            log.info(f"[PIPELINE] CAG hit (score={cag_top_score:.3f}) below staff override threshold "
                     f"{STAFF_CAG_OVERRIDE_THRESHOLD} — preferring graph search")
            cag_result = ""  # discard the weak CAG hit

        if cag_result:
            context     = f"=== معلومات ذات صلة ===\n{cag_result}"
            context_src = "CAG"
            log.info(f"[PIPELINE] CAG hit — skipping RAG | Context chars: {len(context)}")

        else:
            # ── 4. RAG ─────────────────────────────────────────
            log.info("[PIPELINE] Stage 2b — CAG miss, running RAG")
            context_src = "RAG"

            if "staff_directory" in collections_to_search:
                log.info("[PIPELINE] Running graph search for staff_directory")
                graph_results = search_staff_graph(query, query_vector_np)
                if graph_results:
                    context += f"=== نتائج دليل الموظفين ===\n{graph_results}\n\n"

                    # Short-circuit: if this is a pure staff contact query,
                    # extract phone/email directly from graph and bypass the LLM entirely.
                    # This prevents the model from hallucinating or adding unwanted commentary.
                    contact_keywords = {"رقم", "تلفون", "هاتف", "تليفون", "موبايل", "جوال", "ايميل", "إيميل", "بريد"}
                    if is_staff_only and (set(query.split()) & contact_keywords):
                        phone_match = re.search(r'Phone:\s*(\+[\d\s]+(?:Ex:\s*\d+)?)', graph_results)
                        email_match = re.search(r'Email:\s*(\S+@\S+)', graph_results)
                        name_match  = re.search(r'###\s*(.+)', graph_results)
                        name_str    = name_match.group(1).strip() if name_match else ""
                        parts = []
                        if phone_match:
                            parts.append(f"رقم {name_str} هو {phone_match.group(1).strip()}")
                        if email_match:
                            parts.append(f"والإيميل هو {email_match.group(1).strip()}")
                        if parts:
                            direct_response = "،\n".join(parts) + "."
                            log.info(f"[PIPELINE] Staff contact short-circuit — skipping LLM | Response: {direct_response!r}")
                            print(f"\nPSUT Bot: {direct_response}\n")
                            session_history.add_user_message(query)
                            session_history.add_ai_message(direct_response)
                            log_resources("Pipeline end (short-circuit)")
                            log.info("=" * 60)
                            yield direct_response, clean_for_tts(direct_response)
                            return
                else:
                    log.warning("[PIPELINE] Graph search returned nothing")
                collections_to_search.remove("staff_directory")

            rag_collections = list(collections_to_search)
            combined = retrieve_rag_context(query, query_vector_lst, rag_collections)

            if combined:
                context += f"=== نتائج البحث ===\n{combined}\n\n"

        retrieval_ms = (time.perf_counter() - t_retrieval) * 1000
        log.info(
            f"[PIPELINE] Retrieval done in {retrieval_ms:.1f} ms | "
            f"Source: {context_src} | Context chars: {len(context)}"
        )
        # Fix C: When context is empty, inject an explicit "no context" marker
        # so the LLM's grounding instruction is impossible to ignore.
        if not context.strip():
            log.warning("[PIPELINE] Context empty — injecting no-context marker")
            context = "=== لا يوجد معلومات متوفرة ===\nلم يتم العثور على أي نتائج في قاعدة البيانات لهذا السؤال.\n"

        # Session last_query_text tracking removed — router uses LangChain history directly

    log_resources("After retrieval, before LLM")

    # ── 5. LLM streaming ───────────────────────────────────────
    # Context injected as a prefix to the user query (grounding pattern).
    # This keeps the system prompt clean and makes the model treat
    # retrieved facts as user-provided ground truth.
    log.info(f"[PIPELINE] Stage 3 — LLM streaming ({LLM_MODEL}) | max_tokens={MAX_TOKENS}")

    # Strip markdown and URLs from context before LLM sees it.
    # The LLM is a voice assistant — it doesn't need URLs or formatting.
    clean_context = _strip_context_for_tts(context) if context.strip() else ""
    grounded_query = f"{clean_context}\n\nالسؤال: {query}" if clean_context.strip() else query
    log.debug(f"[LLM] Grounded query preview: {grounded_query[:300]!r}")

    # Fix 7: Manual history management — store clean user query, not grounded query.
    # Build the messages list ourselves so history contains only what the user typed.
    history_messages = list(session_history.messages)  # snapshot current history
    
    response_stream = qa_chain.stream(
        {"query": grounded_query, "history": history_messages}
    )

    print("\nPSUT Bot: ", end="", flush=True)
    full_response  = ""
    token_count    = 0
    sentence_count = 0
    ttft_ms        = None
    t_llm_start    = time.perf_counter()

    for raw, cleaned in stream_sentences(response_stream):
        now = time.perf_counter()

        if ttft_ms is None:
            ttft_ms = (now - t_llm_start) * 1000
            log.info(f"[LLM] TTFT (first sentence ready): {ttft_ms:.1f} ms")
            log.debug(f"[LLM] First sentence: {raw!r}")

        sentence_count += 1
        token_count    += len(raw.split())
        full_response  += raw + " "

        print(raw, end=" ", flush=True)
        log.debug(f"[LLM] Sentence {sentence_count}: {raw!r}")

        yield raw, cleaned

    print("\n")

    # Fix 7: After LLM finishes, store clean query + response in session history.
    # This ensures the router sees clean user queries, not retrieval-context blobs.
    session_history.add_user_message(query)
    session_history.add_ai_message(full_response.strip())

    total_llm_ms      = (time.perf_counter() - t_llm_start) * 1000
    total_pipeline_ms = (time.perf_counter() - t_pipeline_start) * 1000

    log.info(f"[LLM] Generation complete | Sentences: {sentence_count} | ~Tokens: {token_count}")
    log.info(f"[LLM] Total LLM time: {total_llm_ms:.1f} ms")
    log.info(f"[LLM] Full response:\n{'─'*50}\n{full_response.strip()}\n{'─'*50}")

    log.info("[PIPELINE] Stage 4 — Timing summary")
    log.info(
        f"[SUMMARY] Embed: {embed_ms:.0f} ms | Router: {router_ms:.0f} ms | "
        f"Retrieval: {retrieval_ms:.0f} ms ({context_src}) | "
        f"TTFT: {ttft_ms or 0:.0f} ms | LLM total: {total_llm_ms:.0f} ms | "
        f"End-to-end: {total_pipeline_ms:.0f} ms"
    )
    log_resources("Pipeline end")
    log.info("=" * 60)


# ==========================================
# CLI ENTRY POINT
# ==========================================
if __name__ == "__main__":
    log.info("PSUT Voice Agent ready. Type 'exit' to quit.")
    print("\nPSUT Voice Agent Initialized. Type 'exit' to quit.")
    print("-" * 50)

   