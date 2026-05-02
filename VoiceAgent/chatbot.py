# ==========================================
# WINDOWS 11 INSTALLATION REQUIREMENTS:
# pip install langchain langchain-core langchain-ollama qdrant-client sentence-transformers networkx psutil pynvml numpy
# ==========================================

import os

# Fix 10: These MUST be set before any sentence_transformers import.
# The library reads the environment at import time, not at model load time.
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

# --- LangChain Imports ---
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.chat_history import InMemoryChatMessageHistory
# RunnableWithMessageHistory removed — Fix 7 uses manual history management


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

SCORE_THRESHOLD  = 0.45   # Minimum cosine similarity to accept a chunk
CAG_THRESHOLD    = 0.50   # Raised from 0.45 to reduce false-positive FAQ matches (Fix 6)
TOP_K            = 4      # Chunks retrieved per collection
ROUTER_TIMEOUT_S = 0.5    # Router max seconds before fallback to all collections
MAX_TOKENS       = 200    # Hard cap on LLM output tokens

# Qdrant RAG collections only. general_faq is CAG (in-memory), never Qdrant.
ACTIVE_COLLECTIONS = {"student_guide", "major_overview"}


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
# Whitelist: Arabic letters (\u0600-\u06FF), digits, Arabic punctuation,
# standard punctuation used in Arabic, space, newline
_ALLOWED_RE  = re.compile(
    r'[^\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF'  # Arabic script blocks
    r'\uFB50-\uFDFF\uFE70-\uFEFF'                    # Arabic presentation forms
    r'0-9\s،؟!\.،:\-_@\.]+',                         # digits, punctuation, email chars
    re.UNICODE
)
# Sentence ending with colon — dangling "عبر الرابط:" after URL strip
_DANGLING_COLON_RE = re.compile(r'[^\s]+:\s*$', re.MULTILINE)


def clean_for_tts(text: str) -> str:
    """
    Whitelist-based TTS cleaner — production approach.
    1. Extract emails and phones, store them, replace with digit-only placeholders
       that survive the whitelist strip (digits are whitelisted).
    2. Strip everything not in the Arabic/digit/punctuation whitelist.
    3. Restore emails and phones using the stored values.
    4. Remove dangling "عبر الرابط:" fragments left after URL removal.
    5. Collapse whitespace.
    """
    # Step 1: extract and placeholder emails and phones.
    # Placeholders use Arabic-script marker + digits only — both survive the whitelist.
    # We use a dict so restoration is exact regardless of whitelist mutation.
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

    # Step 2: whitelist strip — placeholders survive because they are Arabic+digits
    text = _ALLOWED_RE.sub(" ", text)

    # Step 3: restore original emails and phones
    for token, value in store.items():
        text = text.replace(token, value)

    # Step 4: remove dangling colon fragments (e.g. "عبر الرابط:")
    text = _DANGLING_COLON_RE.sub("", text)

    # Step 5: collapse whitespace
    text = re.sub(r'[ \t]+', ' ', text).strip()
    return text


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
                yield cleaned
                
    # Fix 4: Always yield remaining text if it contains meaningful content.
    # Previously, valid responses containing emails (e.g. "محمد العزة هو m.azzeh@psut.edu.jo.")
    # were discarded because the sentence boundary logic was too strict.
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
                yield cleaned
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
# CAG — general_faq.md loaded and chunked in memory
# Searched via cosine similarity before RAG is attempted.
# general_faq is NEVER ingested into Qdrant.
# ==========================================
def _load_and_chunk_faq(path: str, chunk_size: int = 500) -> list:
    """
    Loads general_faq.md, splits into chunks by paragraph, embeds each one.
    Returns list of {text, vector} dicts for in-memory cosine search.
    """
    if not os.path.exists(path):
        log.warning(f"[CAG] general_faq.md not found at: {path}")
        return[]
    with open(path, 'r', encoding='utf-8') as f:
        content = f.read()

    paragraphs = [p.strip() for p in content.split("\n\n") if p.strip()]
    chunks, buffer =[], ""
    for para in paragraphs:
        if len(buffer) + len(para) < chunk_size:
            buffer += "\n\n" + para
        else:
            if buffer:
                chunks.append(buffer.strip())
            buffer = para
    if buffer:
        chunks.append(buffer.strip())

    log.info(f"[CAG] Embedding {len(chunks)} FAQ chunks in memory...")
    _t      = time.perf_counter()
    vectors = embedder.encode(chunks, batch_size=32, show_progress_bar=False)
    log.info(f"[CAG] FAQ chunks embedded in {(time.perf_counter() - _t)*1000:.0f} ms")
    return[{"text": c, "vector": v} for c, v in zip(chunks, vectors)]


FAQ_PATH   = os.path.join(ROOT, "KB", "general_faq.md")
log.info("[INIT] Loading and embedding CAG (general_faq.md)...")
CAG_CHUNKS = _load_and_chunk_faq(FAQ_PATH)
log.info(f"[CAG] Ready — {len(CAG_CHUNKS)} chunks in memory")


def search_cag(query_vector: np.ndarray) -> str:
    """
    Cosine similarity search over in-memory FAQ chunks.
    Returns concatenated text of chunks scoring >= SCORE_THRESHOLD, or "".
    """
    if not CAG_CHUNKS:
        return ""

    results =[]
    q_norm  = np.linalg.norm(query_vector)
    for chunk in CAG_CHUNKS:
        c_norm = np.linalg.norm(chunk["vector"])
        score  = float(np.dot(query_vector, chunk["vector"]) / (q_norm * c_norm + 1e-9))
        if score >= SCORE_THRESHOLD:
            results.append((score, chunk["text"]))

    if not results:
        log.info(f"[CAG] No chunk above threshold {SCORE_THRESHOLD} — will proceed to RAG")
        return ""

    results.sort(key=lambda x: x[0], reverse=True)
    top = results[:TOP_K]
    log.info(f"[CAG] {len(top)} chunk(s) above threshold | Top score: {top[0][0]:.3f}")
    for i, (score, text) in enumerate(top):
        log.debug(f"[CAG] Chunk {i+1} | Score: {score:.3f} | Preview: {text[:200].replace(chr(10), ' ')}")
    return "\n\n".join(t for _, t in top)


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

def get_session_history(session_id: str):
    return session_history

# Session state — keeps last matched staff name for logging/debugging
_session = {
    "last_staff_name": None,   # str | None — set by semantic graph search
}

# _FOLLOWUP_SIGNALS removed — router now uses conversation history instead


# ==========================================
# SYSTEM PROMPT
# Clean static prompt — no CAG/RAG placeholders.
# Context is injected as a prefix to the user query (grounding pattern)
# so the model treats retrieved facts as user-provided, not instructions.
# ==========================================
SYSTEM_TEMPLATE = """You are the official smart voice assistant for Princess Sumaya University for Technology (PSUT). 
Your output is sent directly to a Text-to-Speech engine. 

ABSOLUTE RULES — NEVER VIOLATE UNDER ANY CIRCUMSTANCES:
1. NO EMOJIS: Never use emojis. They create garbage noise in TTS.
2. NO MARKDOWN: The retrieved context contains markdown (like ###, #, **). DO NOT COPY IT. Write in continuous, plain text paragraphs only. No bullet points, no asterisks, no dashes.
3. ARABIC ONLY: Use plain, flowing Arabic sentences. NEVER use English words mid-sentence or any other foreign scripts. 
   - EXCEPTION: Email addresses and phone numbers MUST be written exactly as they appear in the context (using English letters/numbers).
4. NUMBER FORMATTING: Convert all numbers into Arabic words (e.g., write "مئة وعشرون" instead of "120").
5. LENGTH: Keep responses concise (2 to 4 sentences maximum).
6. SUMMARIZE LISTS: If the context contains a long numbered list, summarize only the first two points in plain sentences.
7. CONTINUATION: Give the most important info first. Only ask "بدك أكمل؟" if there is genuinely more unmentioned information in the context.

---

التعليمات الشخصية (Persona & Tone):
أنت المساعد الذكي لجامعة الأميرة سمية للتكنولوجيا.
أجب بأسلوب ودي، احترافي، ومحادثة قريبة من اللهجة الأردنية المحكية (مثال: استخدم كلمات مثل "هسا"، "بتقدر"، "أكيد")، ولكن حافظ على المصطلحات الأكاديمية الرسمية كما هي.

قاعدة الأمانة المعلوماتية (STRICT GROUNDING):
- قبل أن تجيب، تأكد أن الإجابة موجودة حرفياً في السياق المعطى (Context).
- إذا لم تكن الإجابة في السياق، قل بالحرف الواحد: "آسف ما عندي هالمعلومة هسا، بتقدر تتواصل مع القسم المعني أو عمادة القبول والتسجيل."
- لا تستنتج، لا تفترض، ولا تضف أي معلومات خارجية.
- إذا طلب المستخدم رقم هاتف أو إيميل وكان موجوداً في السياق، انسخه كما هو بالضبط دون أي تعديل أو إضافة أرقام من عندك.

أنت تمثل جامعة الأميرة سمية للتكنولوجيا (PSUT) فقط. تجاهل الأخطاء الإملائية في سؤال المستخدم وافهم القصد."""

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
                  "تلفون", "هاتف", "رقم", "تليفون", "موبايل", "جوال"}

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

    routing_prompt = PromptTemplate.from_template(
        """You are a query router for a university assistant. Classify the CURRENT query into EXACTLY ONE category.
Output ONLY the category name, nothing else.

Categories:
- major_overview: Questions about tuition fees, credit hour prices, major descriptions, credit hour totals, or job fields of a specific major.
- student_guide: Questions about university rules, penalties, attendance, GPA, graduation, or academic warnings.
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
            return["major_overview", "student_guide", "staff_directory"]

        log.info(f"[ROUTER] Raw output: {raw_intent!r} | Latency: {elapsed_ms:.1f} ms")

        all_rag =["major_overview", "student_guide", "staff_directory"]
        if intent in["major_overview", "student_guide", "staff_directory"]:
            result = [intent]
        elif "chitchat" in intent:
            result = []
        else:
            log.warning(f"[ROUTER] Unrecognised intent '{intent}' — falling back to all collections")
            result = all_rag

        # Fix 2: Override chitchat → staff_directory if query contains staff signal keywords.
        # The tiny 1.5b router sometimes misclassifies queries starting with casual words
        # like "طيب" as chitchat even when they clearly ask about a staff member.
        if not result:  # chitchat
            query_tokens = set(query.split())
            if query_tokens & _STAFF_SIGNALS:
                result = ["staff_directory"]
                log.info("[ROUTER] Chitchat overridden → staff_directory (signal keywords detected)")

        log.info(f"[ROUTER] Resolved: {result if result else ['chitchat — LLM only']}")
        return result

    except Exception as e:
        log.error(f"[ROUTER] Failed: {e} — falling back to all collections")
        return["major_overview", "student_guide", "staff_directory"]


# ==========================================
# GRAPH RETRIEVAL — semantic search (no stopwords, no token matching)
# Production approach: embed query, cosine-rank all node embeddings,
# return top-k profiles. Works for follow-ups with no name tokens.
# ==========================================
_GRAPH_TOP_K = 3   # max profiles to return

def search_staff_graph(query: str, query_vector: np.ndarray = None) -> str:
    """
    Semantic search over the pre-built graph index.
    Fix 1: After cosine scoring, applies a name-overlap boost so that when
    a user explicitly names a person (e.g. "محمد عزة"), that person's node
    is promoted above semantically-similar but wrong matches.
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
                       "هل", "من", "رقم", "هاتف", "تلفون", "طيب", "يا", "ال", "د"}
    query_tokens = set(query.split()) - _NAME_STOPWORDS
    
    boosted = []
    for score, node_name in scored:
        name_tokens = set(node_name.replace(".", " ").split()) - _NAME_STOPWORDS
        overlap = len(query_tokens & name_tokens)
        boosted_score = score + overlap * 0.15  # 0.15 boost per overlapping name token
        boosted.append((boosted_score, node_name, score, overlap))
    
    boosted.sort(key=lambda x: x[0], reverse=True)
    top     = [(bs, n) for bs, n, _, _ in boosted[:_GRAPH_TOP_K] if bs >= SCORE_THRESHOLD]
    elapsed = (time.perf_counter() - t_start) * 1000

    log.info(f"[GRAPH] Semantic search done in {elapsed:.1f} ms | {len(top)} node(s) above threshold {SCORE_THRESHOLD}")
    
    # Log boosted results for debugging
    for bs, name, orig, ovlp in boosted[:_GRAPH_TOP_K]:
        if bs >= SCORE_THRESHOLD:
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

    # Save last matched name for session continuity (still useful for logging)
    _session["last_staff_name"] = top[0][1]
    log.debug(f"[GRAPH] Saved last_staff_name: {top[0][1]!r}")

    return "\n\n".join(found_profiles)


# ==========================================
# VECTOR RETRIEVAL
# ==========================================
def retrieve_rag_context(query_vector: list, target_collections: list) -> str:
    """
    Searches Qdrant collections. Chunks below SCORE_THRESHOLD are discarded.
    Uses a pre-computed query vector to avoid re-embedding.
    """
    if not target_collections:
        log.info("[VECTOR] No target collections — skipping vector search")
        return ""

    retrieved_texts =[]

    for collection in target_collections:
        log.info(f"[VECTOR] Searching '{collection}' (top_k={TOP_K}, threshold={SCORE_THRESHOLD})...")
        t_search = time.perf_counter()
        try:
            response  = qdrant.query_points(
                collection_name=collection,
                query=query_vector,
                using="dense",
                limit=TOP_K
            )
            results   = response.points
            search_ms = (time.perf_counter() - t_search) * 1000
            log.info(f"[VECTOR] '{collection}' returned {len(results)} chunk(s) in {search_ms:.1f} ms")

            accepted = 0
            for i, hit in enumerate(results):
                score      = getattr(hit, 'score', 0.0)
                chunk_text = hit.payload.get("text", "")
                status     = "ACCEPTED" if score >= SCORE_THRESHOLD else f"REJECTED (score {score:.4f} < {SCORE_THRESHOLD})"
                log.info(f"[VECTOR] Chunk {i+1} | Score: {score:.4f} | {status}")

                if score < SCORE_THRESHOLD:
                    continue

                if chunk_text:
                    preview = chunk_text[:300].replace("\n", " ")
                    log.debug(f"[VECTOR] Chunk {i+1} preview: {preview}")
                    log.debug(f"[VECTOR] Chunk {i+1} full:\n{'─'*50}\n{chunk_text}\n{'─'*50}")
                    retrieved_texts.append(f"Source ({collection}):\n{chunk_text}")
                    accepted += 1
                else:
                    log.warning(f"[VECTOR] Chunk {i+1} from '{collection}' has empty text payload")

            if accepted == 0:
                log.warning(f"[VECTOR] All chunks from '{collection}' rejected by threshold")

        except Exception as e:
            log.error(f"[VECTOR] FAILED searching '{collection}': {e}", exc_info=True)
            continue

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

    if collections_to_search ==[]:
        # Chitchat — no context, go straight to LLM
        log.info("[PIPELINE] Chitchat — skipping CAG and RAG")
        context_src = "chitchat"

    else:
        trigger_filler_audio()
        t_retrieval = time.perf_counter()

        # Fix 2: staff_directory queries skip CAG entirely.
        # The graph always has more precise staff info than FAQ chunks.
        is_staff_only = collections_to_search == ["staff_directory"]

        # ── 3. CAG search (skipped for staff queries) ──────────
        if not is_staff_only:
            log.info("[PIPELINE] Stage 2a — CAG search")
            cag_result = search_cag(query_vector_np)
        else:
            log.info("[PIPELINE] Stage 2a — CAG skipped (staff_directory query)")
            cag_result = ""

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
                            yield direct_response
                            return
                else:
                    log.warning("[PIPELINE] Graph search returned nothing")
                collections_to_search.remove("staff_directory")

            # Fix 8: bias student_guide queries toward undergraduate content
            # by appending بكالوريوس to the embedding query when searching
            # student_guide, unless the query already mentions graduate study.
            grad_signals = {"ماجستير", "دكتوراه", "دراسات عليا", "شامل"}
            rag_collections = list(collections_to_search)
            if ("student_guide" in rag_collections
                    and not any(s in query for s in grad_signals)):
                biased_query    = query + " بكالوريوس"
                biased_vector   = embedder.encode(biased_query).tolist()
                log.info(f"[FIX8] Biasing student_guide search with 'بكالوريوس'")
                # Search student_guide with biased vector, others with original
                sg_results = retrieve_rag_context(biased_vector, ["student_guide"])
                other_cols =[c for c in rag_collections if c != "student_guide"]
                other_results = retrieve_rag_context(query_vector_lst, other_cols) if other_cols else ""
                combined = "\n\n".join(filter(None, [sg_results, other_results]))
            else:
                combined = retrieve_rag_context(query_vector_lst, rag_collections)

            if combined:
                context += f"=== نتائج البحث ===\n{combined}\n\n"

        retrieval_ms = (time.perf_counter() - t_retrieval) * 1000
        log.info(
            f"[PIPELINE] Retrieval done in {retrieval_ms:.1f} ms | "
            f"Source: {context_src} | Context chars: {len(context)}"
        )
        if not context.strip():
            log.warning("[PIPELINE] Context empty — LLM will rely on history only")

        # Session last_query_text tracking removed — router uses LangChain history directly

    log_resources("After retrieval, before LLM")

    # ── 5. LLM streaming ───────────────────────────────────────
    # Context injected as a prefix to the user query (grounding pattern).
    # This keeps the system prompt clean and makes the model treat
    # retrieved facts as user-provided ground truth.
    log.info(f"[PIPELINE] Stage 3 — LLM streaming ({LLM_MODEL}) | max_tokens={MAX_TOKENS}")

    grounded_query = f"{context}\n\nالسؤال: {query}" if context.strip() else query
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

    for sentence in stream_sentences(response_stream):
        now = time.perf_counter()

        if ttft_ms is None:
            ttft_ms = (now - t_llm_start) * 1000
            log.info(f"[LLM] TTFT (first sentence ready): {ttft_ms:.1f} ms")
            log.debug(f"[LLM] First sentence: {sentence!r}")

        sentence_count += 1
        token_count    += len(sentence.split())
        full_response  += sentence + " "

        print(sentence, end=" ", flush=True)
        log.debug(f"[LLM] Sentence {sentence_count}: {sentence!r}")

        yield sentence

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
        f"TTFT: {ttft_ms:.0f} ms | LLM total: {total_llm_ms:.0f} ms | "
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

   