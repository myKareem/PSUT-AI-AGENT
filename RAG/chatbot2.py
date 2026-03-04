# ==========================================
# WINDOWS 11 INSTALLATION REQUIREMENTS:
# pip install langchain langchain-core langchain-ollama qdrant-client sentence-transformers networkx
# ==========================================

import os
import re
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from dhakira.cache.semantic import SemanticCache
from dhakira.config import CacheConfig
import networkx as nx

# --- LangChain Imports ---
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

# ==========================================
# CONFIGURATION
# ==========================================
DB_PATH = "C:\\Users\\20220458\\Desktop\\GP\\PSUT-AI-AGENT\\local_qdrant_db"
MODEL_NAME = "Omartificial-Intelligence-Space/Arabic-Triplet-Matryoshka-V2"
LLM_MODEL = "qwen3:8b"
ROUTER_MODEL = "qwen2.5:0.5b"  # Tiny fast model for intent routing (~50ms vs ~2000ms)

# ==========================================
# SENTENCE BOUNDARY DETECTION FOR TTS CHUNKING
# ==========================================
# Punctuation marks that indicate a natural pause for TTS to speak a complete thought
SENTENCE_BOUNDARY_PATTERN = re.compile(r'[.،؟!؟\n]')


# ==========================================
# SEMANTIC CACHE (Dhakira)
# ==========================================
cache = SemanticCache(CacheConfig(
    enabled=True,
    max_size=32000,
    ttl_seconds=86400
))

print("Loading local embedding model (CPU/GPU)...")
embedder = SentenceTransformer(MODEL_NAME)

print("Connecting to local Qdrant database...")
qdrant = QdrantClient(path=DB_PATH)

print("Loading local Graph database...")
try:
    staff_graph = nx.read_graphml('C:\\Users\\20220458\\Desktop\\GP\\PSUT-AI-AGENT\\local_graph_db.graphml')
    print(f"Graph loaded successfully with {staff_graph.number_of_nodes()} nodes.")
except Exception as e:
    print(f"Warning: Graph DB not found or error loading. Initializing empty graph. Error: {e}")
    staff_graph = nx.DiGraph()

# ==========================================
# LLM SETUP
# ==========================================
lc_llm = ChatOllama(
    model=LLM_MODEL,
    temperature=0.0,
    num_ctx=8192
)

# Tiny router LLM — used only for intent classification to avoid latency from the 8b model
router_llm = ChatOllama(
    model=ROUTER_MODEL,
    temperature=0.0,
    num_ctx=512  # Minimal context needed for a single classification call
)

session_history = InMemoryChatMessageHistory()

def get_session_history(session_id: str):
    return session_history

# ==========================================
# SYSTEM PROMPT — ENGINEERED FOR TTS OUTPUT
# ==========================================
# KEY CHANGES vs the original:
#   - Removed all Markdown formatting instructions (bullets, bold) — TTS reads symbols literally
#   - Numbers must be written as Arabic words to prevent TTS mispronunciation
#   - Responses must be short and conversational; offer to continue rather than dumping long lists
#   - LLM is instructed to tolerate STT transcription errors (phonetic typos from speech recognition)
SYSTEM_TEMPLATE = """أنت المساعد الذكي الرسمي لجامعة الأميرة سمية للتكنولوجيا (PSUT).
مهمتك هي مساعدة الطلاب والإجابة على استفساراتهم الأكاديمية والإدارية بشكل واضح وموضوعي.
يجب أن تجيب دائماً باللهجة العربية الأردنية المحكية.

قواعد أساسية يجب اتباعها بدقة:

أولاً: لا تستخدم أي تنسيق نصي إطلاقاً. لا تستخدم النجوم ولا الهاشتاقات ولا الأقواس ولا النقاط الترقيمية للقوائم. اكتب فقرات عادية بالكامل لأن ردودك ستُقرأ بصوت عالٍ عبر نظام تحويل النص إلى كلام.

ثانياً: اكتب الأرقام بالكلمات العربية دائماً. مثلاً اكتب مئة وخمسون ساعة معتمدة بدلاً من 150. وعشرة آلاف دينار بدلاً من 10000.

ثالثاً: كن موجزاً وطبيعياً في حديثك. إذا كان الجواب يحتوي على خطوات متعددة أو قائمة طويلة، أعطِ الخطوة الأولى أو أهم نقطتين فقط ثم اسأل مثلا " بدك أكمل؟" فقط اسأل اذا كان في خطوات زيادة. لأن المستخدم لا يستطيع استيعاب معلومات كثيرة دفعة واحدة عبر الصوت.

رابعاً: لا تخترع معلومات. إذا لم تجد الإجابة في السياق المعطى، قل فقط: "آسف، ما عندي المعلومة هسا . تواصل مع القسم المعني أو عمادة القبول والتسجيل."

خامساً: أنت تمثل جامعة الأميرة سمية فقط. لا تذكر ولا تقارن بأي جامعة أخرى.

سادساً: تجاهل الأخطاء الإملائية الواردة في سؤال المستخدم وافهم قصده من السياق. نظام تحويل الصوت إلى نص أحياناً يكتب كلمات بشكل مختلف عن المقصود، مثلاً قد يكتب "تسهيل" بدلاً من "تسجيل". استنتج المعنى الصحيح وأجب على هذا الأساس.

=== السياق الثابت (CAG) ===
{cag_context}

=== السياق المسترجع ديناميكياً (RAG) ===
{rag_context}
"""

qa_prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_TEMPLATE),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{query}")
])

qa_chain = qa_prompt | lc_llm | StrOutputParser()

conversational_rag_chain = RunnableWithMessageHistory(
    qa_chain,
    get_session_history,
    input_messages_key="query",
    history_messages_key="history"
)

# ==========================================
# CAG LOADER
# ==========================================
def load_cag_context():
    cag_text = "=== general_faq ===\n"
    staff_path = "C:\\Users\\20220458\\Desktop\\GP\\PSUT-AI-AGENT\\RAG\\KB\\general_faq.md"
    if os.path.exists(staff_path):
        with open(staff_path, 'r', encoding='utf-8') as f:
            cag_text += f.read() + "\n\n"
    return cag_text

print("Building CAG system prompt...")
SYSTEM_CAG_CONTEXT = load_cag_context()

# ==========================================
# FAST INTENT ROUTER (tiny model replaces 8b for classification)
# ==========================================
def classify_intent(query: str) -> list:
    """
    Classifies query intent using a tiny fast LLM (~50-100ms) instead of the 8b model.
    This eliminates the 1-3 second dead-air latency bottleneck before retrieval starts.
    """
    routing_prompt = PromptTemplate.from_template(
        """Analyze the user query and classify it into EXACTLY ONE of the following categories. Output NOTHING ELSE except the category name.

Categories:
- study_plans: Questions about specific courses, prerequisites, or curriculum tables.
- major_overview: Questions about tuition prices, major descriptions, or credit hour totals.
- student_guide: Questions about university rules, penalties, code of conduct, or graduation policies.
- staff_directory: Questions about deans, professors, doctors, or their contact information.
- general_faq: Questions about procedural steps, ID cards, or portal access.

Query: {query}
Category:"""
    )

    routing_chain = routing_prompt | router_llm | StrOutputParser()
    intent = routing_chain.invoke({"query": query}).strip().lower()

    valid_collections = ["study_plans", "major_overview", "student_guide", "staff_directory"]
    if intent in valid_collections:
        return [intent]
    elif "general" in intent:
        return []  # Rely entirely on CAG
    else:
        return valid_collections  # Fallback: search all


# ==========================================
# GRAPH RETRIEVAL
# ==========================================
def search_staff_graph(query: str) -> str:
    stop_words = ["اعطني", "ايميل", "رقم", "هاتف", "دكتور", "دكتورة", "استاذ", "عميد", "عن", "من", "هو", "هي"]
    query_words = [w for w in query.strip().split() if len(w) > 2 and w not in stop_words]
    found_profiles = []

    for node_name, node_data in staff_graph.nodes(data=True):
        if node_data.get('type') == 'University':
            continue
        name_str = str(node_name)
        if any(word in name_str for word in query_words):
            profile = [f"### {node_name}"]
            for key, value in node_data.items():
                if key not in ['type', 'id']:
                    profile.append(f"- {key.capitalize()}: {value}")
            for target in staff_graph.successors(node_name):
                relation = staff_graph.edges[node_name, target].get('relation', 'connected to')
                profile.append(f"- Role Context: {relation} {target}")
            found_profiles.append("\n".join(profile))

    return "\n\n".join(found_profiles[:3]) if found_profiles else ""


# ==========================================
# VECTOR RETRIEVAL
# ==========================================
def retrieve_rag_context(query: str, target_collections: list, top_k: int = 2) -> str:
    if not target_collections:
        print("[DEBUG] No target collections. Skipping vector search.")
        return ""

    query_vector = embedder.encode(query).tolist()
    retrieved_texts = []

    for collection in target_collections:
        try:
            print(f"[DEBUG] Searching collection: {collection}...")
            response = qdrant.query_points(
                collection_name=collection,
                query=query_vector,
                using="dense",
                limit=top_k
            )
            results = response.points
            print(f"[DEBUG] Found {len(results)} chunks in {collection}.")
            for i, hit in enumerate(results):
                chunk_text = hit.payload.get("text", "")
                if chunk_text:
                    print(f"\n>>> CHUNK {i+1} FROM {collection} <<<\n{chunk_text}\n{'>' * 50}")
                    retrieved_texts.append(f"Source ({collection}):\n{chunk_text}")
        except Exception as e:
            print(f"\n[CRITICAL ERROR] Failed searching '{collection}': {e}\n")
            continue

    return "\n\n".join(retrieved_texts)


# ==========================================
# SENTENCE-LEVEL STREAM CHUNKER FOR TTS
# ==========================================
def stream_sentences(response_stream):
    """
    Buffers LLM token chunks and yields complete sentences bounded by Arabic/Latin
    punctuation. This allows the TTS engine to speak the first sentence naturally
    with proper intonation while the LLM generates the next sentence in parallel,
    instead of receiving a robotic stream of individual tokens.
    """
    buffer = ""
    for chunk in response_stream:
        buffer += chunk
        # Yield everything up to (and including) the last sentence boundary found
        while True:
            match = SENTENCE_BOUNDARY_PATTERN.search(buffer)
            if not match:
                break
            end_idx = match.end()
            sentence = buffer[:end_idx].strip()
            buffer = buffer[end_idx:]
            if sentence:
                yield sentence
    # Yield any remaining text that didn't end with punctuation
    if buffer.strip():
        yield buffer.strip()


# ==========================================
# FILLER TRIGGER
# ==========================================
def trigger_filler_audio():
    """
    Called immediately when retrieval is needed, before the slow RAG process starts.
    In a full voice pipeline this would play a pre-recorded audio clip such as
    "ثواني بس أشيك..." so the user hears something while the LLM retrieves context.
    The filler typically lasts 1-2 seconds — exactly the time RAG needs to complete —
    creating an illusion of zero latency.

    Replace the print statement below with your TTS filler playback call,
    e.g.: tts_client.play_filler("ثواني بس أشيك...")
    """
    print("[FILLER] ثواني بس أشيك...")


# ==========================================
# MAIN RESPONSE GENERATOR
# ==========================================
def generate_response(query: str):
    """
    Orchestrates the full Voice AI pipeline:
      1. Semantic cache lookup
      2. Fast intent routing (tiny model, ~50ms)
      3. Filler audio trigger (covers RAG latency)
      4. Graph + Vector retrieval
      5. LLM streaming with sentence-level chunking for TTS
      6. Cache result for future identical queries

    Yields complete sentences so the TTS layer can begin speaking
    the first sentence while the LLM is still generating the rest.
    """
    # 1. Cache check
    cached_response = cache.get(query)
    if cached_response:
        print("[Served from Dhakira Semantic Cache]")
        resp = cached_response.get("response", "")
        session_history.add_user_message(query)
        session_history.add_ai_message(resp)
        # Yield sentence by sentence even from cache for consistent TTS behaviour
        for sentence in re.split(SENTENCE_BOUNDARY_PATTERN, resp):
            s = sentence.strip()
            if s:
                yield s
        return

    # 2. Fast intent classification (tiny router model)
    collections_to_search = classify_intent(query)
    print(f"[Router] Searching: {collections_to_search if collections_to_search else 'CAG Only'}")

    # 3. Filler — play immediately so there's no dead-air during retrieval
    needs_retrieval = bool(collections_to_search)
    if needs_retrieval:
        trigger_filler_audio()

    # 4. Retrieval
    rag_context = ""

    if "staff_directory" in collections_to_search:
        graph_results = search_staff_graph(query)
        if graph_results:
            rag_context += f"=== STAFF GRAPH RESULTS ===\n{graph_results}\n\n"
        collections_to_search.remove("staff_directory")

    vector_results = retrieve_rag_context(query, collections_to_search)
    if vector_results:
        rag_context += f"=== VECTOR SEARCH RESULTS ===\n{vector_results}\n\n"

    # 5. LLM streaming with sentence-level TTS chunking
    response_stream = conversational_rag_chain.stream(
        {
            "query": query,
            "cag_context": SYSTEM_CAG_CONTEXT,
            "rag_context": rag_context
        },
        config={"configurable": {"session_id": "psut_session"}}
    )

    print("\nPSUT Bot: ", end="", flush=True)
    full_response = ""

    for sentence in stream_sentences(response_stream):
        print(sentence, end=" ", flush=True)
        full_response += sentence + " "
        # Each yielded sentence goes directly to the TTS engine in the voice pipeline
        yield sentence

    print("\n")

    # 6. Cache for future identical queries
    cache.put(query, {"response": full_response.strip()})


# ==========================================
# CLI ENTRY POINT (text fallback / testing)
# ==========================================
if __name__ == "__main__":
    print("\nPSUT Voice Agent LLM Module Initialized. Type 'exit' to quit.")
    print("-" * 50)

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in ['exit', 'quit']:
            break
        # Consume the generator — in the real voice pipeline each yielded
        # sentence would be sent to the TTS engine immediately
        for _ in generate_response(user_input):
            pass