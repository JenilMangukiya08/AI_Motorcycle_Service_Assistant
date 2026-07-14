import os
import datetime
import re
import pickle
import numpy as np
import faiss
from openai import OpenAI
from rapidfuzz import fuzz
from spellchecker import SpellChecker
from rapidfuzz import process
from sentence_transformers import SentenceTransformer
from PIL import Image
from rank_bm25 import BM25Okapi
from .data_loader import built_database
import time
from dotenv import load_dotenv

load_dotenv()
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

LOG_FILE = os.path.join(BASE_DIR, "retrieval_log.txt")


#---------------------------------------LOG RETRIVAL--------------------------------------------------------------------------------------------

def log_retrieval(entry: dict):
    """Append one human-readable block with query/chunk/image retrieval info."""
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = []
    lines.append("=" * 70)
    lines.append(f"TIME: {ts}")
    lines.append(f"QUERY (original):  {entry.get('original_query')}")
    lines.append(f"QUERY (corrected): {entry.get('query_corrected')}")
    lines.append(
        f"wants_image={entry.get('wants_image')}  "
        f"wants_tool={entry.get('wants_tool')}  "
        f"wants_steps_with_images={entry.get('wants_steps_with_images')}  "
        f"has_text_intent={entry.get('has_text_intent')}"
    )

    lines.append("\n-- Retrieved Text Chunks --")
    if entry.get("retrieved_chunks"):
        for c in entry["retrieved_chunks"]:
            lines.append(f"  PDF: {c['pdf']}  | Page: {c['page']}")
    else:
        lines.append("  (none)")

    lines.append("\n-- Image Candidates (all scored) --")
    if entry.get("all_image_scores"):
        for c in entry["all_image_scores"]:
            lines.append(f"  score={c['score']:.4f}  label='{c['label']}'  path={c['path']}")
    else:
        lines.append("  (none / wants_image=False)")

    lines.append("\n-- Final Images Used --")
    if entry.get("final_image_paths"):
        for p in entry["final_image_paths"]:
            lines.append(f"  {p}")
    else:
        lines.append("  (none)")

    lines.append(f"\nANSWER PREVIEW: {entry.get('answer_preview', '')}")
    lines.append("=" * 70 + "\n")

    try:
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
    except Exception as e:
        print(f"[rag_engine] Failed to write retrieval log: {e}")


#---------------------------------------------REQUIRED FILES-------------------------------------------------------- 

REQUIRED_FILES = [
    "vector_store.faiss",
    "chunks.pkl",
    "chunk_metadata.pkl",
    "image_data.pkl",
    "chunk_to_images.pkl",
]

if any(not os.path.exists(os.path.join(BASE_DIR, f)) for f in REQUIRED_FILES):
    print("[rag_engine] One or more database files missing — building database now...")
    built_database()
    print("[rag_engine] Database build complete.")


#-----------------------------------------------LOAD DATA FILES----------------------------------------------------


chunks          = pickle.load(open(os.path.join(BASE_DIR, "chunks.pkl"),          "rb"))
chunk_metadata  = pickle.load(open(os.path.join(BASE_DIR, "chunk_metadata.pkl"),  "rb"))
image_data      = pickle.load(open(os.path.join(BASE_DIR, "image_data.pkl"),      "rb"))
chunk_to_images = pickle.load(open(os.path.join(BASE_DIR, "chunk_to_images.pkl"), "rb"))

start = time.time()
index = faiss.read_index(os.path.join(BASE_DIR, "vector_store.faiss"))
end = time.time()

print(f"FAISS Load Time: {(end-start):.4f} sec")
tokenized_chunks = [
    chunk.lower().split()
    for chunk in chunks
]

bm25 = BM25Okapi(tokenized_chunks)
#--------------------------------------------------SPELL CHECKER---------------------------------------------------------


spell = SpellChecker()

TECHNICAL_TERMS = {
    "coolant",
    "radiator",
    "thermostat",
    "lambda",
    "ecu",
    "throttle",
    "injector",
    "swingarm",
    "crankshaft",
    "camshaft",
    "oxygen",
    "sensor",
    "engine",
    "clutch",
    "gearbox",
    "brake",
    "battery",
    "spark",
    "plug"
}

def correct_query(query):
    corrected = []

    for word in query.split():

        if len(word) <= 2:
            corrected.append(word)
            continue

        # Try matching against technical terms first
        match = process.extractOne(word.lower(), TECHNICAL_TERMS)

        if match and match[1] >= 80:
            corrected.append(match[0])
            continue

        # Normal spell correction
        correction = spell.correction(word)

        if correction:
            corrected.append(correction)
        else:
            corrected.append(word)

    return " ".join(corrected)

#----------------------------------LLM CLIENT---------------------------------------------------------------------------

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

client = OpenAI(
    api_key=GROQ_API_KEY,
    base_url="https://api.groq.com/openai/v1"
)
model = SentenceTransformer("BAAI/bge-base-en-v1.5")


#----------------------------------TOOL WORDS-------------------------------------------------------- 


TOOL_WORDS = {
    "tool"
}

#--------------------------Words that indicate the user wants an image at all-----------------------------------------

IMAGE_INTENT_WORDS = {
    "image", "photo", "picture", "diagram", "show", "look","looks",
    "visual", "view", "display", "figure", "illustration",
    "drawing", "schematic","looking","qr"
}

STEPS_WITH_IMAGE_PHRASES = (
    "steps with image", "steps with images", "step with image", "step with diagram",
    "step by step with", "steps and diagram", "steps and image", "steps and images",
    "procedure with diagram", "procedure with image", "with diagram", "with illustration",
    "show me the steps", "show steps", "give steps",
)

PROCEDURE_WORDS = {
    "how", "steps", "step", "procedure", "process", "dismantle", "dismantling",
    "assemble", "assembly", "remove", "removal", "install", "installation",
    "replace", "replacement", "adjust", "adjustment", "repair", "service",
    "disconnect", "connect", "detach", "attach", "loosen", "tighten", "unscrew",
    "disassemble", "reassembly", "overhaul", "refit", "fitting",
}

#------------------------------STEPS WITH IMAGES---------------------------------------------------------------------------

def detect_steps_with_images(query: str) -> bool:
    """True when the user wants a procedure broken into steps with manual diagrams."""
    text = query.lower()
    
    # Exclude informational queries that aren't procedures
    informational_phrases = ["how much", "how many", "how long", "what size", "what capacity", "what is the weight"]
    if any(inf in text for inf in informational_phrases):
        return False

    if any(phrase in text for phrase in STEPS_WITH_IMAGE_PHRASES):
        return True
    words = set(text.split())
    has_procedure = bool(words & PROCEDURE_WORDS) or "how to" in text
    has_visual = any(kw in text for kw in IMAGE_INTENT_WORDS) or bool(words & TOOL_WORDS)
    return has_procedure and has_visual


#---------------------------------------QUERY CLASSIFICATION-------------------------------------------------------


def classify_query(query: str):
    """
    Returns (wants_image: bool, wants_tool: bool)
    wants_image : user explicitly asked for a visual
    wants_tool  : user is asking about a specific tool

    Uses substring matching so plural/variant forms (e.g. "images", "pictures",
    "diagrams") are still detected even though the keyword sets store singular forms.
    """
    text  = query.lower()
    words = set(text.split())

    wants_image = (
        any(kw in text for kw in IMAGE_INTENT_WORDS)
        or bool(words & TOOL_WORDS)
        or detect_steps_with_images(query)
    )
    wants_tool  = bool(words & TOOL_WORDS)
    return wants_image, wants_tool



#----------------------------------IMAGES LINKED BY CHUNKS-------------------------------------------------------


def get_chunk_linked_image_indices(top_indices, selected_bike=None, wants_tool=False):
    """Collect instruction/tool images linked to retrieved text chunks, in page order."""
    linked = []
    seen = set()
    selected_bike_norm = selected_bike.strip().lower() if selected_bike else None

    for idx in top_indices:
        for img_idx in chunk_to_images.get(idx, []):
            if img_idx in seen:
                continue
            img = image_data[img_idx]
            if selected_bike_norm and img.get("bike", "").strip().lower() != selected_bike_norm:
                continue
            is_tool = "instr_images" not in img.get("path", "")
            if wants_tool and not is_tool:
                continue
            if not wants_tool and is_tool:
                continue
            seen.add(img_idx)
            linked.append(img_idx)

    linked.sort(key=lambda i: (image_data[i].get("page", 0), image_data[i].get("y0", 0)))
    return linked


#---------------------------------IMAGE ENTRY SCORING-----------------------------------------------------


def score_image_entry(img, topic_embedding, query_lower, answer_context):
    """Score one image against the query topic and optional answer context."""
    sem_score = float(np.dot(topic_embedding[0], img["embedding"]))
    img_label = img.get("label", "").lower()
    fuzzy_score = max(
        fuzz.token_set_ratio(query_lower, img_label) / 100,
        fuzz.token_set_ratio(answer_context, img_label) / 100 if answer_context else 0,
    )
    return sem_score * 0.6 + fuzzy_score * 0.4


#--------------------EXTRACT TOPIC KEYWORDS (strip intent/stop words)--------------------------------------

STOP_WORDS = IMAGE_INTENT_WORDS | TOOL_WORDS | {
    "me", "the", "a", "an", "of", "for", "in", "on",
    "is", "what", "how", "does", "do", "can", "i", "my",
    "get", "give", "find", "tell", "explain", "about", "please",
    "and", "or", "to", "its"
}

def extract_topic(query: str) -> str:
    words = [w for w in query.lower().split() if w not in STOP_WORDS and len(w) > 2]
    return " ".join(words)



#----------------------SPLIT ANSWER PERFECTLY IN POINTS-------------------------------------------------------------

def split_answer_lines(answer_text: str):
    """Split the LLM answer into clean actionable bullet lines.
    Drops: empty lines, 'couldn't find' lines, and section header lines
    (emoji headers like '🔧 Procedure', '📋 Specifications', '⚠️ Caution').
    Strips bullet markers (* - •) and numeric prefixes."""
    # Section headers the LLM uses — skip these, they're not steps/facts
    SECTION_HEADERS = {"procedure", "specifications", "caution", "warning", "note", "notes"}
    lines = []
    for raw in answer_text.splitlines():
        line = raw.strip()
        if not line:
            continue
        if "couldn't find" in line.lower():
            continue
        # Strip bullet markers first
        clean = re.sub(r'^[\*\-•]\s*', '', line)
        clean = re.sub(r'^\d+[.)]\s*', '', clean)
        
        # Skip if what remains (after stripping emoji) is just a section header word
        stripped_of_emoji = re.sub(r'[^\w\s]', '', clean).strip().lower()
        if stripped_of_emoji in SECTION_HEADERS:
            continue
        if clean:
            lines.append(clean)
    return lines


#---------------------------------------FOLLOW UP THINGS-----------------------------------------------------------


def has_real_topic(query):
    topic = extract_topic(query)
    return len(topic.split()) >= 1


#---------------------------------------METADATA FILTER-----------------------------------------------------------


def extract_metadata_filters(query):

    filters = {}

    page_match = re.search(
        r'page\s+(\d+)',
        query.lower()
    )

    if page_match:
        filters["page"] = int(
            page_match.group(1)
        )

    pdf_match = re.search(
        r'([\w\-]+\.pdf)',
        query.lower()
    )

    if pdf_match:
        filters["pdf"] = pdf_match.group(1)

    return filters

#---------------------------------------User query rewriting-----------------------------------------------------------

def expand_query(query):

    query_lower = query.lower()

    expansions = {
        "oil filter":
            "oil filter oil filter element oil filter holder assembly",

        "swing arm":
            "swing arm rear swing arm suspension arm",

        "brake":
            "brake brake system brake caliper brake disc",

        "clutch":
            "clutch clutch assembly clutch plates"
    }

    expanded_query = query

    for term, expansion in expansions.items():

        if term in query_lower:
            expanded_query += " " + expansion

    return expanded_query


def reformulate_query(query):

    query_lower = query.lower()

    replacements = {
        "remove": "removal procedure",
        "install": "installation procedure",
        "replace": "replacement procedure",
        "fix": "troubleshooting",
        "repair": "troubleshooting",
        "check": "inspection procedure"
    }

    for old, new in replacements.items():
        query_lower = query_lower.replace(old, new)

    return query_lower

def rewrite_query(query):

    query = reformulate_query(query)

    query = expand_query(query)

    return query
#---------------------------------------------------------------------------------------------------
#--------------------------------------MAIN QUERY FUNCTION------------------------------------------
#---------------------------------------------------------------------------------------------------




def ask_question(query, history=None, selected_bike=None):
    if history is None:
        history = []
    # Store original query before any preprocessing
    original_query = query
    
    # Preprocess/Correct spelling
    query_corrected = query.strip()
    
    # Initial classification using original query
    wants_image, wants_tool = classify_query(original_query)
    wants_steps_with_images = detect_steps_with_images(original_query)
    if wants_steps_with_images:
        wants_image = True

    # ── FOLLOW-UP DETECTION ───────────────────────────────────────────────────

    topic_from_history = ""
    text_lower = original_query.lower().strip()
    
    FOLLOWUP_IMAGE_PHRASES = {
        "show image", "show images", "show diagram", "show picture",
        "show photo", "show it", "show them", "show that", "its image",
        "it's image", "the image", "the picture", "the photo"
    }

    TEXT_FOLLOWUP_PHRASES = (
        "explain more", "tell me more", "more detail", "more details",
        "go on", "continue", "what about that", "what about it",
        "and then", "and after that", "next step", "next steps",
        "elaborate", "can you elaborate", "why is that", "why is it",
        "more info", "more information", "in detail", "in more detail","sub parts"
    )

    FOLLOWUP_PRONOUNS = {
        "it",
        "its",
        "that",
        "this",
        "them",
        "those",
        "these"
    }

    query_words = set(original_query.lower().split())
    contains_pronoun = bool(query_words & FOLLOWUP_PRONOUNS)

    is_followup_image = (
        bool(history)
        and wants_image
        and (contains_pronoun or not has_real_topic(original_query) or any(p in text_lower for p in FOLLOWUP_IMAGE_PHRASES))
    )
        
    if is_followup_image:
        # Walk history backwards to find last assistant reply
        for msg in reversed(history):
            if msg["role"] == "user":
                previous_question = msg["content"].strip()

                if previous_question.lower() != original_query.lower():
                    topic_from_history = previous_question
                    break

    is_followup_text = (
        bool(history)
        and not wants_image
        and (contains_pronoun or not has_real_topic(original_query) or any(p in text_lower for p in TEXT_FOLLOWUP_PHRASES))
    )
    
    search_text = rewrite_query(
    query_corrected
    )

    metadata_filters = extract_metadata_filters(
            original_query
        )
    last_user_query = ""
    last_assistant_answer = ""
    
    # We walk history to find the most recent context
    if is_followup_text or is_followup_image:
        for msg in reversed(history):
            if not last_assistant_answer and msg["role"] == "assistant" and msg.get("content", "").strip():
                last_assistant_answer = msg["content"].strip()
            elif not last_user_query and msg["role"] == "user" and msg.get("content", "").strip():
                last_user_query = msg["content"].strip()
            if last_assistant_answer and last_user_query:
                break

    # Define intent indicators
    TEXT_QUERY_WORDS = {
        "what", "how", "why", "when", "where", "which", "tell", "explain",
        "describe", "process", "procedure", "steps", "details", "brief",
        "elaborate", "info", "information", "more", "about"
    }
    PURE_IMAGE_INDICATORS = {
        "image", "photo", "picture", "diagram", "figure", "illustration", "show"
    }

    if is_followup_image and topic_from_history:
        # Combine history topic with current query for better retrieval
        search_text = f"{topic_from_history} {query_corrected}".strip()
    elif is_followup_text and (last_user_query or last_assistant_answer):
        followup_subject = extract_topic(
            last_user_query + " " + last_assistant_answer
        )
        search_text = f"{followup_subject} {query_corrected}".strip()
    else:
        search_text = query_corrected

    
    query_words_set = set(search_text.lower().split())
    has_text_intent = bool(query_words_set & TEXT_QUERY_WORDS) or not (query_words_set & PURE_IMAGE_INDICATORS)
    
    answer = ""
    retrieved_chunks = []
    
    

    print("\n========== FOLLOWUP DEBUG ==========")
    print("ORIGINAL =", original_query)
    print("CORRECTED =", query_corrected)
    print("SEARCH_TEXT =", search_text)
    print("IS_FOLLOWUP_TEXT =", is_followup_text)
    print("IS_FOLLOWUP_IMAGE =", is_followup_image)
    print("TOPIC =", topic_from_history)
    print("====================================\n")






#──────────────────────────────────────────────────────────────────────
    #─────────────────────Vector Search (RAG) for Context─────────────────────
    
    query_embedding = model.encode([search_text], normalize_embeddings=True).astype("float32")
    start = time.time()
    distance, indices = index.search(
    query_embedding,
    k=5
    )
    end = time.time()

    print(
        f"FAISS Query: {(end-start)*1000:.2f} ms"
    )

    #─────────────────────Hybride vector+bm25─────────────────────

    query_tokens = search_text.lower().split()

    bm25_scores = bm25.get_scores(
        query_tokens
    )

    bm25_top_indices = np.argsort(
        bm25_scores
    )[::-1][:20]

    faiss_results = {}

    for idx, score in zip(
        indices[0],
        distance[0]
    ):
        faiss_results[idx] = float(score)

    bm25_results = {}

    max_bm25 = max(bm25_scores)

    for idx in bm25_top_indices:

        bm25_results[idx] = (
            bm25_scores[idx] / max_bm25
            if max_bm25 > 0 else 0
        )

    hybrid_scores = {}

    for idx, score in faiss_results.items():

        hybrid_scores[idx] = score * 0.5

    for idx, score in bm25_results.items():

        hybrid_scores[idx] = (
            hybrid_scores.get(idx, 0)
            + score * 0.5
        )

    hybrid_ranked = sorted(
        hybrid_scores.items(),
        key=lambda x: x[1],
        reverse=True
        )  
    
    hybrid_ranked = hybrid_ranked[:50]


    #─────────────────────filter of metadata─────────────────────
    filtered_results = []

    for idx, score in hybrid_ranked:

        meta = chunk_metadata[idx]

        # Bike filter
        if selected_bike:

            if (
                meta.get("bike", "").lower()
                != selected_bike.lower()
            ):
                continue

        # ─────────────────────Page filter─────────────────────
        if "page" in metadata_filters:

            if (
                meta.get("page")
                != metadata_filters["page"]
            ):
                continue

        if "pdf" in metadata_filters:

            if (
                meta["pdf"].lower()
                != metadata_filters["pdf"]
            ):
                continue

        filtered_results.append(
            (idx, score)
        )

    topic_keywords = [w for w in search_text.lower().split() if len(w) > 3]
    reranked = []
    for idx, score in filtered_results:
        chunk_lower = chunks[idx].lower()
        fuzzy_score = fuzz.token_set_ratio(search_text.lower(), chunk_lower) / 100
        
        # ─────────────────────Keyword match bonus (boost chunks that actually mention the topic)─────────────────────
        matches = sum(1 for kw in topic_keywords if kw in chunk_lower)
        keyword_bonus = (matches / len(topic_keywords)) * 0.4 if topic_keywords else 0
        
        content_bonus = 0
        if len(chunks[idx].split()) > 80:
            content_bonus += 0.1
        if "•" in chunks[idx]:
            content_bonus += 0.1

        combined = float(score) * 0.7 + fuzzy_score * 0.1 + keyword_bonus + content_bonus
        reranked.append((combined, idx))
    reranked.sort(key=lambda x: x[0], reverse=True)
    if selected_bike:
        selected_bike_norm = selected_bike.strip().lower()
        reranked = [(s, idx) for s, idx in reranked if chunk_metadata[idx].get("bike", "").strip().lower() == selected_bike_norm]

    top_indices = [idx for _, idx in reranked[:8]]

    print("\n===== TOP CHUNKS =====")

    for idx in top_indices:
        print(
            f"PDF={chunk_metadata[idx]['pdf']} "
            f"PAGE={chunk_metadata[idx]['page']} "
            f"TYPE={chunk_metadata[idx]['type']}"
        )

        print(chunks[idx][:300])
        print("-" * 80)

    context_parts = []
    valid_indices = []
    for idx in top_indices:
        
        chunk_text = chunks[idx]       
        toc_lines = re.findall(
            r'^\d+\.\d+\.\d+\s+[A-Z ]+$',
            chunk_text,
            flags=re.MULTILINE
        )
        section_lines = re.findall(
            r'^\d+\.\s+[A-Z ]+$',
            chunk_text,
            flags=re.MULTILINE
        )

        if len(toc_lines) >= 3 or len(section_lines) >= 3:
            continue
        valid_indices.append(idx)

        

    for idx in valid_indices:
        meta = chunk_metadata[idx]
        pdf_name = os.path.basename(meta.get("pdf", "Unknown"))
        context_parts.append(f"\n[SOURCE: PDF:{pdf_name} | PAGE: {meta.get('page')}]\n\n{chunks[idx]}\n")
        retrieved_chunks.append({
            "bike": meta.get("bike"),
            "pdf": pdf_name,
            "page": meta.get("page"),
            "content": chunks[idx]
        })

    
    context = "\n\n".join(context_parts)

    conversation = ""

    for msg in history[-6:]:   # last 6 messages
        conversation += f"{msg['role']}: {msg['content']}\n"
    
    # 2. LLM Answer (only if user wants text)
    if has_text_intent:
        steps_with_images_note = ""
        if wants_steps_with_images:
            steps_with_images_note = """
CRITICAL: The user specifically asked for "STEPS WITH IMAGES".
- IGNORE Rule 3 (Hierarchy) for this specific request.
- Do NOT use A/B/C or i/ii/iii numbering.
- Instead, provide a simple flat numbered list (1., 2., 3. ...) of actionable steps.
- Each numbered line MUST be a single step from the service manual.
- DO NOT add empty lines between steps.
- Do not mix steps from different procedures.
- Focus on the single most relevant procedure.
"""
        followup_note = ""
        resolved_question = query_corrected
        if is_followup_text and (last_user_query or last_assistant_answer):
            followup_note = f"""
The user's question is a FOLLOW-UP to the previous exchange below — it refers
back to that topic ("more", "that", "it", "continue", etc.) rather than
introducing a new subject. Answer as a continuation of the same topic, using
the previous question/answer only to understand what is being referred to.
Previous user question: {last_user_query}
Previous answer given: {last_assistant_answer[:600]}
"""
            resolved_question = f"{query_corrected} (regarding: {last_user_query})" if last_user_query else query_corrected

        prompt = f"""
You are a motorcycle service manual assistant.
Rules:
1. Use ONLY the provided context. If the context contains information about DIFFERENT components or procedures than the one requested, DISCARD them entirely. Focus only on the exact component or task asked for.
2. If NONE of the context contains information relevant to the specific question, reply with EXACTLY this single line: "I couldn't find information for this." Do not add citations or extra text.
3. If the context DOES contain relevant information, answer using ONLY the relevant parts. Ignore and do not mention any context sources that are irrelevant.
4. Context marked TYPE: table may contain raw numbers or specs. extract any values that answer the question.
5. Cite source: [PDF_NAME | Page PAGE_NUMBER] after each fact, only for facts you actually used.
6. Format the answer in a technician-friendly way.

Use:

• steps

📋 Specifications
• value

⚠️ Caution
• warning

Only include sections that exist in the context.

Keep explanations simple and easy to understand.
7. Use a single bullet point per line. DO NOT add empty lines between points.
8. If the question is a direct technical query, provide the data points in a list.
9. Never merge steps from multiple procedures. 
10. If other related procedures are found in the retrieved context:
   - Do NOT include their steps.
   - Only mention it under "Related Sections".
11. Preserve the original order and numbering of the selected procedure.
12. Leave one blank line only between major steps or major sections.
Do not insert blank lines between closely related bullet points belonging to the same subheading.

Do NOT convert subheadings into numbered steps.
Keep the document hierarchy.
Only number major procedures/problems.

For example:
1. Loss of Compression
Symptoms:
• ...
Troubleshooting:
• ...
2. Excessive Blow-By
Symptoms:
• ...
Troubleshooting:
• ...
Do not create:
1. Loss of Compression
2. Symptoms
3. Troubleshooting

13. Format all major headings in Markdown bold.
Examples:
**Introduction**
**Specifications**
**Safety Measures During Assembly**
**Troubleshooting**

14. When a procedure contains numbered sections, use this hierarchy:
1. Main Topic
   **Symptoms**
   • symptom 1
   • symptom 2
   **Troubleshooting**
   • step 1
   • step 2
2. Next Main Topic
   **Symptoms**
   • symptom 1
   **Troubleshooting**
   • step 1

15. NEVER convert subheadings into numbered steps.\
Correct:
1. Loss of Compression
   **Symptoms**
   • Reduced engine power
   **Troubleshooting**
   • Perform compression test
Wrong:
1. Loss of Compression
2. Symptoms
3. Troubleshooting

16. Preserve document hierarchy exactly as it appears in the manual.
Major procedure/problem:
1.
2.
3.
Sub-sections:
**Symptoms**
**Troubleshooting**
**Inspection**
**Removal**
**Installation**
**Notes**
**Caution**
must remain as subheadings and never become numbered items.

17. Leave one blank line:
- Between major numbered sections.
- Between a subheading and the next subheading.
Example:
1. Loss of Compression
**Symptoms**
• Reduced engine power
• Excessive smoke
**Troubleshooting**
• Perform compression test
• Inspect piston rings
2. Excessive Blow-By
**Symptoms**
• Oil escaping into crankcase
**Troubleshooting**
• Inspect piston rings

18. If the manual contains lettered sub-sections, preserve them.
Example:
1. Loss of Compression
**Symptoms**
a. Reduced engine power
b. Excessive smoke
**Troubleshooting**
a. Perform compression test
b. Inspect piston rings
Do not convert a., b., c. into numbered steps.

19. Preserve bullet hierarchy.
Example:
**Troubleshooting**
• Perform compression test
  - Inspect piston rings
  - Inspect cylinder walls
• Replace damaged components

20. For procedures, keep step numbering exactly as in the manual.
Do not renumber or flatten nested steps.

21. If a bullet line ends with ":" and is followed by one or more bullets,
treat that line as a subheading.

Example:

Inspect Chain Tension:
• Ensure correct tension.
• Check for wear.

Must become:

**Inspect Chain Tension**
• Ensure correct tension.
• Check for wear.

22. Do not render subheadings as bullets.

Wrong:
• Inspect Chain Tension:
  - Ensure correct tension

Correct:
**Inspect Chain Tension**
• Ensure correct tension

23. When a section contains multiple troubleshooting cases,
format each case as a bold subheading followed by its bullet points.

Example:

**Troubleshooting**

**Inspect Chain Tension**
• Ensure correct tension.

**Cleaning & Lubrication**
• Clean chain regularly.

**Chain Noise**
• Check for clicking or grinding noises.

24. If a line is written entirely in uppercase or ends with a colon and introduces a list,
display it as a bold heading.

Example:

DRIVE CHAIN NEEDS TO BE INSPECTED FOR:

becomes

**DRIVE CHAIN NEEDS TO BE INSPECTED FOR**
• Damaged Rollers
• Loose Pins
• Dry or Rusted Links
{steps_with_images_note}
{followup_note}

Previous Conversation:
{conversation}
Context:
{context}
Question:
{resolved_question}
Answer:
"""
        try:
            response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2
            )
            answer = response.choices[0].message.content.strip()
            answer = answer.replace("###", "\n")
            answer = answer.replace("##", "\n")
            answer = re.sub(
                r'•\s*(.+?):\s*\n\s*-\s*',
                lambda m: f'\n**{m.group(1).strip()}**\n• ',
                answer
            )

            lines = answer.splitlines()
            # ----------------------------
            # STEP SPLITTING
            # ----------------------------

            steps = []

            for line in answer.split("\n"):
                line = line.strip()

                if not line:
                    continue

                steps.append(line)

            new_lines = []
            step_no = 1

            best_chunk = top_indices[0]

            candidate_images = []

            for chunk_id in top_indices[:2]:

                if chunk_id in chunk_to_images:

                    for img_idx in chunk_to_images[chunk_id]:

                        candidate_images.append(
                            image_data[img_idx]
                        )
            answer_blocks = []
            if wants_steps_with_images:
                for step in steps:
                    step_embedding = model.encode(
                        [step],
                        normalize_embeddings=True
                    ).astype("float32")[0]

                    best_score = 0
                    best_image = None

                    for img in candidate_images:
                        if "embedding" not in img:
                            continue

                        score = np.dot(
                            step_embedding,
                            img["embedding"]
                        )

                        if score > best_score:
                            best_score = score
                            best_image = img

                    images = []
                    if best_image and best_score > 0.65:
                        images.append(best_image["path"])

                    answer_blocks.append({
                        "text": step,
                        "images": images
                    })
            else:
                # If not steps-with-images, we treat the whole thing as a single answer string
                # We'll rely on chat.answer in the frontend
                pass
#------------------------------------------------------------------------------------------------------------------------
            # Final cleanup of the answer string (preserving indentation)
            answer = "\n".join(answer.splitlines())
                        

            # Safety net: if the model still produced one "couldn't find" line
            # per source instead of a single message, collapse it to one line.
            lines = [l.strip() for l in answer.splitlines() if l.strip()]
            if lines and all("couldn't find" in l.lower() for l in lines):
                answer = "I couldn't find information for this."
        except Exception as e:
            answer = f"Error: {e}"

    # 3. Image Search (only if user wants an image)
    image_candidates = []  # list of dicts: {"path", "score", "label"}
    all_image_scores = []  # every scored image, for logging/debugging
    chunk_linked_indices = []

    # Hard guard: never show images unless a bike is selected. Without this,
    # every filter below is a no-op (they're all gated on "if selected_bike_norm
    # and ..."), so a missing/empty selected_bike silently let images from
    # every bike in the database compete for the same query.
    no_bike_selected = not (selected_bike and selected_bike.strip())

    if wants_image and no_bike_selected:
        wants_image = False
        no_bike_warning = "Please select a bike model to see related images."
    else:
        no_bike_warning = ""

    if wants_image:
        topic = extract_topic(original_query)
        # If this is a follow-up image request with no topic, use the
        # topic extracted from the last assistant answer in history.
        if not topic or (is_followup_image and topic_from_history):
            topic = topic_from_history
        query_lower = topic if topic else original_query.lower()

        # Use the retrieved/generated answer text to help match the right image,
        # so images are selected based on what was actually found, not just the raw query.
        answer_context = ""
        if answer and "couldn't find information" not in answer.lower():
            answer_context = extract_topic(answer)

        combined_text = f"{query_lower} {answer_context}".strip()
        topic_embedding = model.encode([combined_text], normalize_embeddings=True).astype("float32")

        selected_bike_norm = selected_bike.strip().lower() if selected_bike else None
        chunk_linked_indices = get_chunk_linked_image_indices(
            top_indices, selected_bike=selected_bike, wants_tool=wants_tool
        )
        chunk_linked_paths = {image_data[i]["path"] for i in chunk_linked_indices}

        
        for img in image_data:
            is_table = "instr_images" not in img.get("path", "")
            if wants_tool and not is_table:
                continue
            if not wants_tool and is_table:
                continue
            if selected_bike_norm and img.get("bike", "").strip().lower() != selected_bike_norm:
                continue

            final_score = score_image_entry(img, topic_embedding, query_lower, answer_context)
            if img["path"] in chunk_linked_paths:
                final_score *= 1.2

            all_image_scores.append({
                "path": img["path"],
                "score": final_score,
                "label": img.get("label", "").lower(),
                "from_chunk": img["path"] in chunk_linked_paths,
            })

        all_image_scores.sort(key=lambda x: x["score"], reverse=True)
        best_score = all_image_scores[0]["score"] if all_image_scores else 0

        # Prefer images from the same manual pages as retrieved procedure text
        if chunk_linked_indices:
            chunk_candidates = [
                c for c in all_image_scores
                if c["from_chunk"]
            ]
            if chunk_candidates:
                image_candidates = chunk_candidates[:8]
            elif wants_steps_with_images:
                image_candidates = [
                    {
                        "path": image_data[i]["path"],
                        "score": 0.5,
                        "label": image_data[i].get("label", "").lower(),
                        "from_chunk": True,
                    }
                    for i in chunk_linked_indices[:8]
                ]

        if not image_candidates:
            image_candidates = [c for c in all_image_scores if c["score"] >= best_score * 0.90][:5]
            if best_score < 0.45:
                image_candidates = []

        # Fallback: if nothing cleared the threshold, still show the closest matches
        if not image_candidates and all_image_scores:
            image_candidates = all_image_scores[:3]

    # 4. Build final response based on what the user asked for
    final_answer = answer if has_text_intent else ""
    # answer_lines = clean step/fact lines used for image interleaving only.
    # final_answer keeps the original LLM-formatted text (with emoji headers etc.)
    answer_lines = split_answer_lines(answer) if has_text_intent else []

    filtered_lines = []

    for line in answer_lines:

        line = line.strip()

        if not line:
            continue

        # Skip section headings
        if line.startswith("📋"):
            continue

        if line.startswith("⚠️"):
            continue

        if "Related Sections" in line:
            continue

        if len(line) < 10:
            continue

        filtered_lines.append(line)

    answer_lines = filtered_lines

    # For follow-up image requests, pull the previous answer's lines so we
    # can still interleave images against the actual steps (no new LLM call).
    if is_followup_image and not answer_lines:
        for msg in reversed(history):
            if msg["role"] == "assistant" and msg.get("content", "").strip():
                answer_lines = split_answer_lines(msg["content"])
                break

    final_images = []
    answer_blocks = []
    other_images = []

    if wants_image and answer_lines:
        # Interleave: for each step/line, attach the best-matching unused image.
        remaining = list(image_candidates)
        line_embeddings = model.encode(answer_lines, normalize_embeddings=True).astype("float32")
        ordered_page_images = [image_data[i]["path"] for i in chunk_linked_indices]
        ordered_pool = [p for p in ordered_page_images if any(c["path"] == p for c in remaining)]
        used_paths = set()

        current_step = 1
        last_pool_idx = -1
        for line_text, line_emb in zip(answer_lines, line_embeddings):
            # DETECT & CLEAN HEADINGS:
            clean_text = line_text.strip()
            lower_clean = clean_text.lower()
            bold_match = bool(re.match(r'^\*+.*\*+$', clean_text))
            is_keyword_header = any(kw in lower_clean for kw in ["procedure", "removal", "installation", "steps", "instructions"])
            is_short = len(clean_text.split()) < 12
            
            is_heading = (
                bold_match 
                or clean_text.isupper() 
                or clean_text.endswith(":")
                or (is_keyword_header and is_short)
                or any(h == lower_clean for h in ["symptoms", "troubleshooting", "inspection", "removal", "installation", "note", "caution", "specifications", "procedure", "steps"])
            )

            if is_heading:
                # Clean up broken/mixed markers like *Removal** -> **Removal**
                stripped = re.sub(r'^\*+|\*+$', '', clean_text)
                clean_text = f"**{stripped}**"
                step_val = None
            else:
                step_val = current_step
                current_step += 1

            block_images = []
            best_score = 0.0

            # ── IMPROVED IMAGE ORDERING ──────────────────────────────────────
            # If we are in "steps with images" mode, we should follow the manual's sequence
            if wants_steps_with_images and ordered_pool and not is_heading:
                # Sequential look-ahead: we search for the next matching image in the PDF order
                best_pool_idx = None
                for i in range(last_pool_idx + 1, min(last_pool_idx + 4, len(ordered_pool))):
                    path = ordered_pool[i]
                    if path in used_paths: continue
                    
                    img_entry = next((im for im in image_data if im["path"] == path), None)
                    if not img_entry: continue
                    
                    sem_score = float(np.dot(line_emb, img_entry["embedding"]))
                    fuzzy_score = fuzz.token_set_ratio(clean_text.lower(), img_entry.get("label", "")) / 100
                    score = sem_score * 0.6 + fuzzy_score * 0.4
                    
                    if score > 0.18: # Reasonable match found in sequence
                        best_score = score
                        best_pool_idx = i
                        break 

                if best_pool_idx is not None:
                    path = ordered_pool[best_pool_idx]
                    block_images.append(path)
                    used_paths.add(path)
                    last_pool_idx = best_pool_idx
                    # Remove from global remaining pool so it's not reused elsewhere
                    remaining = [c for c in remaining if c["path"] != path]

            # Fallback for general image requests (non-sequential)
            elif not is_heading:
                best_idx = None
                for i, cand in enumerate(remaining):
                    img_entry = next((im for im in image_data if im["path"] == cand["path"]), None)
                    if not img_entry: continue
                    sem_score = float(np.dot(line_emb, img_entry["embedding"]))
                    fuzzy_score = fuzz.token_set_ratio(clean_text.lower(), cand["label"]) / 100
                    step_score = sem_score * 0.6 + fuzzy_score * 0.4
                    if cand.get("from_chunk"): step_score *= 1.15
                    if step_score > best_score:
                        best_score = step_score
                        best_idx = i

                match_threshold = 0.20
                if best_idx is not None and best_score > match_threshold:
                    chosen = remaining.pop(best_idx)
                    block_images.append(chosen["path"])
                    used_paths.add(chosen["path"])

            answer_blocks.append({
                "text": clean_text,
                "images": block_images,
                "step": step_val,
                "match_score": best_score
            })


        # For steps-with-images queries, fill unmatched steps from page-ordered manual diagrams
        if wants_steps_with_images and ordered_pool:

            for block in answer_blocks:

                if block["images"]:
                    continue

                if block.get("match_score", 0) < 0.10:
                    continue

        # Any unused candidates with a decent score go into "other images"
        other_images = [c["path"] for c in remaining if c["score"] > 0.18][:6]

        # If no step ended up with an image AND no other_images, fall back to
        # showing the top overall candidates so the user still sees something.
        if not any(b["images"] for b in answer_blocks) and not other_images and image_candidates:
            other_images = [c["path"] for c in image_candidates[:4]]



    elif wants_image:
        # Image-only mode (no text), or text query produced no usable lines:
        # show a flat gallery of the top images.
        final_images = [c["path"] for c in image_candidates[:5]]

    # Log retrieval details for debugging/analysis
    final_image_paths = list(final_images)
    for b in answer_blocks:
        final_image_paths.extend(b["images"])
    final_image_paths.extend(other_images)

    log_retrieval({
            "original_query": original_query,
            "query_corrected": query_corrected,
            "is_followup_text": is_followup_text,
            "is_followup_image": is_followup_image,
            "topic_from_history": topic_from_history,
            "wants_image": wants_image,
            "wants_tool": wants_tool,
            "wants_steps_with_images": wants_steps_with_images,
            "has_text_intent": has_text_intent,
            "retrieved_chunks": [
            {
                "bike": c.get("bike", ""),
                "pdf": c.get("pdf", ""),
                "page": c.get("page", ""),
                "content": c.get("content", "")[:500]
            }
            for c in retrieved_chunks
            ],
            "all_image_scores": all_image_scores[:15],
            "final_image_paths": final_image_paths,
            "answer_blocks": answer_blocks,
            "answer_preview": final_answer[:1000] if final_answer else "",
        })
    

    print("\nQUESTION =", original_query)
    print("IS_FOLLOWUP_TEXT =", is_followup_text)
    print("IS_FOLLOWUP_IMAGE =", is_followup_image)
    print("TOPIC_FROM_HISTORY =", topic_from_history)

    if not (final_answer or final_images or answer_blocks):
        return {
            "answer": "I couldn't find information for this.",
            "images": [],
            "warning": no_bike_warning,
        }
    
    return {
        "answer": final_answer,
        "images": final_images,
        "answer_blocks": answer_blocks,
        "other_images": other_images,
        "retrieved_chunks": retrieved_chunks,
        "warning": no_bike_warning,
    }



if __name__ == "__main__":
    tests = [
        "show me the magneto holder tool",
        "show throttle operation diagram",
        "how to check throttle operation and show me the diagram",
    ]
    for q in tests:
        print(f"\n{'='*60}\nQ: {q}")
        r = ask_question(q)
        print(f"Answer: {r['answer'][:100] if r['answer'] else '(none)'}")
        print(f"Image:  {r['images']}")