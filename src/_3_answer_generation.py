#!/usr/bin/env python
# coding: utf-8

# # Answering & Evaluation (Chroma → Ollama)
# 
# **Goal:** Take retrieved legal articles (from Chroma) and generate a **grounded, structured answer** using a *local* Ollama model (e.g., `llama3:8b`).
# 
# **What we’ll do:**
# 1) Load the Chroma collection
# 2) Retrieve top-K relevant articles (uses the helpers from Notebook 2)
# 3) Build a clean context block with citations like `[OR Art. 269d – OR.pdf]`
# 4) Call **Ollama HTTP API** locally to generate the answer
# 5) Run a small evaluation set of typical user questions
# 

# Imports & Paths

# In[1]:


import os, json, requests
from pathlib import Path
from typing import List, Tuple

import chromadb, logging
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

try:
    BASE_DIR = Path(__file__).resolve().parent
except NameError:
    # Jupyter fallback
    BASE_DIR = Path(os.getcwd())

CHROMA_DIR = (BASE_DIR.parent / "store").resolve()
CHROMA_COLLECTION = "swiss_private_rental_law"

EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Retrieval knobs
TOP_K  = 5
PRE_K  = 20
MAX_CTX_CHARS = 8000

# Ollama local settings
OLLAMA_HOST  = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3:8b")

logging.getLogger("chromadb").setLevel(logging.DEBUG)


# We verify:
# - Chroma store exists and is readable
# - The collection is present
# - Ollama is reachable and model is available
# 

# Chroma & Embedder helpers (same logic as indexing_and_retrieval)

# In[2]:


# Disable analytics/telemetry
os.environ["CHROMA_TELEMETRY_ENABLED"] = "false"
os.environ["POSTHOG_DISABLED"] = "true"

def get_client():
    return chromadb.PersistentClient(path=str(CHROMA_DIR))

def get_collection(name=CHROMA_COLLECTION):
    client = get_client()
    return client.get_collection(name)

_embedder = None
def embedder():
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer(EMBED_MODEL_NAME)
    return _embedder


# Check collection & doc count

# In[3]:


try:
    col = get_collection()
    print("Collection:", CHROMA_COLLECTION, "| count:", col.count())
except Exception as e:
    raise SystemExit(f"❌ Could not open Chroma collection. Did you run Notebook 2? Error: {e}")


# Check Ollama is running

# In[4]:


def check_ollama(host=OLLAMA_HOST, model=OLLAMA_MODEL):
    try:
        r = requests.get(host, timeout=5)
        ok_base = r.status_code in (200, 404)  # / returns 404 often, that's fine if host reachable
    except Exception as e:
        return False, f"Host not reachable: {e}"

    try:
        # quick no-op generate to ensure model is present
        test = requests.post(
            f"{host}/api/generate",
            json={"model": model, "prompt": "OK", "stream": False},
            timeout=20
        )
        ok_model = (test.status_code == 200)
        return ok_model, None if ok_model else f"Model call failed: {test.text[:200]}"
    except Exception as e:
        return False, f"Model not available: {e}"

ok, err = check_ollama()
print("Ollama ready:", ok, "| model:", OLLAMA_MODEL)
if not ok:
    print("Hint: Run `ollama pull llama3:8b` and ensure Ollama is running.")
    if err: print("Details:", err)


# We reuse a lightweight retrieval pipeline:
# - Embed the query
# - Query Chroma (optionally prefetch `PRE_K` and re-rank)
# - Format a **compact context** with clear citations
# 

# Retrieve & (optional) re-rank + pack context

# In[5]:


def retrieve(query: str, k: int = TOP_K, k_pre: int = PRE_K, collection_name: str = CHROMA_COLLECTION):
    col = get_collection(collection_name)
    q_emb = embedder().encode([query], normalize_embeddings=True).tolist()[0]
    res = col.query(query_embeddings=[q_emb], n_results=k_pre, include=['documents','metadatas','distances'])

    docs  = res.get('documents', [[]])[0]
    metas = res.get('metadatas', [[]])[0]
    dists = res.get('distances', [[]])[0]
    prelim = list(zip(docs, metas, dists))

    # Optional: cross-encoder rerank (commented out; requires transformers/torch)
    try:
        from sentence_transformers import CrossEncoder
        reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        scores = reranker.predict([(query, d) for d,_,_ in prelim]).tolist()
        prelim = [p for p,_ in sorted(zip(prelim, scores), key=lambda x: x[1], reverse=True)]
    except Exception:
        prelim = sorted(prelim, key=lambda x: x[2])  # distance ascending

    return prelim[:k]

def pack_context(retrieved, max_chars=MAX_CTX_CHARS, per_source_cap=3):
    """
    Build the context string and an id_map so we can later map used IDs back to metadata.
    Returns: context_text, id_map (list of dicts with id, law, article, source)
    """
    ctx, total, seen = [], 0, {}

    for doc, meta, dist in retrieved:
        key = (meta.get("law"), meta.get("article"))
        seen[key] = seen.get(key, 0) + 1
        if seen[key] > per_source_cap:
            continue

        stamp = f"[{meta.get('law','?')} {meta.get('title','?')} – {meta.get('source')}]"
        block = f"{stamp}\n{doc.strip()}\n\n"
        if total + len(block) > max_chars:
            break

        ctx.append(block)
        total += len(block)
    return "".join(ctx)


# ### Prompt design
# 
# We force a strict structure for answers and **forbid** using anything outside the provided context.
# 
# **Format required:**
# 1) One-sentence answer.
# 2) Numbered steps/options (say if they apply to Tenant or Landlord).
# 3) Forms required (exact names if present).
# 4) Articles to read next (e.g., Art. 269 OR; Art. 19 VMWG).
# 
# Then **References** as `[LAW Art.X – filename]`.
# 

# In[6]:


PROMPT = """You are a Swiss rental-law assistant.
Answer ONLY from the CONTEXT. If insufficient, say so.
Do NOT refer to yourself, your role, or your identity in the answer.
Start directly with the content requested (no introductions).

FORMAT STRICTLY:
1) "**Antwort**:" One concise sentence.
2) "**Schritte/Optionen**:" NUMBERED points tailored to the given Perspective.
3) "**Formulare**:" bullet list of exact official form names if present in CONTEXT, otherwise write "Keine für diesen Fall gefunden."
5) "**Referenzen**:" bullet list of distinct sources from CONTEXT as [law title – filename].

CONTEXT:
{context}

QUESTION:
{question}
"""

def answer_with_ollama(question: str, perspective: str, k=TOP_K, model=OLLAMA_MODEL, host=OLLAMA_HOST):
    """
    Query Ollama with retrieved context and return:
    (generated_answer, used_references, hits)
    """

    # 1. Retrieve documents
    hits = retrieve(question, k=k)
    context = pack_context(hits, max_chars=MAX_CTX_CHARS)

    # 2. Generate answer
    prompt = PROMPT.format(
        context=context,
        question=f"[Perspective: {perspective}] {question}"
    )

    # 3) Define the structured output schema
    schema = {
        "type": "object",
        "properties": {
            "answer": {"type": "string"},
            "steps": {
                "type": "array",
                "items": {"type": "string", "maxLength": 180},
                "minItems": 2,
                "maxItems": 8,
                "uniqueItems": False
            },
            "forms": {
                "type": "array",
                "items": {"type": "string"},
                "minItems": 0,
                "maxItems": 10
            },
            "references": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "law":     {"type": "string"},
                        "title": {"type": "string"},
                        "source":  {"type": "string"}
                    },
                    "required": ["law", "title", "source"]
                }
            }
        },
        "required": ["answer", "references"]
    }

    # 4) Call Ollama /api/chat with schema-enforced output
    r = requests.post(
        f"{host}/api/chat",
        json={
            "model": model,
            "messages": [
                {"role": "system", "content": prompt},
                {"role": "user", "content":
                    "Beantworte die Frage gemäss obigen Vorgaben. "
                    "Fülle die Felder des JSON-Schemas aus. "
                    "Sprache: Deutsch (Schweiz), Du-Form. "
                    "Für 'steps' gilt: Gib 2–8 kurze Einträge zurück, "
                    "jeder Eintrag genau EIN Schritt, EINE Zeile, KEINE Nummerierung oder Zeilenumbrüche. "
                    "Für 'forms': gib die genauen offiziellen Bezeichnungen aus dem CONTEXT zurück (leer, wenn keine). "
                    "Gib NUR JSON zurück."
                },
            ],
            "stream": False,
            "format": schema,
            "options": {"temperature": 0}
        },
        timeout=120
    )

    if r.status_code != 200:
        return f"[Ollama error {r.status_code}]: {r.text}", hits

    # 5) Parse the JSON content returned by /api/chat
    data = r.json()
    content = data.get("message", {}).get("content", "")
    try:
        parsed = json.loads(content) if isinstance(content, str) else content
    except Exception as e:
        return f"[Parse error]: {e}\nRaw: {content}", [], hits

    answer_text = (parsed.get("answer") or "").strip()
    steps = parsed.get("steps")
    if isinstance(steps, str):
        steps = [s.strip() for s in steps.split("\n") if s.strip()]
    steps = steps or []

    forms = parsed.get("forms")
    if isinstance(forms, str):
        forms = [f.strip() for f in forms.split("\n") if f.strip()]
    forms = forms or []

    references = parsed.get("references") or []

    return answer_text, steps, forms, references, hits


# Try a realistic query and inspect the sources retrieved.
# 

# Single question test

# In[7]:


def single_question_test():
    q = "Wie fechte ich eine Mietzinserhöhung an? Welches Formular ist nötig?"
    ans, hits = answer_with_ollama(q, perspective="Tenant", k=6)
    print("=== ANSWER ===\n", ans, "\n")
    print("=== SOURCES ===")
    for _, m, _ in hits:
        print(f"- {m.get('law')} Art.{m.get('article')} – {m.get('source')}")


# In[8]:


#single_question_test()


# We’ll run several canonical questions to check:
# - Structure & clarity of answers
# - That references point to the right law/articles
# - That forms are extracted when present (from VMWG, OR)
# 

# Batch evaluation

# In[9]:


def batch_evaluation():
    eval_questions = [
        ("Wie fechte ich eine Mietzinserhöhung an? Welches Formular ist nötig?", "Tenant", "English"),
        ("Welche Rechte habe ich bei Mängeln in der Wohnung?", "Tenant", "German"),
        ("Darf der Vermieter während laufendem Schlichtungsverfahren kündigen?", "Landlord", "English"),
        ("Wann sind Mietzinserhöhungen wegen energetischer Verbesserungen zulässig?", "Landlord", "German"),
    ]

    for q, perspective in eval_questions:
        print("\n" + "="*150)
        print("Q:", q, "| Perspective:", perspective)
        print("="*150)
        ans, hits = answer_with_ollama(q, perspective=perspective, k=6)
        print("\n--- ANSWER ---\n", ans[:2000])  # trim for display
        print("\n--- REFERENCES ---")
        refs = {(m.get('law'), m.get('article'), m.get('source')) for _, m, _ in hits}
        for law, art, src in refs:
            print(f"[{law} Art.{art} – {src}]")


# In[10]:


#batch_evaluation()


# ### Common issues & fixes
# 
# - **`Collection … count: 0`**  
#   Run Notebook 2 (Indexing) first to build the Chroma collection.
# 
# - **Ollama error / not reachable**  
#   Ensure Ollama is running and the model is available:  
#   `ollama serve` (if needed), then `ollama pull llama3:8b`.
# 
# - **Answers not following format**  
#   Tighten the prompt (you can add: “If you deviate from the format, respond: ‘Insufficient’”).
# 
# - **Irrelevant citations**  
#   Increase `k` or enable cross-encoder re-rank (install `transformers`, `torch`).
# 
# - **Prefer a specific law**  
#   Add a `where={"law": "OR"}` filter in the `col.query(...)` call inside `retrieve()`.
# 

# # ✅ Wrap-up
# 
# - Answers are now generated **locally** with Ollama using strictly the retrieved legal context.
# - Citations are explicit and article-level, boosting trust.
# - You can toggle perspective (“Tenant” / “Landlord”) to tailor steps.
# 
# **Next (optional):** Build a tiny Streamlit UI (`app.py`) with a dropdown (Perspective), textbox (Question), and output panel (Answer + References).
# 

# In[ ]:




