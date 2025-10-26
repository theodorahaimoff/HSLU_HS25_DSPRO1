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

# In[2]:


import os, json, requests
from pathlib import Path
from typing import List, Tuple

import chromadb, logging
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# Paths (same as Notebook 2)
CHROMA_DIR = Path("../store")
CHROMA_COLLECTION = "swiss_private_rental_law"

# Embedding model — must match what you used when indexing
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Retrieval knobs
TOP_K  = 5
PRE_K  = 20
MAX_CTX_CHARS = 8000

# Ollama local settings
OLLAMA_HOST  = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3:8b")

logging.getLogger("chromadb").setLevel(logging.ERROR)


# We verify:
# - Chroma store exists and is readable
# - The collection is present
# - Ollama is reachable and model is available
# 

# Chroma & Embedder helpers (same logic as indexing_and_retrieval)

# In[3]:


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

# In[4]:


try:
    col = get_collection()
    print("Collection:", CHROMA_COLLECTION, "| count:", col.count())
except Exception as e:
    raise SystemExit(f"❌ Could not open Chroma collection. Did you run Notebook 2? Error: {e}")


# Check Ollama is running

# In[5]:


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

# In[6]:


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
    ctx, total, seen = [], 0, {}
    for doc, meta, dist in retrieved:
        key = (meta.get("law"), meta.get("article"))
        seen[key] = seen.get(key, 0) + 1
        if seen[key] > per_source_cap:
            continue
        stamp = f"[{meta.get('law','?')} Art.{meta.get('article','?')} – {meta.get('source')}]"
        block = f"{stamp}\n{doc.strip()}\n\n"
        if total + len(block) > max_chars:
            break
        ctx.append(block); total += len(block)
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

# In[62]:


PROMPT = """You are a Swiss rental-law assistant.
Answer ONLY from the CONTEXT. If insufficient, say so.
Do NOT refer to yourself, your role, or your identity in the answer.
Start directly with the content requested (no introductions).

FORMAT STRICTLY:
1) "Answer:" Write ONE concise sentence summarizing the answer.
2) "Steps/Options:" A section with numbered points (1., 2., 3., …) depending on the perspective you're given (Tenant or Landlord).
3) "Forms:" A list of the forms needed (exact names if present) in bullet points (- item).
4) "Read next:" A list of articles to read next (e.g., Art. 269 OR; Art. 19 VMWG) in bullet points (- item).
5) "References:" A list of the distinct sources as [law name Art.X – filename] in bullet points (- item).
6) Respond ONLY in the specified language and keep exactly this structure and order, do NOT merge multiple steps with semicolons or commas

CONTEXT:
{context}

QUESTION:
{question}
"""

def answer_with_ollama(question: str, perspective: str, language: str, k=TOP_K, model=OLLAMA_MODEL, host=OLLAMA_HOST):
    hits = retrieve(question, k=k)
    context = pack_context(hits, max_chars=MAX_CTX_CHARS)
    prompt = PROMPT.format(context=context, question=f"[Perspective: {perspective}] [Language: {language}] {question}")

    r = requests.post(f"{host}/api/generate",
                      json={"model": model, "prompt": prompt, "stream": False},
                      timeout=120)
    if r.status_code != 200:
        return f"[Ollama error {r.status_code}]: {r.text}", hits

    text = r.json().get("response", "").strip()
    return text, hits


# Try a realistic query and inspect the sources retrieved.
# 

# Single question test

# In[63]:


q = "Wie fechte ich eine Mietzinserhöhung an? Welches Formular ist nötig?"
ans, hits = answer_with_ollama(q, perspective="Tenant", language="German", k=6)
print("=== ANSWER ===\n", ans, "\n")
print("=== SOURCES ===")
for _, m, _ in hits:
    print(f"- {m.get('law')} Art.{m.get('article')} – {m.get('source')}")


# We’ll run several canonical questions to check:
# - Structure & clarity of answers
# - That references point to the right law/articles
# - That forms are extracted when present (from VMWG, OR)
# 

# Batch evaluation

# In[64]:


eval_questions = [
    ("Wie fechte ich eine Mietzinserhöhung an? Welches Formular ist nötig?", "Tenant", "English"),
    ("Welche Rechte habe ich bei Mängeln in der Wohnung?", "Tenant", "German"),
    ("Darf der Vermieter während laufendem Schlichtungsverfahren kündigen?", "Landlord", "English"),
    ("Wann sind Mietzinserhöhungen wegen energetischer Verbesserungen zulässig?", "Landlord", "German"),
]

for q, perspective, language in eval_questions:
    print("\n" + "="*150)
    print("Q:", q, "| Perspective:", perspective, "| Language: ", language)
    print("="*150)
    ans, hits = answer_with_ollama(q, perspective=perspective, language=language, k=6)
    print("\n--- ANSWER ---\n", ans[:2000])  # trim for display
    print("\n--- REFERENCES ---")
    refs = {(m.get('law'), m.get('article'), m.get('source')) for _, m, _ in hits}
    for law, art, src in refs:
        print(f"[{law} Art.{art} – {src}]")


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




