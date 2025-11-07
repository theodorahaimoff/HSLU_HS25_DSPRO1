#!/usr/bin/env python
# coding: utf-8

# # Indexing & Retrieval (JSON → ChromaDB)
# 
# **Goal:**  
# Turn our per-article JSON files into a *persistent* ChromaDB index with semantic search.
# 
# **What we do here:**
# 1) Load all article-level JSONs (from `data/json/LAW/LAW_Art_*.json`)  
# 2) Embed each article using `sentence-transformers`  
# 3) Store vectors + rich metadata in **ChromaDB (PersistentClient)**  
# 4) Test retrieval (KNN) and inspect the hits for correctness
# 
# **Why this matters:**  
# A clean, persistent index lets us (a) query instantly, (b) cite exact articles, and (c) add new sources later without redoing everything.
# 

# ⚙️ Imports & Paths

# In[1]:


import os, time, json, hashlib
from pathlib import Path
from typing import List

import chromadb, logging
from openai import OpenAI
from tqdm import tqdm

# Paths (keep consistent with Notebook 1)
DATA_JSON = Path("../data/json")
CHROMA_DIR = Path("../store")
CHROMA_DIR.mkdir(parents=True, exist_ok=True)

# Collection name
CHROMA_COLLECTION = "text-embedding-3-small"

# Retrieval knobs
TOP_K  = 5     # final results returned
PRE_K  = 20    # prefetch for (optional) re-ranking

logging.getLogger("chromadb").setLevel(logging.ERROR)
os.environ["CHROMA_TELEMETRY_ENABLED"] = "false"
os.environ["POSTHOG_DISABLED"] = "true"


# In[2]:


try:
    import tomllib  # Python ≥3.11
except ModuleNotFoundError:
    import tomli as tomllib

def load_oai_token() -> str:
    """
    Loads the OpenAI API token from:
    1) streamlit.secrets (if available)
    2) .streamlit/secrets.toml (searched from cwd upwards)
    3) Environment variables (OAI_TOKEN / OPENAI_API_KEY)
    Works in both notebooks and Streamlit apps.
    """
    # --- 1) Try Streamlit secrets ---
    try:
        import streamlit as st
        token = dict(st.secrets).get("env", {}).get("OAI_TOKEN")
        if token:
            return token
    except Exception:
        pass

    # --- 2) Try loading secrets.toml manually ---
    # In Jupyter, we don’t have __file__, so we start from cwd.
    cwd = Path.cwd()
    candidates = [
        cwd / ".streamlit" / "secrets.toml",
        cwd.parent / ".streamlit" / "secrets.toml",
        cwd.parent.parent / ".streamlit" / "secrets.toml",
    ]

    for p in candidates:
        if p.exists():
            try:
                with p.open("rb") as f:
                    data = tomllib.load(f)
                token = data.get("env", {}).get("OAI_TOKEN")
                if token:
                    return token
            except Exception:
                pass

    # --- 3) Fallback to environment vars ---
    token = os.getenv("OAI_TOKEN") or os.getenv("OPENAI_API_KEY") or ""
    return token

def mask(t: str) -> str:
    return t[:4] + "…" + t[-4:] if t and len(t) > 12 else "(unset)"

OAI_TOKEN = load_oai_token()
if not OAI_TOKEN:
    raise EnvironmentError(
        "OpenAI key not found. Put it in `.streamlit/secrets.toml` under [env].OAI_TOKEN "
        "or set OAI_TOKEN/OPENAI_API_KEY in your environment."
    )

print("✅ OpenAI token loaded:", mask(OAI_TOKEN))

OAI = OpenAI(api_key=OAI_TOKEN)
EMBED_MODEL_NAME = "text-embedding-3-small"


# ## Design Choices
# 
# - **Per-article documents**: Each vector represents exactly one legal article (header + body).  
# - **Metadata**: We store `law`, `article`, `source`, `path`. This enables citations like **[OR Art. 269d – OR.pdf]**.  
# - **Persistent index**: We use `chromadb.PersistentClient` so the index survives kernel restarts.  
# - **Normalised embeddings**: Improves cosine-similarity behavior.
# 

# 🧱 Chroma helpers

# In[3]:


def get_client():
    return chromadb.PersistentClient(path=str(CHROMA_DIR))

def get_collection(client=None, name=CHROMA_COLLECTION):
    client = client or get_client()
    return client.get_or_create_collection(name)

def list_collections():
    client = get_client()
    return client.list_collections()

def wipe_collection(name=CHROMA_COLLECTION):
    client = get_client()
    try:
        client.delete_collection(name)
        print(f"Deleted collection: {name}")
    except Exception as e:
        print("Delete failed:", e)


# 🧠 Embedder init

# In[4]:


def embed_batch(texts: List[str], *, model: str = EMBED_MODEL_NAME, retries: int = 5) -> List[List[float]]:
    """
    Embed a batch of texts with OpenAI, with basic retries on 429/5xx.
    Hard-fail on 401 (bad/missing key).
    """
    delay = 1.0
    for attempt in range(retries):
        try:
            resp = OAI.embeddings.create(model=model, input=texts)
            return [d.embedding for d in resp.data]
        except Exception as e:
            # Inspect common API errors
            msg = str(e)
            if "401" in msg or "AuthenticationError" in msg:
                raise  # bad/missing key – don't retry
            if any(code in msg for code in ("429", "500", "502", "503", "504")) and attempt < retries - 1:
                time.sleep(delay)
                delay = min(delay * 2, 10)
                continue
            # Not retriable or out of retries
            raise

def embed_query(text: str) -> List[float]:
    return embed_batch([text])[0]


# 📥 Load JSON files

# In[5]:


def load_article_jsons(root: Path = DATA_JSON):
    files = sorted(root.rglob("*.json"))  # recursively loads all JSONs under all subfolders
    items = []
    for fp in files:
        try:
            data = json.loads(fp.read_text(encoding="utf-8"))
            doc_text = f"{data.get('header','')}\n{data.get('text','')}".strip()
            if len(doc_text) < 50:
                continue
            items.append({
                "id": hashlib.md5(fp.as_posix().encode("utf-8")).hexdigest()[:16],
                "text": doc_text,
                "meta": {
                    "source": data.get("source"),
                    "law": data.get("law"),
                    "title": data.get("header"),
                    "article": data.get("article"),
                    "path": fp.as_posix()
                }
            })
        except Exception as e:
            print("Skip", fp, "→", e)
    return items

articles = load_article_jsons()
print("Found", len(articles), "articles.")
if articles:
    print("Example:", articles[0]["meta"])


# ## Build / Update the Index
# 
# We’ll:
# - batch-embed the articles,
# - add them to a persistent collection,
# - print counts to confirm.
# 
# > Re-running is safe: Chroma deduplicates by IDs (we use md5 of file path).
# 

# 🏗️ Build/Update index

# In[6]:


def wipe_collection(name="swiss_private_rental_law"):
    chromadb.PersistentClient(path=str(CHROMA_DIR)).delete_collection(name)

wipe_collection("swiss_private_rental_law")

def build_index(items, batch_size=96, sleep_s=0.0):
    """
    - Batches texts, calls OpenAI embeddings
    - Upserts (ids, documents, metadatas, embeddings) into Chroma
    """
    client = get_client()
    col = get_collection(client)
    print("Collection:", CHROMA_COLLECTION, "| existing docs:", col.count())

    ids_buf, docs_buf, metas_buf = [], [], []

    for it in tqdm(items, desc="Indexing"):
        ids_buf.append(it["id"])
        docs_buf.append(it["text"])
        metas_buf.append(it["meta"])

        if len(ids_buf) >= batch_size:
            embs = embed_batch(docs_buf)
            col.upsert(ids=ids_buf, documents=docs_buf, metadatas=metas_buf, embeddings=embs)
            ids_buf, docs_buf, metas_buf = [], [], []
            if sleep_s:
                time.sleep(sleep_s)

    if ids_buf:
        embs = embed_batch(docs_buf)
        col.upsert(ids=ids_buf, documents=docs_buf, metadatas=metas_buf, embeddings=embs)

    print("Done. Chunks in collection:", col.count())
    return col

collection = build_index(articles)


# In[7]:


def assert_collection_dim(col, expected_dim: int):
    peek = col.get(limit=1, include=['embeddings'])
    if peek.get('embeddings'):
        dim = len(peek['embeddings'][0])
        if dim != expected_dim:
            raise RuntimeError(f"Collection dim={dim} != expected {expected_dim}. "
                               "Reindex with the same embedding model used at query time.")

# OpenAI text-embedding-3-small is 1536 dims
assert_collection_dim(get_collection(), 1536)


# ## Retrieval Helpers
# 
# - `retrieve(query, k, k_pre)`: embeds the query, does ANN search in Chroma, optionally re-ranks.  
# - `pack_context(...)`: formats retrieved docs for readability and later prompting.
# 

# 🧰 Retrieve & (optional) Re-rank

# In[8]:


def retrieve(query: str, k: int = TOP_K, k_pre: int = PRE_K, collection_name: str = CHROMA_COLLECTION):
    col = get_collection()
    q_emb = embed_query(query)
    res = col.query(
        query_embeddings=[q_emb],
        n_results=k_pre,
        include=['documents','metadatas','distances']
    )

    docs  = res.get('documents', [[]])[0]
    metas = res.get('metadatas', [[]])[0]
    dists = res.get('distances', [[]])[0]
    prelim = list(zip(docs, metas, dists))

    # distance ascending (smaller = closer)
    prelim = sorted(prelim, key=lambda x: x[2])
    return prelim[:k]

def pack_context(retrieved, max_chars=8000, per_source_cap=3):
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


# ## Quick Tests
# 
# We try a few canonical questions to verify that:
# - the right laws show up (OR / VMWG / StGB),
# - the retrieved articles look relevant,
# - metadata is present for citations.
# 

# In[9]:


queries = [
    "Wie fechte ich eine Mietzinserhöhung an? Welches Formular ist nötig?",
    "Welche Rechte habe ich bei Mängeln in der Wohnung?",
    "Ist eine Kündigung während eines laufenden Schlichtungsverfahrens zulässig?",
]

for q in queries:
    print("Q:", q)
    hits = retrieve(q, k=5)
    for i, (doc, meta, dist) in enumerate(hits, 1):
        print(f"  {i}. [{meta.get('law')} {meta.get('title')}] {meta.get('source')}  dist={dist:.3f}")
    print()


# 👀  Inspect one context block

# In[10]:


sample_q = "Wie fechte ich eine Mietzinserhöhung an? Welches Formular ist nötig?"
hits = retrieve(sample_q, k=6)
ctx = pack_context(hits, max_chars=3000)
print(ctx[:1500])


# # ✅ Summary
# 
# - We built a **persistent ChromaDB index** from per-article JSONs.  
# - Retrieval returns focused legal articles with clean metadata for citations.  
# - Optional cross-encoder rerank is wired (enable if installed).
# 
# **Next:** `3_Answering_and_Evaluation.ipynb`  
# We will:
# - assemble prompts,
# - answer via **Ollama HTTP** or **OpenAI API**,
# - enforce a strict output format (1-sentence answer, steps, forms, references),
# - run a small evaluation set (sanity checks, error cases).
# 

# In[ ]:





# In[ ]:




