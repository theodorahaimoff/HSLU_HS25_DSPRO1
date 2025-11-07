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

# In[9]:


import os, json, hashlib
from pathlib import Path

import chromadb, logging
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# Paths (keep consistent with Notebook 1)
DATA_JSON = Path("../data/json")
CHROMA_DIR = Path("../store")
CHROMA_DIR.mkdir(parents=True, exist_ok=True)

# Collection name
CHROMA_COLLECTION = "swiss_private_rental_law"

# Embedding model (fast + solid)
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Retrieval knobs
TOP_K  = 5     # final results returned
PRE_K  = 20    # prefetch for (optional) re-ranking

logging.getLogger("chromadb").setLevel(logging.ERROR)


# ## Design Choices
# 
# - **Per-article documents**: Each vector represents exactly one legal article (header + body).  
# - **Metadata**: We store `law`, `article`, `source`, `path`. This enables citations like **[OR Art. 269d – OR.pdf]**.  
# - **Persistent index**: We use `chromadb.PersistentClient` so the index survives kernel restarts.  
# - **Normalised embeddings**: Improves cosine-similarity behavior.
# 

# 🧱 Chroma helpers

# In[10]:


# Disable analytics / telemetry
os.environ["CHROMA_TELEMETRY_ENABLED"] = "false"
os.environ["POSTHOG_DISABLED"] = "true"

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

# In[11]:


_embedder = None

def embedder():
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer(EMBED_MODEL_NAME)
    return _embedder


# 📥 Load JSON files

# In[12]:


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

# In[13]:


def build_index(items, batch_size=64):
    client = get_client()
    col = get_collection(client)
    print("Collection:", CHROMA_COLLECTION, "| existing docs:", col.count())

    ids, docs, metas = [], [], []
    model = embedder()

    for it in tqdm(items, desc="Indexing"):
        ids.append(it["id"])
        docs.append(it["text"])
        metas.append(it["meta"])

        if len(ids) >= batch_size:
            embs = model.encode(docs, show_progress_bar=False, normalize_embeddings=True).tolist()
            col.upsert(ids=ids, documents=docs, metadatas=metas, embeddings=embs)
            ids, docs, metas = [], [], []

    if ids:
        embs = model.encode(docs, show_progress_bar=False, normalize_embeddings=True).tolist()
        col.upsert(ids=ids, documents=docs, metadatas=metas, embeddings=embs)

    print("Done. Chunks in collection:", col.count())
    return col

collection = build_index(articles)


# ## Retrieval Helpers
# 
# - `retrieve(query, k, k_pre)`: embeds the query, does ANN search in Chroma, optionally re-ranks.  
# - `pack_context(...)`: formats retrieved docs for readability and later prompting.
# 

# 🧰 Retrieve & (optional) Re-rank

# In[14]:


def retrieve(query: str, k: int = TOP_K, k_pre: int = PRE_K, collection_name: str = CHROMA_COLLECTION):
    col = get_collection()
    q_emb = embedder().encode([query], normalize_embeddings=True).tolist()[0]
    res = col.query(query_embeddings=[q_emb], n_results=k_pre, include=['documents','metadatas','distances'])

    docs  = res.get('documents', [[]])[0]
    metas = res.get('metadatas', [[]])[0]
    dists = res.get('distances', [[]])[0]
    prelim = list(zip(docs, metas, dists))

    # Optional cross-encoder re-rank: uncomment if you installed transformers/torch
    try:
        from sentence_transformers import CrossEncoder
        rnk = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        scores = rnk.predict([(query, d) for d,_,_ in prelim]).tolist()
        prelim = [p for p,_ in sorted(zip(prelim, scores), key=lambda x: x[1], reverse=True)]
    except Exception:
        # Fallback: sort by distance asc (smaller = closer)
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

# In[15]:


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

# In[16]:


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




