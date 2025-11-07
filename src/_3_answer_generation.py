#!/usr/bin/env python
# coding: utf-8

# # Answering & Evaluation (Chroma → OpenAI)
# 
# Take retrieved legal articles (from Chroma) and generate a **grounded, structured answer** using the GPT-4o-mini model.
# 

# ## Imports & Paths

# In[1]:


import os, json, logging

from pathlib import Path

import chromadb
from chromadb.config import Settings
from openai import OpenAI
from jsonschema import validate, ValidationError

# Retrieval knobs
TOP_K  = 5
PRE_K  = 20
MAX_CTX_CHARS = 8000

logging.getLogger("chromadb").setLevel(logging.DEBUG)


# In[2]:


def get_base_dir() -> Path:
    """
    Returns the project base directory that works both:
    - in normal scripts (via __file__)
    - in notebooks (via current working directory)
    """
    try:
        return Path(__file__).resolve().parent
    except NameError:
        # __file__ not defined (e.g., in Jupyter or interactive)
        return Path(os.getcwd()).resolve()

def load_manifest():
    base_dir =  get_base_dir()
    store_dir = (base_dir.parent / "store").resolve()
    mf = json.loads((store_dir / "manifest.json").read_text(encoding="utf-8"))
    mf["store_dir"] = str(store_dir / mf["dir"])         # absolute path to the versioned dir
    return mf

MF = load_manifest()

CHROMA_SETTINGS = Settings(anonymized_telemetry=False, allow_reset=True)
EMBED_MODEL_NAME   = MF["model"]       # replaces hardcoded "text-embedding-3-small"
CHROMA_COLLECTION  = MF["collection"]  # replaces hardcoded collection name
CHROMA_DIR         = MF["store_dir"]   # replaces path to store
EXPECTED_DIM       = MF["dim"]


# In[3]:


try:
    import streamlit as st  # noqa
except Exception:
    st = None

def _get_oai_token():
    try:
        s = dict(st.secrets)
        return s.get("env", {}).get("OAI_TOKEN") or os.getenv("OAI_TOKEN") or ""
    except Exception:
        return os.getenv("OAI_TOKEN") or ""

OAI_TOKEN = _get_oai_token()

if not OAI_TOKEN:
    if st is not None:
        # stop Streamlit cleanly with a visible error
        st.error("OpenAI Token fehlt. Lege es in `.streamlit/secrets.toml` unter `[env].OAI_TOKEN` "
                 "oder als Env-Var `OAI_TOKEN` an.")
        st.stop()
    else:
        # non-Streamlit context (CLI/tests)
        raise EnvironmentError("OAI_TOKEN missing. Set env var or Streamlit secret.")

OAI_CLIENT = OpenAI(api_key=OAI_TOKEN)
OAI_MODEL = "gpt-4o-mini"


# ## Chroma & Embedder helpers

# In[4]:


def get_client():
    return chromadb.PersistentClient(path=CHROMA_DIR, settings=CHROMA_SETTINGS)

def get_collection(name: str = CHROMA_COLLECTION):
    client = get_client()
    return client.get_collection(name)

def _assert_dim(col, expected=EXPECTED_DIM):
    peek = col.get(limit=1, include=["embeddings"])
    if peek.get("embeddings"):
        dim = len(peek["embeddings"][0])
        if dim != expected:
            raise RuntimeError(f"Index dim={dim} != manifest dim={expected}. Update manifest or rebuild index.")

def embed_query(text: str) -> list[float]:
    resp = OAI_CLIENT.embeddings.create(
        model=EMBED_MODEL_NAME,
        input=text
    )
    return resp.data[0].embedding


# ### Check collection & doc count

# In[5]:


if __name__ == "__main__":
    # local dev only
    try:
        col = get_collection()
        print("Collection:", CHROMA_COLLECTION, "| count:", col.count())
        _assert_dim(col)
    except Exception as e:
        print("Chroma check failed:", e)


# ### Retrieve & re-rank + pack context

# In[6]:


def retrieve(query: str, k: int = TOP_K, k_pre: int = PRE_K, collection_name: str = CHROMA_COLLECTION):
    col = get_collection(collection_name)

    # 1) embed the query with OpenAI
    q_emb = embed_query(query)

    # 2) query Chroma (no reranker, distance sort)
    res = col.query(
        query_embeddings=[q_emb],
        n_results=k_pre,
        include=['documents', 'metadatas', 'distances']
    )

    docs  = res.get('documents', [[]])[0]
    metas = res.get('metadatas', [[]])[0]
    dists = res.get('distances', [[]])[0]

    prelim = list(zip(docs, metas, dists))
    prelim = sorted(prelim, key=lambda x: x[2])  # smaller distance = closer
    return prelim[:k]

def pack_context(retrieved, max_chars=MAX_CTX_CHARS, per_source_cap=3):
    #Build context string from retrieved docs.
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


# ## Prompt design
# 
# We force a strict structure for answers and **forbid** using anything outside the provided context.
# 
# **Format required:**
# 1) One-sentence answer.
# 2) Numbered steps/options (say if they apply to Tenant or Landlord).
# 3) Forms required (exact names if present).
# 4) Sources (e.g. OR Art.x, Name).
# 

# In[7]:


# --- Prompt (escaped braces; single {question}) ---
PROMPT = """You are a Swiss rental-law assistant.
Answer ONLY from the CONTEXT. If insufficient, say so.
Write in German (Switzerland) in strictly 'Du'-Form.
Do NOT refer to yourself, your role, or your identity in the answer.
Start directly with the content requested (no introductions).
Return STRICTLY a JSON object with this shape, no extra keys, no markdown fences, no numbering:

"answer": "one concise sentence",\\n'
"steps": ["one unique action per entry for the given perspective, no numbering, 2-4 items"],\\n'
"forms": ["exact official names from CONTEXT, or empty array"],\\n'
"references": [{{"law": "OR", "title": "Art.x, Article Title", "source": "OR.pdf"}}]\\n'

CONTEXT:
{context}

QUESTION:
{question}
"""

# --- JSON schema for validation ---
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
                    "law":   {"type": "string"},
                    "title": {"type": "string"},
                    "source":{"type": "string"}
                },
                "required": ["law", "title", "source"]
            }
        }
    },
    "required": ["answer", "references"]
}

def answer_with_openai(question: str, perspective: str, k=TOP_K, model=OAI_MODEL, max_chars=MAX_CTX_CHARS):
    """
    Query OpenAI with retrieved context and return:
    (generated_answer, steps, forms, references, hits)
    """
    # 1) Retrieve documents
    hits = retrieve(question, k=k)
    context = pack_context(hits, max_chars=max_chars)

    # 2) Build prompt
    prompt = PROMPT.format(
        context=context,
        question=f"Perspective: {perspective}, Question: {question}"
    )

    # 3) Call Chat Completions in JSON mode (SDK 2.7.1)
    try:
        resp = OAI_CLIENT.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": [{"type": "text", "text": prompt}]},
                {"role": "user",   "content": [{"type": "text", "text":
                    "Beantworte die Frage gemäss obigen Vorgaben. "
                    "Fülle die Felder des JSON-Schemas aus. "
                    "Sprache: Deutsch (Schweiz), Du-Form. "
                    "Für 'steps' gilt: Gib 2–8 kurze Einträge zurück, "
                    "jeder Eintrag genau EIN Schritt, EINE Zeile, KEINE Nummerierung oder Zeilenumbrüche. "
                    "Für 'forms': gib die genauen offiziellen Bezeichnungen aus dem CONTEXT zurück (leer, wenn keine). "
                    "Gib NUR JSON zurück."
                }]}
            ],
            temperature=0,
            response_format={"type": "json_object"}
        )
    except Exception as e:
        return f"[OpenAI error]: {e}", [], [], [], hits

    # 4) Parse JSON
    content = resp.choices[0].message.content or ""
    try:
        parsed = json.loads(content)
    except Exception as e:
        return f"[Parse error]: {e}\nRaw: {content}", [], [], [], hits

    # 5) Validate schema
    try:
        validate(instance=parsed, schema=schema)
    except ValidationError as ve:
        return f"[Schema validation error]: {ve.message}\nRaw: {parsed}", [], [], [], hits

    # 6) Normalize & return
    answer_text = (parsed.get("answer") or "").strip()

    steps = parsed.get("steps") or []
    if isinstance(steps, str):
        steps = [s.strip() for s in steps.split("\n") if s.strip()]

    forms = parsed.get("forms") or []
    if isinstance(forms, str):
        forms = [f.strip() for f in forms.split("\n") if f.strip()]

    references = parsed.get("references") or []

    return answer_text, steps, forms, references, hits


# ## Local Testing

# ### Single question test

# In[8]:


def single_question_test():
    q = "Wie fechte ich eine Mietzinserhöhung an? Welches Formular ist nötig?"
    ans, steps, forms, references, hits = answer_with_openai(q, perspective="Mieter:in", k=6)
    print("=== ANSWER ===\n", ans, "\n")
    print("=== STEPS ===\n", steps, "\n")
    print("=== FORMS ===\n", forms, "\n")
    print("=== SOURCES ===\n", references, "\n")


# In[9]:


#single_question_test()


# ### Batch evaluation

# In[10]:


def batch_evaluation():
    eval_questions = [
        ("Wie fechte ich eine Mietzinserhöhung an? Welches Formular ist nötig?", "Mieter:in"),
        ("Welche Rechte habe ich bei Mängeln in der Wohnung?", "Mieter:in"),
        ("Darf der Vermieter während laufendem Schlichtungsverfahren kündigen?", "Vermieter:in"),
        ("Wann sind Mietzinserhöhungen wegen energetischer Verbesserungen zulässig?", "Vermieter:in"),
    ]

    for q, perspective in eval_questions:
        print("\n" + "="*150)
        print("Q:", q, "| Perspective:", perspective)
        print("="*150)
        ans, steps, forms, references, hits = answer_with_openai(q, perspective=perspective, k=6)
        print("\n--- ANSWER ---\n", ans[:2000])  # trim for display
        print("=== STEPS ===\n", steps[:2000])
        print("=== FORMS ===\n", forms)
        print("=== SOURCES ===\n", references)


# In[11]:


#batch_evaluation()


# In[ ]:




