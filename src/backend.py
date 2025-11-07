#!/usr/bin/env python
# coding: utf-8

# In[5]:


import os, json, logging, re
from pathlib import Path
from typing import Iterable, Tuple
import streamlit as st
from openai import OpenAI
from jsonschema import validate

try:
    import chromadb
except Exception as e:
    st.sidebar.error(
        "ChromaDB couldn’t be imported." + str(e)
    )
    raise

# Retrieval knobs
TOP_K  = 5
PRE_K  = 20
MAX_CTX_CHARS = 8000

logger = logging.getLogger("SwissRentalLawApp")


# In[6]:


def get_base_dir() -> Path:
    """
    Returns the project base directory that works both:
    - in normal scripts (via __file__)
    - in notebooks (via current working directory)
    """
    try:
        return Path().parent.resolve()
    except NameError:
        # __file__ not defined (e.g., in Jupyter or interactive)
        return Path(os.getcwd()).resolve()

store_dir = get_base_dir()/ "store"

def load_manifest():
    try:
        mf = json.loads((store_dir / "manifest.json").read_text(encoding="utf-8"))
        return mf
    except Exception as e:
        st.sidebar.error(f"Manifest error: {e}")
        return None

def _mf():
    return load_manifest()

def _embedding_model_name():
    return _mf()["model"]

def _collection_name():
    return _mf()["collection"]

def _chroma_dir():
    return store_dir / _collection_name()

def _expected_dim():
    return _mf()["dim"]


# In[7]:


def _get_oai_token():
    try:
        s = dict(st.secrets)
        return s.get("env", {}).get("OAI_TOKEN") or os.getenv("OAI_TOKEN") or ""
    except Exception:
        return os.getenv("OAI_TOKEN") or ""

def get_oai_client():
    key = _get_oai_token()
    if not key:
        raise EnvironmentError("OpenAI token missing (set OAI_TOKEN or OPENAI_API_KEY).")
    return OpenAI(api_key=key)

def embed_query(text: str) -> list:
    client = get_oai_client()
    resp = client.embeddings.create(model=_embedding_model_name(), input=text)
    return resp.data[0].embedding


# In[8]:


def get_client():
    return chromadb.PersistentClient(path=_chroma_dir())

def get_collection(name: str | None = None):
    name = name or _collection_name()
    col = get_client().get_collection(name)
    logger.info(f"Loaded Chroma collection '{name}' with {col.count()} entries.")
    return col


# In[11]:


def retrieve(query: str, k: int = TOP_K, k_pre: int = PRE_K, collection_name: str | None = None):
    col = get_collection(collection_name)
    q_emb = embed_query(query)
    res = col.query(query_embeddings=[q_emb], n_results=k_pre, include=['documents','metadatas','distances'])
    docs  = res.get('documents', [[]])[0]
    metas = res.get('metadatas', [[]])[0]
    dists = res.get('distances', [[]])[0]
    prelim = sorted(list(zip(docs, metas, dists)), key=lambda x: x[2])
    return prelim[:k]

def pack_context(retrieved, max_chars=MAX_CTX_CHARS, per_source_cap=3):
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


# In[12]:


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


def answer_with_openai(question: str, perspective: str, k=TOP_K, model="gpt-4o-mini", max_chars=MAX_CTX_CHARS):
    hits = retrieve(question, k=k)
    context = pack_context(hits, max_chars=max_chars)

    prompt = PROMPT.format(context=context, question=f"Perspective: {perspective}, Question: {question}")

    client = get_oai_client()
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": [{"type": "text", "text": prompt}]},
            {"role": "user",   "content": [{"type": "text", "text":
                "Beantworte die Frage gemäss obigen Vorgaben. Fülle die Felder des JSON-Schemas aus. "
                "Sprache: Deutsch (Schweiz), Du-Form. Für 'steps' gilt: Gib 2–8 kurze Einträge zurück, "
                "jeder Eintrag genau EIN Schritt, EINE Zeile, KEINE Nummerierung oder Zeilenumbrüche. "
                "Für 'forms': gib die genauen offiziellen Bezeichnungen aus dem CONTEXT zurück (leer, wenn keine). "
                "Gib NUR JSON zurück."
            }]}
        ],
        temperature=0,
        response_format={"type": "json_object"}
    )
    content = resp.choices[0].message.content or ""
    parsed = json.loads(content)  # will raise clearly if not JSON
    validate(instance=parsed, schema=schema)

    steps = parsed.get("steps") or []
    forms = parsed.get("forms") or []
    refs  = parsed.get("references") or []

    # normalize
    if isinstance(steps, str): steps = [s.strip() for s in steps.split("\n") if s.strip()]
    if isinstance(forms, str): forms = [f.strip() for f in forms.split("\n") if f.strip()]

    return (parsed.get("answer","").strip(), steps, forms, refs, hits)


# In[ ]:


def sanitize_step(step: str) -> str:
    """Remove rare leading artifact ']:' from a step string."""
    return re.sub(r"^\s*\]:\s*", "", step or "").strip()

def format_numbered(items: Iterable[str]) -> str:
    """Format an iterable into a numbered Markdown list."""
    items = [i for i in items if i]
    return "\n" + "\n".join(f"{i + 1}. {t}" for i, t in enumerate(items)) if items else "Keine Schritte gefunden."

def format_bulleted(items: Iterable[str]) -> str:
    """Format an iterable into a bulleted Markdown list."""
    items = [i for i in items if i]
    return "\n" + "\n".join(f"- {t}" for t in items) if items else ""


# In[ ]:


def generate_answer(question: str, perspective: str) -> Tuple[str, str, str, str]:
    """
    Retrieve context, query the Hugging Face model, and return formatted Markdown sections.

    Returns:
        (answer_text, steps_md, forms_md, sources_md)
    """
    try:
        logger.debug(f"Generating answer | Perspective: {perspective} | Question: {question[:80]}")

        answer_text, steps, forms, references, _ = answer_with_openai(
            question, perspective=perspective, k=TOP_K
        )

        # --- Steps ---
        if isinstance(steps, str):
            steps = [s.strip() for s in steps.split("\n") if s.strip()]
        clean_steps = [sanitize_step(s) for s in (steps or [])]
        steps_md = format_numbered(clean_steps)

        # --- Forms ---
        if isinstance(forms, str):
            forms = [f.strip() for f in forms.split("\n") if f.strip()]
        forms_md = format_bulleted(forms) if forms else "Keine Formulare gefunden."

        # --- Sources ---
        if references:
            dedup = {(r.get("law", "?"), r.get("title", "?")) for r in references}
            ordered = sorted(dedup, key=lambda x: (x[0], x[1]))
            sources_md = format_bulleted([f"**{law}** {title}" for law, title in ordered])
        else:
            sources_md = "Keine Quellen gefunden."

        return (answer_text or "").strip(), steps_md, forms_md, sources_md

    except Exception:
        logger.exception("Error during answer generation.")
        raise


# In[ ]:




