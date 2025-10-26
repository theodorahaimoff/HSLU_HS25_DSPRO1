#!/usr/bin/env python
# coding: utf-8

# # Data Preparation (PDF → JSON)
# 
# **Goal:**  
# Convert all Swiss rental-law PDFs (OR, VMWG, StGB) into clean, structured JSON files —  
# where **each JSON = exactly one legal article**.
# 
# This makes later retrieval and referencing much easier and more accurate.
# 
# **Context:**  
# - Splitting at *article-level granularity* instead of page chunks.
# - Adding metadata (law name, article number, source).
# - Keeping a clean and reproducible data pipeline.

# ⚙️ Imports and Setup

# In[1]:


import re, json
from pathlib import Path
import pymupdf
from tqdm import tqdm

# Paths
DATA_RAW = Path("../data/raw")     # PDFs go here
DATA_JSON = Path("../data/json")   # Will hold one JSON per article
DATA_JSON.mkdir(parents=True, exist_ok=True)


# ### Explanation
# We’ll use:
# - **PyMuPDF (fitz)** to extract text page by page.  
# - **Regex** to detect “Art. XXX” headers.  
# - **tqdm** for nice progress bars.  
# 
# We’ll store results as JSON so each file can be directly embedded later.
# 

# 🧩 Helper Functions

# In[2]:


# --- Cleaning & Splitting ---
ART_HEADER = re.compile(r"(?m)^\s*(Art\.\s*\d+[a-zA-Z]*\b[^\n]*)\s*$")

def clean_text(t: str) -> str:
    """Normalize whitespace and remove artifacts."""
    t = t.replace("\x0c", " ").replace("\u00ad", "")
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"\s+\n", "\n", t)
    t = re.sub(r"\n\s+", "\n", t)
    return t.strip()

def read_pdf_text(pdf_path: Path) -> str:
    """Extract all text from a PDF using PyMuPDF."""
    pages = []
    with pymupdf.open(pdf_path) as doc:
        for p in doc:
            pages.append(p.get_text("text"))
    return clean_text("\n".join(pages))

def split_articles(full_text: str):
    """Split a document into (header, body) per article."""
    headers = list(ART_HEADER.finditer(full_text))
    articles = []
    for i, m in enumerate(headers):
        start = m.start()
        end = headers[i+1].start() if i+1 < len(headers) else len(full_text)
        block = full_text[start:end].strip()
        header_line = m.group(1).strip()
        body = block[len(header_line):].strip()
        articles.append((header_line, body))
    return articles

def parse_article_number(header_line: str):
    m = re.search(r"Art\.\s*(\d+[a-zA-Z]*)", header_line)
    return m.group(1) if m else None


# ### Explanation
# Each article in Swiss laws starts with `Art.` followed by a number or letter.  
# This regex isolates those headers and splits the PDF into article blocks.  
# We also extract the article number (e.g. `269d`, `325bis`) for metadata.
# 

# 📄 Process PDFs → Save JSON

# In[3]:


def detect_law_tag(stem: str) -> str:
    s = stem.upper()
    if "OR" in s: return "OR"
    if "VMWG" in s: return "VMWG"
    if "STGB" in s or "STG" in s: return "StGB"
    return stem

def ingest_pdf(pdf_path: Path):
    law = detect_law_tag(pdf_path.stem)
    text = read_pdf_text(pdf_path)
    articles = split_articles(text)

    out_dir = DATA_JSON / law
    out_dir.mkdir(parents=True, exist_ok=True)

    for header, body in articles:
        art_nr = parse_article_number(header) or "NA"
        payload = {
            "law": law,
            "article": art_nr,
            "header": header,
            "text": body,
            "source": pdf_path.name
        }
        out_fp = out_dir / f"{law}_Art_{art_nr}.json"
        out_fp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    return len(articles)


# ▶️ Run Conversion

# In[4]:


pdfs = sorted(DATA_RAW.glob("*.pdf"))
print("Found PDFs:", [p.name for p in pdfs])

total_articles = 0
for pdf in tqdm(pdfs, desc="Processing PDFs"):
    total_articles += ingest_pdf(pdf)

print(f"✅ Done! Created ~{total_articles} article JSON files.")


# ### Explanation
# Each PDF is scanned and split into articles.
# Every JSON file now represents **exactly one article** (e.g. `OR_Art_269d.json`).
# We'll use these later to build embeddings with ChromaDB.
# 

# 👀 Quick Inspection

# In[5]:


samples = list((DATA_JSON / "OR").glob("*.json"))[:3]
for s in samples:
    print("File:", s.name)
    data = json.loads(s.read_text(encoding="utf-8"))
    print(f"Header: {data['header']}")
    print(f"Excerpt: {data['text'][:250]}...\n")


# ### Explanation
# We quickly check that:
# - Articles are correctly separated.  
# - Text doesn’t include the next article.  
# - Metadata (law, article number) is stored correctly.
# 

# # ✅ Summary
# We now have a clean, article-level dataset ready for indexing.
# 
# **Next notebook: `2_Indexing_and_Retrieval.ipynb`**
# We'll:
# - Load all JSONs,
# - Embed them with Sentence Transformers,
# - Store them in a persistent ChromaDB collection for fast semantic search.
# 
# **Benefits of this structure**
# - Easier to debug and explain
# - Perfect granularity (one legal article per data point)
# - Can easily add new laws or update existing ones
# 

# In[ ]:




