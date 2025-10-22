# Rental Law RAG (Swiss OR / VMWG / StGB)

**What it does**  
Retrieval-Augmented QA for Swiss rental law. It splits PDFs into **per-article** JSON, builds a **Chroma** vector index, and answers questions with a **local Ollama** model (`llama3:8b`), citing `[LAW Art.X – filename]`.

---

## Repo Layout

rental_law_rag/
├── README.md
├── requirements.txt
├── .gitignore
├── .env.example
├── data/
│ ├── raw/ # put PDFs here (OR.pdf, VMWG.pdf, STGB.pdf)
│ └── json/ # generated per-article JSON (gitignored)
├── store/ # Chroma index (gitignored)
├── src/
│ ├── pdf_ingest.py # PDF → per-article JSON
│ ├── build_index.py # JSON → Chroma index
│ ├── retrieve.py # retrieval helpers + context pack
│ ├── answer.py # call Ollama to answer from context
│ └── cli.py # simple CLI interface
└── notebooks/
├── 0_Installations.ipynb (+ .py)
├── 1_Data_Preparation.ipynb (+ .py)
├── 2_Indexing_and_Retrieval.ipynb (+ .py)
└── 3_Answering_and_Evaluation.ipynb (+ .py)



---

## Setup

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt


ollama serve            # in one terminal
ollama pull llama3:8b   # once


## Usage

### 1️⃣ Add PDFs
Put the 3 law PDFs into `data/raw/`:
- `OR.pdf`
- `VMWG.pdf`
- `STGB.pdf`

### 2️⃣ Build JSON dataset
Run **Notebook 1** (`1_Data_Preparation.ipynb`)  
→ generates per-article JSON files in `data/json/`.

### 3️⃣ Build Chroma index
Run **Notebook 2** (`2_Indexing_and_Retrieval.ipynb`)  
→ creates the persistent database in `store/`.

### 4️⃣ Ask questions
Run **Notebook 3** (`3_Answering_and_Evaluation.ipynb`)  
→ asks Ollama locally (model `llama3:8b`) and prints legal answers with citations.

## Notes for Collaborators
- Generated folders (`data/json/`, `store/`) are **git-ignored** — everyone rebuilds them locally.
- If Ollama isn’t running, start it:
  ```bash
  ollama serve
  ollama pull llama3:8b

