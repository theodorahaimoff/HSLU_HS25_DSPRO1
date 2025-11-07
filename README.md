# Rental Law RAG (Swiss OR / VMWG / StGB)

**What it does**  
Retrieval-Augmented QA for Swiss rental law. It splits PDFs into **per-article** JSON, builds a **Chroma** vector index, and answers questions with a **local Ollama** model (`llama3:8b`), citing `[LAW Art.X – filename]`.

---

## Repo Layout
```bash
rental_law_rag/
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   ├── raw/
│   │   ├── OR.pdf
│   │   ├── STGB.pdf
│   │   └── VMWG.pdf
│   └── json/
│       └── .gitkeep
├── store/
│   └── .gitkeep
├── notebooks/
│   ├── 0_installations.ipynb
│   ├── 1_data_preparation.ipynb
│   ├── 2_indexing_and_retrieval.ipynb
│   └── 3_answer_generation_local_ollama.ipynb
└── src/
    ├── _0_installations.py
    ├── _1_data_preparation.py
    ├── _2_indexing_and_retrieval.py
    ├── _3_answer_generation.py
    └── main_local_ollama.py
```



---

## Setup

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt


ollama serve            # in one terminal
ollama pull llama3:8b   # once
```

## Usage

### 1️⃣ Add PDFs
Put the 3 law PDFs into `data/raw/`:
- `OR.pdf`
- `VMWG.pdf`
- `STGB.pdf`

### 2️⃣ Install packages
Run **Notebook 0** (`0_installation.ipynb`) \
→ installs the packages needed for the next steps

### 3️⃣ Build JSON dataset
Run **Notebook 1** (`1_data_preparation.ipynb`)  
→ generates per-article JSON files in `data/json/`.

### 4️⃣ Build Chroma index
Run **Notebook 2** (`2_indexing_and_retrieval.ipynb`)  
→ creates the persistent database in `store/`.

### 5️⃣ Ask questions
Run **Notebook 3** (`3_answering_and_evaluation.ipynb`)  
→ asks Ollama locally (model `llama3:8b`) and prints legal answers with citations.

### 6️⃣ Run Streamlit app
Run the following command on your terminal
```bash
streamlit run src/main_local_ollama.py
```
The application's GUI should now be available under http://localhost:8501/

## Notes for Collaborators
- Generated folders (`data/json/`, `store/`) are **git-ignored** — everyone rebuilds them locally.
- If Ollama isn’t running, start it using:
  ```bash
  ollama serve
  ollama pull llama3:8b
  ```
- After editing any of the notebooks, generate the respective Python script:
  ```bash
  jupyter nbconvert --to script notebooks/1_data_preparation.ipynb --output "_1_data_preparation.py" --output-dir=src
  ```
  > 👉 **Note** \
  > Make sure you keep the _ character in front of the output file name, Python has issues when a script begins with a number