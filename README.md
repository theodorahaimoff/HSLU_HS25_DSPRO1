# <img src="img/logo.png" alt="logo" width="30"/> rently

**What it does**  

**rently** is a Retrieval-Augmented Generation (**RAG**) application that answers legal questions about Swiss rental law. \
It uses **OpenAI** embeddings to index articles from the Obligationenrecht (OR), Verordnung über die Miete und Pacht von Wohn- und Geschäftsräumen (VMWG), and Strafgesetzbuch (StGB) into a persistent **ChromaDB** store. \
When a user asks a question, **rently** retrieves the most relevant legal articles, builds a concise context, and generates a structured, grounded answer using the **GPT-4o-mini** model. 

Find the productive application here: [rently-ch.streamlit.app](https://rently-ch.streamlit.app/)

## 🗂️ Repo Layout
```bash
HSLU_HS25_DSPRO1/
├── README.md
├── requirements.txt
├── .gitignore
├── .streamlit/
│   └── config.toml          # contains Streamlit global configuration
├── data/
│   ├── raw/                  # input PDFs (OR.pdf, VMWG.pdf, STGB.pdf)
│   └── json/                 # per-article JSON files (auto generated)
│      └── .gitkeep 
├── notebooks/
│   ├── 0_installations.ipynb
│   ├── 1_data_preparation.ipynb
│   ├── 2_indexing_and_retrieval.ipynb
│   └── 3_answer_generation.ipynb
├── src/                          
│   ├── logs/
│   │   └── .gitkeep 
│   ├── app_backend.py            # Streamlit Backend (generated from notebook)
│   ├── app.py                    # Streamlit UI
│   └── cloud_debug_app.py        # helper for debugging Streamlit Cloud
└── store/                        # persistent Chroma database used by Streamlit
    ├── UID/
    ├── chroma.sqlite3
    └── manifest.json         # manifest containing the database information of the productive application

```

## ⚙️ Setup (local)

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```
### Environment

Add your OpenAI API key to .streamlit/secrets.toml:
```bash
[env]
OAI_TOKEN = "sk-s..."
```

## 🧩 Initialisation
If you're not running the application it the first time, skip to **step 6** 

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
→ creates embeddings using OpenAI text-embedding-3-small and stores them persistently in `store/`.

### 5️⃣ Ask questions
Run **Notebook 3** (`3_answering_and_evaluation.ipynb`)  
→ queries Chroma and generates structured JSON answers using GPT-4o-mini.

If you made any changes to the notebook update the **App Backend**
```bash
  jupyter nbconvert --to script notebooks/3_answer_generation.ipynb --output "app_backend" --output-dir=src --TagRemovePreprocessor.enabled=True --TagRemovePreprocessor.remove_cell_tags='["noexport"]'
```
> 👉 **Note** \
> Any code fields that shouldn't be exported into the backend should be tagged as `noexport`. Make sure the ones you do export are actually needed for the app backend.

### 6️⃣ Launch Streamlit app
Run the following command on your terminal
```bash
streamlit run src/app.py
```
The application's GUI should now be available under http://localhost:8501/

### 7️⃣ Deployment to Streamlit Cloud (optional)
Push to GitHub and deploy the app to Streamlit Cloud via your Streamlit account dashboard. \
Once deployed, the Cloud app gets automatically updated after every commit.
> 👉 **Note** \
> Add your `OAI_TOKEN` to Streamlit Secrets.

## 🤝 Notes for Collaborators
- Logs and JSON files are **git-ignored** — they're rebuilt locally.
- Secrets are **git-ignored** due to security concerns.
- The app uses OpenAI embeddings (`dimension = 1536`). Mixing embedding models requires re-indexing.
- If the error `ModuleNotFound` pops up, there's a dependency issue. Either there's a mismatch of package versions or a package isn't supported by the Streamlit Python version (3.13.9). 