"""
Swiss Rental-Law Assistant (Streamlit App)
------------------------------------------

This app provides an interactive interface for querying Swiss rental law.
It uses a persistent Chroma vector store for semantic retrieval and a local
Ollama model for grounded answer generation.

Structure:
1. Imports and configuration
2. Logging setup
3. Database initialization
4. Answer generation helper
5. Streamlit UI
"""

import os
import logging
from pathlib import Path
import streamlit as st
import chromadb
from _3_answer_generation import answer_with_ollama


# ============================================================
# 1. Configuration
# ============================================================

OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3:8b")
OLLAMA_EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")

CHROMA_DIR = Path(__file__).resolve().parent.parent / "store"
CHROMA_COLLECTION = "swiss_private_rental_law"

TOP_K = 5
MAX_CTX_CHARS = 8000

st.set_page_config(page_title="Schweizer Mietrechts-Assistent", page_icon="⚖️", layout="centered")
st.title("⚖️ Schweizer Mietrechts-Assistent")
st.markdown("Der Schweizer Mietrechts-Assistent ist ein KI-gestütztes Tool, das Fragen zum **Schweizer Mietrecht** beantwortet. Die Antworten basieren auf juristischen Quellen, werden jedoch automatisch generiert und sind **keine** rechtliche Beratung. Sie dienen lediglich als Orientierungshilfe.")

# ============================================================
# 2. Logging Setup
# ============================================================

LOG_DIR = Path(__file__).resolve().parent / "logs"
LOG_DIR.mkdir(exist_ok=True)

LOG_FILE = LOG_DIR / "app.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("SwissRentalLawApp")
logger.info("Starting Swiss Rental-Law Assistant UI...")
logging.getLogger("torch").setLevel(logging.DEBUG)

# ============================================================
# 3. Database Connection
# ============================================================

try:
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    col = client.get_collection(CHROMA_COLLECTION)
    count = col.count()
    logger.info(f"Loaded Chroma collection '{CHROMA_COLLECTION}' "
                f"from '{CHROMA_DIR}' with {count} entries.")
except Exception as e:
    count = 0
    logger.exception(f"Failed to load Chroma collection '{CHROMA_COLLECTION}'")
    st.sidebar.error("Datenbank konnte nicht geladen werden. Siehe Logdatei für Details.")


# Streamlit sidebar info
#st.sidebar.markdown(f"**Database Path:** `{CHROMA_DIR}`")
#st.sidebar.markdown(f"**Collection:** `{CHROMA_COLLECTION}` — {count} entries")


# ============================================================
# 4. Helper Function: Generate Answer
# ============================================================

def generate_answer(question: str, perspective: str):
    """
    Retrieve relevant context, query the Ollama model,
    and return the formatted answer and sources.
    """
    try:
        logger.info(f"Generiere Antwort | Perspektive: {perspective} |"
                    f"Frage: {question[:80]}...")
        ans, steps, forms, references, hits = answer_with_ollama(
            question,
            perspective=perspective,
            k=TOP_K
        )

        if not references:
            logger.warning("No references found for query.")
            sources = "Keine Quellen gefunden."
        else:
            sources_list = sorted(set(
                f"- {r.get('law','?')} Art.{r.get('article','?')} - {r.get('source','?')}"
                for r in references
            ))
            sources = "\n" + "\n".join(sources_list)
            logger.info(f"Retrieved {len(sources_list)} source entries.")

        if not steps:
            logger.warning("No steps found for query.")
            steps_text = "Keine Schritte gefunden."
        else:
            steps_list = [f"{i+1}. {s}" for i, s in enumerate(steps)]
            steps_text = "\n" + "\n".join(steps_list)
            logger.info(f"Retrieved {len(steps_list)} steps entries.")

        if not forms:
            logger.warning("No forms found for query.")
            forms_text = "Keine Formulare gefunden."
        else:
            forms_list = [f"- {f}" for f in forms]
            forms_text = "\n" + "\n".join(forms_list)
            logger.info(f"Retrieved {len(forms_list)} forms entries.")

        return ans, steps_text, forms_text, sources

    except Exception as e:
        logger.exception("Error during answer generation.")
        raise


# ============================================================
# 5. Streamlit UI
# ============================================================

with st.form("query_form"):
    question = st.text_area(
        "Gib deine Rechtsfrage ein:",
        placeholder="Beispiel: Wie fechte ich eine Mietzinserhöhung an? Welches Formular ist nötig?",
        height=100
    )
    col1, col2 = st.columns(2)
    with col1:
        perspective = st.selectbox("Perspektive", ["Mieter:in", "Vermieter:in"])
    submitted = st.form_submit_button("Antwort generieren ⚙️")

if submitted and question.strip():
    st.info("Antwort wird generiert, bitte warten...")
    try:
        ans, steps, forms, sources = generate_answer(question, perspective)
        st.markdown("## 💡 **Antwort**")
        st.markdown(ans)
        st.markdown("#### ✅ **Schritte/Optionen**")
        st.markdown(steps)
        st.markdown("---")
        st.markdown("#### 🧾 **Formulare**")
        st.markdown(forms)
        st.markdown("---")
        st.markdown("#### 📚 **Quellen**")
        st.markdown(sources)
    except Exception as e:
        st.error("Ein unerwarteter Fehler ist aufgetreten. Bitte siehe Logdatei für Details.")
