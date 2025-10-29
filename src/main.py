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

st.set_page_config(page_title="Swiss Rental-Law Assistant", page_icon="⚖️", layout="centered")
st.title("⚖️ Swiss Rental-Law Assistant")
st.markdown("Ask questions about **Swiss rental law**.")

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
logging.getLogger("torch").setLevel(logging.ERROR)

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
    st.sidebar.error("Database could not be loaded. Check logs for details.")


# Streamlit sidebar info
#st.sidebar.markdown(f"**Database Path:** `{CHROMA_DIR}`")
#st.sidebar.markdown(f"**Collection:** `{CHROMA_COLLECTION}` — {count} entries")


# ============================================================
# 4. Helper Function: Generate Answer
# ============================================================

def generate_answer(question: str, perspective: str, language: str):
    """
    Retrieve relevant context, query the Ollama model,
    and return the formatted answer and sources.
    """
    try:
        logger.info(f"Generating answer | Perspective: {perspective} | "
                    f"Language: {language} | Query: {question[:80]}...")
        ans, hits = answer_with_ollama(
            question,
            perspective=perspective,
            language=language,
            k=TOP_K
        )

        if not hits:
            logger.warning("No sources found for query.")
            sources = "No sources found."
        else:
            sources_list = sorted(set(
                f"- {m.get('law', '?')} Art.{m.get('article', '?')} - {m.get('source', '?')}"
                for _, m, _ in hits
            ))
            sources = "\n" + "\n".join(sources_list)
            logger.info(f"Retrieved {len(hits)} source entries.")

        return ans, sources

    except Exception as e:
        logger.exception("Error during answer generation.")
        raise


# ============================================================
# 5. Streamlit UI
# ============================================================

with st.form("query_form"):
    question = st.text_area(
        "Enter your legal question:",
        placeholder="Example: Wie fechte ich eine Mietzinserhöhung an? Welches Formular ist nötig?",
        height=100
    )
    col1, col2 = st.columns(2)
    with col1:
        perspective = st.selectbox("Perspective", ["Tenant", "Landlord"])
    with col2:
        language = st.selectbox("Language", ["German", "English"])
    submitted = st.form_submit_button("Generate Answer ⚙️")

if submitted and question.strip():
    st.info("Generating response, please wait...")
    try:
        ans, sources = generate_answer(question, perspective, language)
        st.markdown("### 🧾 **Answer**")
        st.markdown(ans)
        st.markdown("---")
        st.markdown("### 📚 **Sources**")
        st.markdown(sources)
    except Exception as e:
        st.error("An unexpected error occurred. Please check the logs for details.")
