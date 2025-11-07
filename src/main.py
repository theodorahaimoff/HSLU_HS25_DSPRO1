"""
Swiss Rental-Law Assistant (Streamlit App)
------------------------------------------

This app provides an interactive interface for querying Swiss rental law.
It uses a persistent Chroma vector store for semantic retrieval and the
Hugging Face Inference API for grounded answer generation.
"""

import re, logging
from pathlib import Path
from typing import Iterable, Tuple
import streamlit as st
from backend import (
    TOP_K, _collection_name,
    get_collection,
    answer_with_openai
)

# ============================================================
# Logging
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
    ],
    force=True
)

logger = logging.getLogger("SwissRentalLawApp")
logger.info("Starting Swiss Rental-Law Assistant...")
logging.getLogger("torch").setLevel(logging.INFO)
logging.getLogger("chromadb").setLevel(logging.DEBUG)

# ============================================================
# Configuration
# ============================================================
st.set_page_config(
    page_title="Schweizer Mietrechts-Assistent",
    page_icon="⚖️",
    layout="centered"
)
st.title("⚖️ Schweizer Mietrechts-Assistent")
st.markdown(
    "Der Schweizer Mietrechts-Assistent ist ein KI-gestütztes Tool, das Fragen zum "
    "**Schweizer Mietrecht** beantwortet. Die Antworten basieren auf juristischen Quellen, "
    "werden jedoch automatisch generiert und sind **keine** rechtliche Beratung. "
    "Sie dienen lediglich als Orientierungshilfe."
)

# ============================================================
# Database Connection
# ============================================================
try:
    name = _collection_name()
    col = get_collection(name)
    logger.info(f"Loaded Chroma collection '{name}' "
                f"with {col.count()} entries.")
except Exception:
    logger.exception(f"Failed to load Chroma collection '{name}'")
    st.sidebar.error("Datenbank konnte nicht geladen werden. Siehe Logdatei für Details.")


# ============================================================
# Markdown Helper Functions
# ============================================================

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

# ============================================================
# Answer Generation
# ============================================================

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


# ============================================================
# Streamlit UI
# ============================================================
with st.form("query_form"):
    question = st.text_area(
        "Gib deine Rechtsfrage ein:",
        placeholder="Beispiel: Wie fechte ich eine Mietzinserhöhung an? Welches Formular ist nötig?",
        height=100
    )
    perspective = st.selectbox("Perspektive", ["Mieter:in", "Vermieter:in"])
    submitted = st.form_submit_button("Antwort generieren ⚙️")

if submitted and question.strip():
    with st.spinner("Antwort wird generiert, bitte warten..."):
        try:
            ans, steps, forms, sources = generate_answer(question, perspective)
            logger.debug(f"Answer generated successfully for question: {question[:60]}")

            with st.container(border=True):
                st.header("💡 Antwort")
                st.markdown(ans)
                st.subheader("✅ Schritte/Optionen")
                st.markdown(steps)
                st.subheader("📄 Formulare")
                st.markdown(forms)
                st.subheader("📚 Quellen")
                st.markdown(sources)

        except Exception:
            st.error("Ein unerwarteter Fehler ist aufgetreten. Bitte siehe Logdatei für Details.")
            st.stop()