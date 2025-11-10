"""
Swiss Rental-Law Assistant (Streamlit App)
------------------------------------------

This app provides an interactive interface for querying Swiss rental law.
It uses a persistent Chroma vector store for semantic retrieval and the
OpenAI API for grounded answer generation.
"""

import logging
from pathlib import Path
import streamlit as st
from app_backend import generate_answer

# ============================================================
# Logging
# ============================================================

LOG_DIR = Path().resolve()  / "logs"
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

css = """
.st-key-answer-container {
    background-color: #EAE8FF;
}
"""
st.html(f"<style>{css}</style>")

# ============================================================
# Configuration
# ============================================================
st.set_page_config(
    page_title="rently — dein digitaler Mietrechtsassistent 🇨🇭",
    page_icon="img/logo.png",
    layout="centered"
)
logo, title = st.columns(spec=[0.1,0.9], gap="small", vertical_alignment="bottom", width=700)
with logo:
    st.image("img/logo.png", width=70)
with title:
    st.title("rently")
st.header("Dein digitaler Mietrechtsassistent")
st.markdown(
    "Mit *rently* bekommst du schnelle, präzise Antworten zu deinen Fragen rund um das Schweizer Mietrecht. \n \n " 
    "Ob Mietzinserhöhung, Kündigung oder Nebenkosten – *rently* durchsucht die relevanten Gesetze (OR, VMWG, StGB) und liefert dir klare Schritte, passende Formulare und rechtliche Grundlagen."
)

# ============================================================
# Streamlit UI
# ============================================================
with st.form("query_form"):
    question = st.text_area(
        label="Gib deine Rechtsfrage ein:",
        placeholder="Beispiel: Wie fechte ich eine Mietzinserhöhung an? Welches Formular ist nötig?",
        height=100
    )
    perspective = st.selectbox(label="Perspektive", options=["Mieter:in", "Vermieter:in"])
    submitted = st.form_submit_button(label="Antwort generieren", type="primary")


if submitted and question.strip():
    with st.spinner("Antwort wird generiert, bitte warten..."):
        try:
            ans, steps, forms, sources = generate_answer(question, perspective)
            logger.debug(f"Answer generated successfully for question: {question[:60]}")

            with st.container(border=True, key="answer-container"):
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

st.caption("*rently basiert auf KI und ersetzt keine Rechtsberatung.*")