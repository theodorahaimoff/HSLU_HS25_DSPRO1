"""
Swiss Rental-Law Assistant (Streamlit App)
------------------------------------------

This app provides an interactive interface for querying Swiss rental law.
It uses a persistent Chroma vector store for semantic retrieval and the
Hugging Face Inference API for grounded answer generation.
"""

from backend import *

# ============================================================
# Logging
# ============================================================

LOG_DIR = Path().resolve() / "logs"
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