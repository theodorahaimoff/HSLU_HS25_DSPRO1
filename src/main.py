"""
Interactive Streamlit interface for the Swiss rental-law assistant.

This app uses the existing retrieval and answer generation pipeline
to provide a simple chat-style GUI for testing and demonstration.

Structure:
1. Imports and setup
2. Vector database initialization
3. Helper function for response generation
4. Streamlit UI components
5. Response display
"""

# 1. Imports and setup
import os
import streamlit as st
from pathlib import Path
import chromadb
from _3_answer_generation import answer_with_ollama

# Configuration
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3:8b")
OLLAMA_EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")

CHROMA_DIR = Path(__file__).resolve().parent.parent / "store"
CHROMA_COLLECTION = "swiss_private_rental_law"

TOP_K = 5
MAX_CTX_CHARS = 8000

# Streamlit page settings
st.set_page_config(
    page_title="Swiss Rental-Law Assistant",
    page_icon="⚖️",
    layout="centered",
)
st.title("⚖️ Swiss Rental-Law Assistant")
st.markdown("Ask questions about **Swiss rental law**.")

# 2. Connect to the persistent Chroma store
try:
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    col = client.get_collection(CHROMA_COLLECTION)
    count = col.count()
except Exception as e:
    count = 0
    print("⚠️ Could not load Chroma collection:", e)

# DB Debugging information
st.sidebar.markdown(f"**DB Path:** `{CHROMA_DIR}`")
st.sidebar.markdown(f"**Collection:** `{CHROMA_COLLECTION}` — {count} entries")
print(f"Collection: {CHROMA_COLLECTION} | count: {count}")

# 3. Helper function
def generate_answer(question, perspective, language):
    """
    Retrieve relevant context, query the Ollama model, and return the formatted answer.
    """
    ans, hits = answer_with_ollama(
        question,
        perspective=perspective,
        language=language,
        k=TOP_K
    )
    print("HITS:", hits)

    if not hits:
        sources = "No sources found."
    else:
        sources_list = sorted(set(
            f"- {m.get('law', '?')} Art.{m.get('article', '?')} - {m.get('source', '?')}"
            for _, m, _ in hits
        ))
        sources = "\n" + "\n".join(sources_list)

    return ans, sources


# 4. Streamlit UI components
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

# 4. Response display
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
        st.error(f"An error occurred while generating the answer: {e}")
