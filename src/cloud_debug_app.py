import os, json
import streamlit as st
from openai import OpenAI
import chromadb
from pathlib import Path
import sys, subprocess, platform

st.set_page_config(page_title="Cloud Debug", page_icon="🛠️")

# Secrets + env print
def _mask(t):
    return t[:4] + "…" + t[-4:] if t and len(t) > 12 else "(unset)"

OAI = (os.getenv("OAI_TOKEN") or
       (dict(st.secrets).get("env", {}).get("OAI_TOKEN") if st.secrets else ""))

st.write("✅ Streamlit running")
st.write("OpenAI token present:", bool(OAI))
st.write("OpenAI token (masked):", _mask(OAI))

# Show runtime info
st.write("Python:", sys.version)
st.write("Platform:", platform.platform())

result = subprocess.run(
    [sys.executable, "-m", "pip", "list"],
    capture_output=True,
    text=True
)

with st.expander("PIP packages:"):
    st.code(result.stdout, language=None)


# Try a minimal OpenAI ping (no JSON mode to reduce failure surface)
if st.button("Ping OpenAI"):
    try:
        client = OpenAI(api_key=OAI) if OAI else None
        if not client:
            st.error("No OAI token.")
        else:
            r = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": "pong"}],
                temperature=0
            )
            st.success("OpenAI reachable ✅")
            st.code(r.choices[0].message.content)
    except Exception as e:
        st.exception(e)


store_dir = Path().parent.resolve() / "store"
mf = json.loads((store_dir / "manifest.json").read_text(encoding="utf-8"))

MODEL_NAME = mf["model"]
COLLECTION_NAME = mf["collection"]
EXPECTED_DIM = mf["dim"]
DIR = mf["dir"]
COLLECTION_PATH = store_dir / DIR
st.write("Chroma manifest present:", bool(mf))
st.write("Chroma collection name:", COLLECTION_NAME)
st.write("Chroma path:", DIR)
st.write("Store path:", store_dir)
st.write("Chroma collection path:", COLLECTION_PATH)


def get_collection(name=COLLECTION_NAME):
    client = chromadb.PersistentClient(path=str(COLLECTION_PATH))
    return client.get_collection(name)

COLLECTION = get_collection()


st.write("Chroma collection has value:", COLLECTION.count())

#embed = client.embeddings.create(model=MODEL_NAME, input="Kündigungsfrist")
#st.write("Embedding present:", embed)