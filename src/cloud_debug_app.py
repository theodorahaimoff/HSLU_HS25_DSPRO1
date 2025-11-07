import os
import streamlit as st
from openai import OpenAI

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
import sys, platform
st.write("Python:", sys.version)
st.write("Platform:", platform.platform())

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
