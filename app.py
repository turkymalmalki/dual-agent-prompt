"""
app.py — Streamlit UI for Dual-Agent Prompt Engineering System
"""
import json
import os
import time
import streamlit as st
from langchain_core.messages import AIMessage

from graph import MAX_REVISIONS, run_pipeline
from utils.rag import extract_text_from_upload

st.set_page_config(page_title="Dual-Agent Engineer", page_icon="⚙️", layout="wide")

st.markdown("""
<style>
.stApp { background-color: #0d0f14; color: #e2e8f0; }
.bubble-engineer { padding: 1rem; border-radius: 10px; margin-bottom: 1rem; border-left: 4px solid #4f8ef7; background: rgba(79, 142, 247, 0.08); white-space: pre-wrap; }
.bubble-expert { padding: 1rem; border-radius: 10px; margin-bottom: 1rem; border-left: 4px solid #34d399; background: rgba(52, 211, 153, 0.08); white-space: pre-wrap; }
.layer-card { background: #141720; border: 1px solid #252a36; border-radius: 10px; padding: 1rem; margin-bottom: 0.75rem; white-space: pre-wrap; }
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("## ⚙️ DUAL-AGENT\n<div style='color:#64748b;font-size:0.8rem;'>Max 3 rounds · Claude 3.5 Sonnet</div><hr>", unsafe_allow_html=True)
    task = st.text_area("Task", placeholder="Enter task...")
    domain = st.text_input("Domain", placeholder="e.g. Legal Writing")
    use_rag = st.toggle("Use External Sources (RAG)", False)
    rag_context = ""
    if use_rag:
        uploaded_file = st.file_uploader("Upload Reference Document", type=["pdf", "docx", "txt"])
        if uploaded_file:
            with st.spinner("Extracting..."):
                rag_context = extract_text_from_upload(uploaded_file)
            if rag_context: st.success("Extracted text successfully.")
    
    st.markdown("<hr>", unsafe_allow_html=True)
    run_clicked = st.button("▶ Run Pipeline", type="primary", use_container_width=True, disabled=not (task and domain))
    
    with st.expander("🔑 API Key Override"):
        api_key = st.text_input("Anthropic API Key", type="password")
        if api_key: os.environ["ANTHROPIC_API_KEY"] = api_key

if run_clicked:
    if not os.getenv("ANTHROPIC_API_KEY"):
        st.error("ANTHROPIC_API_KEY is not set.")
        st.stop()
    
    with st.spinner("🔄 Running dual-agent pipeline..."):
        try:
            final_prompt, messages = run_pipeline(task.strip(), domain.strip(), use_rag, rag_context)
            
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("💬 Dialogue")
                for msg in messages:
                    if isinstance(msg, AIMessage) and msg.name in ("Engineer", "Expert"):
                        css = "bubble-engineer" if msg.name == "Engineer" else "bubble-expert"
                        st.markdown(f"<div class='{css}'><strong>{msg.name}</strong><br>{msg.content}</div>", unsafe_allow_html=True)
            
            with col2:
                st.subheader("🏆 7-Layer Prompt")
                for key in ["Role", "Context", "Constraints", "Structure", "Examples", "Vocabulary", "Quality_Metrics"]:
                    val = final_prompt.get(key, "")
                    if val: st.markdown(f"<div class='layer-card'><strong>{key}</strong><br>{val}</div>", unsafe_allow_html=True)
                
                json_str = json.dumps(final_prompt, indent=2)
                st.download_button("⬇ Download JSON", json_str, "prompt.json", "application/json")
        except Exception as e:
            st.error(f"Error: {str(e)}")
            else:
    # ── Empty state (Welcome Screen) ──
    st.markdown(
        """
        <div style="display: flex; flex-direction: column; align-items: center; justify-content: center; height: 60vh; text-align: center;">
            <div style="font-size:3.5rem;margin-bottom:1rem;">⚙️</div>
            <div style="font-size:1.4rem;font-weight:bold;margin-bottom:1rem;color:#e2e8f0;letter-spacing:0.05em;">
                Ready to Engineer
            </div>
            <div style="font-size:0.9rem;color:#94a3b8;line-height:1.8;max-width:500px;">
                1. Enter your <strong>API key</strong> in the sidebar.<br>
                2. Describe your <strong>task</strong>.<br>
                3. Select an <strong>expert domain</strong>.<br>
                4. Optionally upload a <strong>reference document</strong> for grounding.<br>
                5. Hit <strong style="color:#4f8ef7;">▶ Run Pipeline</strong>.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
