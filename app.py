"""
app.py
------
Dual-Agent Prompt Engineering System — Streamlit UI.

Run locally:
    streamlit run app.py

Deploy to Streamlit Cloud:
    • Push all files to a public GitHub repo.
    • Add OPENAI_API_KEY to Settings → Secrets.
"""

from __future__ import annotations

import json
import os
import time
from typing import Dict, List, Optional

import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import BaseMessage

load_dotenv()

# ---------------------------------------------------------------------------
# Page config — must be the very first Streamlit call
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Dual-Agent Prompt Engineer",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS — dark, editorial, monospace-accented aesthetic
# ---------------------------------------------------------------------------

st.markdown(
    """
<style>
/* ── Google Fonts ── */
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600&family=Syne:wght@400;700;800&display=swap');

/* ── Global reset & base ── */
html, body, [class*="css"] {
    font-family: 'Syne', sans-serif;
    background-color: #0d0f14;
    color: #e2e8f0;
}

/* ── Hide Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }

/* ── App container ── */
.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
    max-width: 1400px;
}

/* ── Hero title ── */
.hero-title {
    font-size: 2.6rem;
    font-weight: 800;
    letter-spacing: -1.5px;
    background: linear-gradient(120deg, #7ee8fa 0%, #80ff72 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 0.2rem;
    line-height: 1.1;
}
.hero-sub {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.78rem;
    color: #4a5568;
    letter-spacing: 2px;
    text-transform: uppercase;
    margin-bottom: 2rem;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background-color: #111318;
    border-right: 1px solid #1e2330;
}
[data-testid="stSidebar"] .block-container { padding-top: 1.5rem; }

.sidebar-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.65rem;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: #4a5568;
    margin-bottom: 4px;
}

/* ── Input widgets ── */
textarea, input[type="text"], .stSelectbox > div {
    background-color: #161b27 !important;
    border: 1px solid #2d3748 !important;
    border-radius: 8px !important;
    color: #e2e8f0 !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.82rem !important;
}
textarea:focus, input[type="text"]:focus {
    border-color: #7ee8fa !important;
    box-shadow: 0 0 0 2px rgba(126,232,250,0.15) !important;
}

/* ── Primary button ── */
.stButton > button[kind="primary"], div.stButton > button {
    background: linear-gradient(135deg, #7ee8fa 0%, #80ff72 100%);
    color: #0d0f14;
    font-family: 'Syne', sans-serif;
    font-weight: 700;
    font-size: 0.9rem;
    letter-spacing: 0.5px;
    border: none;
    border-radius: 8px;
    padding: 0.55rem 1.6rem;
    width: 100%;
    cursor: pointer;
    transition: opacity 0.2s, transform 0.1s;
}
div.stButton > button:hover { opacity: 0.9; transform: translateY(-1px); }
div.stButton > button:active { transform: translateY(0); }

/* ── Download button ── */
div.stDownloadButton > button {
    background: #1a2035;
    color: #7ee8fa;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.78rem;
    border: 1px solid #2d3748;
    border-radius: 8px;
    padding: 0.45rem 1.2rem;
    width: 100%;
    transition: border-color 0.2s;
}
div.stDownloadButton > button:hover { border-color: #7ee8fa; }

/* ── Chat bubbles ── */
.bubble-wrap { display: flex; margin-bottom: 1rem; }
.bubble-wrap.engineer { flex-direction: row; }
.bubble-wrap.expert   { flex-direction: row-reverse; }

.bubble-avatar {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.62rem;
    font-weight: 600;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    padding: 0 0.6rem;
    writing-mode: vertical-rl;
    text-orientation: mixed;
    color: #4a5568;
    flex-shrink: 0;
}

.bubble {
    border-radius: 12px;
    padding: 0.9rem 1.1rem;
    font-size: 0.84rem;
    line-height: 1.65;
    max-width: 90%;
}
.bubble.engineer {
    background: #151c2c;
    border: 1px solid #1e3a5f;
    color: #a8d8ea;
    border-top-left-radius: 4px;
    margin-left: 4px;
}
.bubble.expert {
    background: #141f1a;
    border: 1px solid #1a3d2e;
    color: #a8e6cf;
    border-top-right-radius: 4px;
    margin-right: 4px;
}
.bubble-name {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.65rem;
    letter-spacing: 1.5px;
    font-weight: 600;
    text-transform: uppercase;
    margin-bottom: 6px;
}
.bubble.engineer .bubble-name { color: #7ee8fa; }
.bubble.expert   .bubble-name { color: #80ff72; }

/* ── 7-Layer cards ── */
.layer-card {
    background: #111318;
    border: 1px solid #1e2330;
    border-radius: 12px;
    padding: 1rem 1.2rem;
    margin-bottom: 0.8rem;
    position: relative;
    transition: border-color 0.2s;
}
.layer-card:hover { border-color: #2d3748; }

.layer-badge {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.6rem;
    font-weight: 600;
    letter-spacing: 2px;
    text-transform: uppercase;
    padding: 3px 10px;
    border-radius: 20px;
    display: inline-block;
    margin-bottom: 0.5rem;
}
.layer-content {
    font-size: 0.83rem;
    line-height: 1.7;
    color: #cbd5e0;
}

/* Layer badge colors */
.badge-role       { background: #1a1a3e; color: #9b8ff7; border: 1px solid #3b357a; }
.badge-context    { background: #1a2c1a; color: #80ff72; border: 1px solid #2d5a2d; }
.badge-constraints{ background: #2c1a1a; color: #ff9f7f; border: 1px solid #5a2d2d; }
.badge-structure  { background: #1a252c; color: #7ee8fa; border: 1px solid #2d4a5a; }
.badge-examples   { background: #2c2a1a; color: #ffd97d; border: 1px solid #5a502d; }
.badge-vocabulary { background: #1a2c2c; color: #7afcf0; border: 1px solid #2d5a58; }
.badge-quality    { background: #2c1a2c; color: #f07af4; border: 1px solid #5a2d5a; }

/* ── Divider ── */
.section-divider {
    border: none;
    border-top: 1px solid #1e2330;
    margin: 1.5rem 0;
}

/* ── Status / info box ── */
.status-box {
    background: #0e1520;
    border: 1px solid #1e3a5f;
    border-radius: 8px;
    padding: 0.8rem 1rem;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.78rem;
    color: #7ee8fa;
    display: flex;
    align-items: center;
    gap: 8px;
}

/* ── Column headers ── */
.col-header {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.65rem;
    letter-spacing: 2.5px;
    text-transform: uppercase;
    color: #4a5568;
    margin-bottom: 1rem;
    padding-bottom: 0.5rem;
    border-bottom: 1px solid #1e2330;
}
</style>
""",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Hero header
# ---------------------------------------------------------------------------

st.markdown(
    '<div class="hero-title">Dual-Agent Prompt Engineer</div>'
    '<div class="hero-sub">// Engineer × Expert × LangGraph × GPT-4o-mini</div>',
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Sidebar — inputs
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown('<div class="sidebar-label">Configuration</div>', unsafe_allow_html=True)
    st.markdown("---")

    openai_key = st.text_input(
        "OpenAI API Key",
        type="password",
        value=os.getenv("OPENAI_API_KEY", ""),
        help="Your key is used only during this session and never stored.",
        placeholder="sk-...",
    )
    if openai_key:
        os.environ["OPENAI_API_KEY"] = openai_key

    st.markdown('<div class="sidebar-label" style="margin-top:1.2rem">Task</div>', unsafe_allow_html=True)
    task = st.text_area(
        label="task_input",
        label_visibility="collapsed",
        placeholder="Describe the task you want a world-class prompt for…",
        height=130,
        key="task_input",
    )

    st.markdown('<div class="sidebar-label" style="margin-top:1rem">Expert Domain</div>', unsafe_allow_html=True)
    domain = st.selectbox(
        label="domain_select",
        label_visibility="collapsed",
        options=[
            "General",
            "Medical",
            "Legal",
            "Software Engineering",
            "Marketing",
            "Finance",
            "Education",
            "Scientific Research",
            "Creative Writing",
            "Data Science",
            "Cybersecurity",
            "Product Management",
        ],
        key="domain_select",
    )

    st.markdown('<div class="sidebar-label" style="margin-top:1rem">Reference Document (RAG)</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        label="rag_upload",
        label_visibility="collapsed",
        type=["pdf", "docx", "txt"],
        help="Optional: upload a PDF, DOCX, or TXT to ground the agents' knowledge.",
        key="rag_upload",
    )

    st.markdown("<br>", unsafe_allow_html=True)
    run_button = st.button("⚙ Run Pipeline", use_container_width=True)

# ---------------------------------------------------------------------------
# Helper: render chat bubbles
# ---------------------------------------------------------------------------

_LAYER_META: Dict[str, Dict[str, str]] = {
    "Role":            {"badge": "badge-role",        "emoji": "🎭"},
    "Context":         {"badge": "badge-context",     "emoji": "🌐"},
    "Constraints":     {"badge": "badge-constraints", "emoji": "🚧"},
    "Structure":       {"badge": "badge-structure",   "emoji": "📐"},
    "Examples":        {"badge": "badge-examples",    "emoji": "💡"},
    "Vocabulary":      {"badge": "badge-vocabulary",  "emoji": "📖"},
    "Quality_Metrics": {"badge": "badge-quality",     "emoji": "✅"},
}


def _render_chat(messages: List[BaseMessage]) -> None:
    for msg in messages:
        name = getattr(msg, "name", "") or ""
        role_class = "engineer" if name == "Engineer" else "expert"
        display_name = "Prompt Engineer" if name == "Engineer" else f"{domain} Expert"

        content_html = (
            msg.content
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace("\n", "<br>")
            .replace("**", "<b>", 1)
        )
        # naive bold: replace ** pairs
        import re
        content_html = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", msg.content)
        content_html = content_html.replace("\n", "<br>")

        st.markdown(
            f"""
            <div class="bubble-wrap {role_class}">
                <div class="bubble {role_class}">
                    <div class="bubble-name">{display_name}</div>
                    {content_html}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def _render_layers(layers: Dict[str, str]) -> None:
    for layer_key in [
        "Role", "Context", "Constraints", "Structure",
        "Examples", "Vocabulary", "Quality_Metrics",
    ]:
        value = layers.get(layer_key, "—")
        meta = _LAYER_META.get(layer_key, {"badge": "", "emoji": "•"})
        display_name = layer_key.replace("_", " ")

        value_html = value.replace("\n", "<br>")

        st.markdown(
            f"""
            <div class="layer-card">
                <span class="layer-badge {meta['badge']}">{meta['emoji']} {display_name}</span>
                <div class="layer-content">{value_html}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


# ---------------------------------------------------------------------------
# Main execution block
# ---------------------------------------------------------------------------

if run_button:
    # ── Validation ──────────────────────────────────────────────────────────
    if not os.getenv("OPENAI_API_KEY"):
        st.error("🔑 Please enter your OpenAI API Key in the sidebar.")
        st.stop()

    if not task.strip():
        st.warning("✏️ Please describe a task before running the pipeline.")
        st.stop()

    # ── RAG extraction ───────────────────────────────────────────────────────
    rag_context: Optional[str] = None
    if uploaded_file is not None:
        from utils.rag import extract_text
        with st.spinner("📄 Extracting document text…"):
            rag_context = extract_text(uploaded_file)
        if rag_context:
            st.success(
                f"✅ Extracted {len(rag_context):,} characters from **{uploaded_file.name}**"
            )

    # ── Pipeline ─────────────────────────────────────────────────────────────
    from graph import run_pipeline  # import here to avoid circular issues at startup

    with st.spinner("🤖 Agents are collaborating… (3 rounds × 2 agents)"):
        t0 = time.time()
        try:
            result = run_pipeline(
                task=task.strip(),
                domain=domain,
                rag_context=rag_context,
            )
        except Exception as exc:
            st.error(f"Pipeline failed: {exc}")
            st.stop()
        elapsed = time.time() - t0

    st.markdown(
        f'<div class="status-box">✓ Pipeline complete in {elapsed:.1f}s &nbsp;·&nbsp; '
        f'{len(result.messages)} messages &nbsp;·&nbsp; {MAX_REVISIONS} rounds</div>',
        unsafe_allow_html=True,
    )
    st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)

    # ── Two-column results layout ─────────────────────────────────────────────
    col_chat, col_prompt = st.columns([1, 1], gap="large")

    with col_chat:
        st.markdown('<div class="col-header">// Agent Dialogue</div>', unsafe_allow_html=True)
        _render_chat(result.messages)

    with col_prompt:
        st.markdown('<div class="col-header">// 7-Layer Extracted Prompt</div>', unsafe_allow_html=True)

        if result.final_prompt:
            _render_layers(result.final_prompt)

            st.markdown("<br>", unsafe_allow_html=True)

            json_bytes = json.dumps(result.final_prompt, indent=2, ensure_ascii=False).encode("utf-8")
            st.download_button(
                label="⬇ Download JSON",
                data=json_bytes,
                file_name="7_layer_prompt.json",
                mime="application/json",
                use_container_width=True,
            )
        else:
            st.warning("The extraction step did not return a structured prompt. Check the chat on the left for the Engineer's final draft.")

# ---------------------------------------------------------------------------
# Empty state — show instructions
# ---------------------------------------------------------------------------

else:
    st.markdown(
        """
        <div style="
            margin-top: 3rem;
            padding: 2.5rem;
            background: #0e1118;
            border: 1px dashed #1e2330;
            border-radius: 16px;
            text-align: center;
        ">
            <div style="font-size:2.5rem; margin-bottom:1rem">⚙️</div>
            <div style="font-size:1.1rem; font-weight:700; color:#e2e8f0; margin-bottom:0.5rem">
                Ready to Engineer
            </div>
            <div style="
                font-family: 'JetBrains Mono', monospace;
                font-size: 0.78rem;
                color: #4a5568;
                line-height: 1.9;
            ">
                1. Enter your <b style="color:#7ee8fa">OpenAI API key</b> in the sidebar<br>
                2. Describe your <b style="color:#80ff72">task</b> (the more specific, the better)<br>
                3. Select an <b style="color:#ffd97d">expert domain</b><br>
                4. Optionally upload a <b style="color:#f07af4">reference document</b> for grounding<br>
                5. Hit <b style="color:#7afcf0">⚙ Run Pipeline</b>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── How it works teaser ──────────────────────────────────────────────────
    c1, c2, c3 = st.columns(3)
    cards = [
        ("🤖 Prompt Engineer", "#1e3a5f", "#7ee8fa",
         "Drives the dialogue through 3 structured rounds: Basics → Deep Dive → Final Draft."),
        ("🧠 Domain Expert", "#1a3d2e", "#80ff72",
         "Answers questions with domain authority, optionally grounded in your uploaded document."),
        ("🔬 Extraction Node", "#2c1a3d", "#9b8ff7",
         "Uses structured output to atomise the draft into 7 clean, downloadable JSON layers."),
    ]
    for col, (title, bg, color, desc) in zip([c1, c2, c3], cards):
        with col:
            st.markdown(
                f"""
                <div style="
                    background:{bg}22;
                    border:1px solid {bg};
                    border-radius:12px;
                    padding:1.2rem;
                    height:100%;
                ">
                    <div style="font-size:0.9rem;font-weight:700;color:{color};margin-bottom:0.5rem">{title}</div>
                    <div style="font-size:0.8rem;color:#718096;line-height:1.6">{desc}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
