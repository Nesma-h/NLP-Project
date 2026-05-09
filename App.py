# ─────────────────────────────────────────────────────────────────────────────
# app.py  |  launch with:  streamlit run app.py
# ─────────────────────────────────────────────────────────────────────────────

import streamlit as st
import numpy as np
import nltk
from nltk.tokenize import sent_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer

# ── NLTK setup ────────────────────────────────────────────────────────────────
for pkg, path in [("punkt",     "tokenizers/punkt"),
                  ("punkt_tab", "tokenizers/punkt_tab"),
                  ("stopwords", "corpora/stopwords")]:
    try:
        nltk.data.find(path)
    except LookupError:
        nltk.download(pkg, quiet=True)

# ── BART model loader (cached — loads once per session) ───────────────────────
@st.cache_resource(show_spinner=False)
def load_bart():
    import torch
    from transformers import BartTokenizer, BartForConditionalGeneration
    model_name = "facebook/bart-large-cnn"
    tokenizer  = BartTokenizer.from_pretrained(model_name)
    model      = BartForConditionalGeneration.from_pretrained(model_name)
    model.eval()
    return tokenizer, model

# ── TF-IDF extractive summarizer ─────────────────────────────────────────────
def tfidf_summary(text: str, n_sentences: int = 3) -> str:
    sentences = sent_tokenize(text)
    if len(sentences) <= n_sentences:
        return text
    vectorizer   = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(sentences)
    scores       = np.array(tfidf_matrix.mean(axis=1)).flatten()
    top_indices  = sorted(np.argsort(scores)[-n_sentences:])
    return " ".join([sentences[i] for i in top_indices])

# ── BART abstractive summarizer ───────────────────────────────────────────────
def bart_summary(text: str, n_sentences: int = 3) -> str:
    import torch
    tokenizer, model = load_bart()
    max_out = max(60, n_sentences * 50)
    min_out = max(30, n_sentences * 25)
    inputs  = tokenizer(text, return_tensors="pt", max_length=1024, truncation=True)
    with torch.no_grad():
        ids = model.generate(
            inputs["input_ids"],
            max_length=max_out,
            min_length=min_out,
            num_beams=4,
            length_penalty=2.0,
            early_stopping=True,
        )
    return tokenizer.decode(ids[0], skip_special_tokens=True)

# ── Unified pipeline ─────────────────────────────────────────────────────────
def summarize_text(text: str, method: str = "bart", n_sentences: int = 3) -> str:
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Input text must be a non-empty string.")
    if method == "tfidf":
        return tfidf_summary(text, n_sentences)
    elif method == "bart":
        return bart_summary(text, n_sentences)
    else:
        raise ValueError(f"Unsupported method: {method}")

# ── Helper ────────────────────────────────────────────────────────────────────
def word_count(text: str) -> int:
    return len(text.split()) if text.strip() else 0

def safe_sent_count(text: str) -> int:
    """Safely count sentences, returns 0 for empty input."""
    if not text or not text.strip():
        return 0
    try:
        return len(sent_tokenize(text))
    except Exception:
        return 0

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Text Summarization",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700;900&family=Source+Sans+3:wght@300;400;600&display=swap');

html, body, [class*="css"] { font-family: 'Source Sans 3', sans-serif; }

/* Hide menu and footer but keep header transparent so the sidebar toggle remains visible */
#MainMenu { visibility: hidden; }
footer    { visibility: hidden; }
header    { background: transparent !important; box-shadow: none !important; }

/* Ensure the sidebar collapse/expand toggle is always accessible */
[data-testid="collapsedControl"] {
    visibility: visible !important;
    display: flex !important;
    opacity: 1 !important;
    pointer-events: auto !important;
}

.stApp { background: #0f0d0d; }

/* ── Header ── */
.app-header   { padding: 2.5rem 0 1.5rem 0; border-bottom: 2px solid #C0392B; margin-bottom: 2rem; }
.app-title    { font-family: 'Playfair Display', Georgia, serif; font-size: 2.8rem; font-weight: 900;
                color: #ffffff; letter-spacing: -0.5px; margin: 0; line-height: 1.1; }
.app-title span { color: #C0392B; }
.app-subtitle { font-family: 'Source Sans 3', sans-serif; font-size: 0.95rem; color: #888;
                margin-top: 0.4rem; font-weight: 300; letter-spacing: 0.5px; }

/* ── Sidebar ── */
[data-testid="stSidebar"] { background: #1a1414 !important; border-right: 1px solid #2a1a1a; }
[data-testid="stSidebar"] .stMarkdown h2 {
    font-family: 'Playfair Display', serif; color: #C0392B; font-size: 1.1rem;
    border-bottom: 1px solid #2a1a1a; padding-bottom: 0.5rem;
}
[data-testid="stSidebar"] label {
    color: #ccc !important; font-size: 0.85rem !important; font-weight: 600 !important;
    letter-spacing: 0.8px !important; text-transform: uppercase !important;
}
[data-testid="stSidebar"] .stRadio > div { gap: 0.5rem; }
[data-testid="stSidebar"] .stRadio label {
    text-transform: none !important; letter-spacing: 0 !important;
    font-size: 0.95rem !important; font-weight: 400 !important; color: #ddd !important;
}
[data-testid="stSidebar"] p { color: #666; font-size: 0.8rem; line-height: 1.5; }

/* ── Method card ── */
.method-card { background: #1e1515; border: 1px solid #2a1a1a; border-left: 3px solid #C0392B;
               border-radius: 4px; padding: 0.8rem 1rem; margin-top: 0.5rem; }
.method-card p { color: #aaa !important; font-size: 0.82rem !important;
                 margin: 0 !important; line-height: 1.5 !important; }

/* ── BART notice ── */
.bart-notice { background: #1a1110; border: 1px solid #3a2020; border-left: 3px solid #e67e22;
               border-radius: 4px; padding: 0.7rem 1rem; margin-top: 0.6rem; }
.bart-notice p { color: #aaa !important; font-size: 0.78rem !important;
                 margin: 0 !important; line-height: 1.5 !important; }

/* ── Text area ── */
.stTextArea textarea {
    background: #1a1414 !important; border: 1px solid #2a1a1a !important;
    border-radius: 4px !important; color: #e8e8e8 !important;
    font-family: 'Source Sans 3', sans-serif !important; font-size: 0.95rem !important;
    line-height: 1.7 !important; caret-color: #C0392B !important;
}
.stTextArea textarea:focus { border-color: #C0392B !important; box-shadow: 0 0 0 1px #C0392B !important; }
.stTextArea label {
    color: #aaa !important; font-size: 0.78rem !important; font-weight: 600 !important;
    letter-spacing: 1px !important; text-transform: uppercase !important;
}

/* ── Button ── */
.stButton > button {
    background: #C0392B !important; color: white !important; border: none !important;
    border-radius: 3px !important; font-family: 'Source Sans 3', sans-serif !important;
    font-size: 0.88rem !important; font-weight: 600 !important; letter-spacing: 1.5px !important;
    text-transform: uppercase !important; padding: 0.6rem 2rem !important;
    transition: background 0.2s ease !important; width: 100% !important;
}
.stButton > button:hover { background: #922B21 !important; }

/* ── Stats ── */
.stats-row { display: flex; gap: 1rem; margin-bottom: 1.5rem; }
.stat-box  { flex: 1; background: #1a1414; border: 1px solid #2a1a1a;
             border-radius: 4px; padding: 0.9rem 1.2rem; text-align: center; }
.stat-value { font-family: 'Playfair Display', serif; font-size: 1.8rem;
              font-weight: 700; color: #C0392B; line-height: 1; }
.stat-label { font-size: 0.72rem; color: #666; letter-spacing: 1px;
              text-transform: uppercase; margin-top: 0.3rem; }

/* ── Summary output ── */
.summary-box {
    background: #1a1414; border: 1px solid #2a1a1a; border-top: 3px solid #C0392B;
    border-radius: 4px; padding: 1.6rem 1.8rem; margin-top: 0.5rem;
}
.summary-header {
    font-family: 'Playfair Display', serif; font-size: 0.75rem; color: #C0392B;
    letter-spacing: 2px; text-transform: uppercase; margin-bottom: 1rem;
    display: flex; align-items: center; gap: 0.5rem;
}
.summary-text { color: #e0e0e0; font-size: 1rem; line-height: 1.85; font-weight: 300; }

/* ── Misc ── */
.section-label { font-size: 0.72rem; color: #666; letter-spacing: 1.5px;
                 text-transform: uppercase; margin-bottom: 0.5rem; font-weight: 600; }
.stAlert { background: #1e1515 !important; border: 1px solid #C0392B !important;
           color: #e0e0e0 !important; border-radius: 4px !important; }
</style>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="app-header">
    <div class="app-title">Text <span>Summarization</span></div>
    <div class="app-subtitle">Extractive &amp; Abstractive Approaches — TF-IDF · BART</div>
</div>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙ Configuration")

    st.markdown('<div class="section-label">Method</div>', unsafe_allow_html=True)
    method = st.radio(
        label="method",
        options=["bart", "tfidf"],
        index=0,
        format_func=lambda x: "🧠 Abstractive (BART)" if x == "bart" else "📌 Extractive (TF-IDF)",
        label_visibility="collapsed",
    )

    if method == "bart":
        st.markdown("""
        <div class="method-card"><p>
        Uses <strong style="color:#C0392B;">facebook/bart-large-cnn</strong>, a transformer
        fine-tuned on news summarization. Generates fluent, abstractive text that paraphrases
        the source document. Model is loaded once and cached for the session.
        </p></div>
        <div class="bart-notice"><p>
        ⏱ First run downloads ~1.6 GB model weights — subsequent runs are instant.
        </p></div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="method-card"><p>
        Ranks sentences by their mean TF-IDF weight and returns the top-N in original
        document order. Fast, lightweight, and reliable for keyword-rich texts.
        Requires no model download.
        </p></div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-label">Output Length</div>', unsafe_allow_html=True)
    n_sentences = st.slider(
        label="sentences",
        min_value=1,
        max_value=10,
        value=3,
        label_visibility="collapsed",
    )
    st.markdown(
        f'<p style="color:#555; font-size:0.8rem;">← {n_sentences} sentence{"s" if n_sentences > 1 else ""} in output</p>',
        unsafe_allow_html=True,
    )

    # FIX: Only pre-warm BART when the user actually selects BART method
    if method == "bart":
        with st.spinner("Loading BART model…"):
            load_bart()
        st.markdown('<p style="color:#3a7a3a; font-size:0.78rem;">✔ Model ready</p>', unsafe_allow_html=True)

    st.markdown("<br><hr style='border-color:#1e1515'>", unsafe_allow_html=True)
    st.markdown(
        '<p style="color:#444; font-size:0.75rem; text-align:center;">Text Summarization Project<br>TF-IDF · BART</p>',
        unsafe_allow_html=True,
    )

# ── Main layout ───────────────────────────────────────────────────────────────
col_input, col_output = st.columns([1, 1], gap="large")

with col_input:
    st.markdown('<div class="section-label">Input Document</div>', unsafe_allow_html=True)
    input_text = st.text_area(
        label="input",
        height=320,
        placeholder="Paste your article, document, or any text here…",
        label_visibility="collapsed",
    )

    wc = word_count(input_text)
    # FIX: use safe_sent_count to avoid tokenizing empty strings
    sc = safe_sent_count(input_text)

    st.markdown(f"""
    <div class="stats-row">
        <div class="stat-box">
            <div class="stat-value">{wc:,}</div>
            <div class="stat-label">Words</div>
        </div>
        <div class="stat-box">
            <div class="stat-value">{sc}</div>
            <div class="stat-label">Sentences</div>
        </div>
        <div class="stat-box">
            <div class="stat-value">{len(input_text):,}</div>
            <div class="stat-label">Characters</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    generate = st.button("▶  Generate Summary")

with col_output:
    st.markdown('<div class="section-label">Summary Output</div>', unsafe_allow_html=True)

    if generate:
        if not input_text.strip():
            st.warning("Please enter some text before generating a summary.")
        else:
            # FIX: warn user if input is too short for the requested summary length
            if sc > 0 and sc <= n_sentences and method == "tfidf":
                st.warning(
                    f"Input only has {sc} sentence{'s' if sc > 1 else ''}. "
                    f"Returning the full text as the summary."
                )

            spinner_msg = "Generating abstractive summary with BART…" if method == "bart" \
                          else "Scoring sentences with TF-IDF…"
            with st.spinner(spinner_msg):
                try:
                    result = summarize_text(input_text, method=method, n_sentences=n_sentences)

                    result_wc   = word_count(result)
                    # FIX: guard against division by zero when wc is 0
                    compression = round((1 - result_wc / wc) * 100) if wc > 0 else 0
                    label       = "BART" if method == "bart" else "TF-IDF"

                    st.markdown(f"""
                    <div class="stats-row">
                        <div class="stat-box">
                            <div class="stat-value">{result_wc:,}</div>
                            <div class="stat-label">Words</div>
                        </div>
                        <div class="stat-box">
                            <div class="stat-value">{n_sentences}</div>
                            <div class="stat-label">Sentences</div>
                        </div>
                        <div class="stat-box">
                            <div class="stat-value">{compression}%</div>
                            <div class="stat-label">Compressed</div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                    # FIX: escape HTML special characters in result to prevent XSS / rendering bugs
                    import html as html_module
                    safe_result = html_module.escape(result)

                    st.markdown(f"""
                    <div class="summary-box">
                        <div class="summary-header">● {label} Summary</div>
                        <div class="summary-text">{safe_result}</div>
                    </div>
                    """, unsafe_allow_html=True)

                except Exception as e:
                    st.error(f"Error generating summary: {e}")
    else:
        st.markdown("""
        <div class="summary-box" style="min-height:320px; display:flex; align-items:center; justify-content:center;">
            <div style="text-align:center; color:#333;">
                <div style="font-size:2.5rem; margin-bottom:0.8rem;">◈</div>
                <div style="font-size:0.8rem; letter-spacing:2px; text-transform:uppercase; color:#444;">
                    Awaiting input
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)