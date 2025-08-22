# app.py
# ------------------------------------------------------------
# JVAI Financial Policy Chatbot — Notebook-faithful Streamlit app
# Pipeline: Extraction → Chunking → Embedding+FAISS → Hybrid Search
#           → Strict Post-processing → Minimal Memory (topic heuristic)
# ------------------------------------------------------------

import os, re, io, random
from collections import Counter, deque
from typing import List, Dict, Tuple

import numpy as np
import streamlit as st
import pdfplumber
import faiss
from sentence_transformers import SentenceTransformer

# -----------------------------
# Streamlit config (must be first Streamlit call)
# -----------------------------
st.set_page_config(page_title="JVAI Financial Policy Chatbot", page_icon="💬", layout="wide")

# -----------------------------
# Config
# -----------------------------
PDF_PATH = "data/For Task - Policy file.pdf"   # <- your bundled policy PDF
TOP_K = 5                                      # default retrieval depth (matches notebook)
MAX_CHARS_PER_CHUNK = 600                      # chunk size consistent with notebook

# -----------------------------
# Utilities
# -----------------------------
def set_seed(s=42):
    random.seed(s); np.random.seed(s)
set_seed(42)

# -----------------------------
# Extraction (path → pages list[{page, text}])
# -----------------------------
def extract_pages(pdf_path: str) -> List[Dict]:
    if not os.path.exists(pdf_path):
        st.error(f"PDF not found at '{pdf_path}'. Place your file there or update PDF_PATH.")
        st.stop()

    docs = []
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            text = page.extract_text() or ""
            # normalize spaces; keep newlines for conservative paragraph breaks
            text = re.sub(r'[ \t]+', ' ', text)
            text = text.strip()
            if text:
                docs.append({"page": i, "text": text})
    return docs

# -----------------------------
# Chunking (paragraphs → sentence-batched ≤ MAX_CHARS_PER_CHUNK)
# -----------------------------
HEADING_RX = re.compile(r'^\s*(Table\s+\d+\.\d+(?:\.\d+)?|[A-Z][A-Z \-]{6,}|[0-9]+\.[0-9]+.*)$')

def split_into_paragraphs(text: str) -> List[str]:
    # conservative split on blank lines
    return [p.strip() for p in re.split(r'\n{2,}', text) if p.strip()]

def detect_section(paras: List[str]) -> str:
    for p in paras[:3]:
        if HEADING_RX.match(p):
            return p[:120]
    return ""

def fine_chunks(pages: List[Dict], max_chars: int = MAX_CHARS_PER_CHUNK) -> List[Dict]:
    chunks = []
    for p in pages:
        paras = split_into_paragraphs(p["text"])
        section = detect_section(paras)
        for para in paras:
            if len(para) > max_chars:
                # sentence-batch long paragraphs
                sentences = re.split(r'(?<=[.!?])\s+', para)
                buf = ""
                for s in sentences:
                    if len(buf) + 1 + len(s) <= max_chars:
                        buf = (buf + " " + s).strip()
                    else:
                        if buf:
                            chunks.append({"text": buf.strip(), "page": p["page"], "section": section})
                        buf = s
                if buf:
                    chunks.append({"text": buf.strip(), "page": p["page"], "section": section})
            else:
                chunks.append({"text": para.strip(), "page": p["page"], "section": section})
    return chunks

# -----------------------------
# Embedding & FAISS
# -----------------------------
@st.cache_resource(show_spinner=False)
def load_model():
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

@st.cache_resource(show_spinner=False)
def build_index(chunks: List[Dict]):
    model = load_model()
    texts = [c["text"] for c in chunks]
    emb = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
    faiss.normalize_L2(emb)
    index = faiss.IndexFlatIP(emb.shape[1])
    index.add(emb)
    return model, index, emb

# -----------------------------
# Hybrid search (semantic + light keyword bonus)
# -----------------------------
FIN_TERMS = [
    "debt","tax","taxation","gsp","gross state product","net assets",
    "superannuation","credit rating","balanced budget","infrastructure",
    "net interest","interest expense","interest revenue","capital works"
]

def hybrid_search(query: str, k: int, chunks: List[Dict], index: faiss.IndexFlatIP, model) -> List[Dict]:
    qv = model.encode([query], convert_to_numpy=True)
    faiss.normalize_L2(qv)

    overfetch = min(len(chunks), max(k * 3, k))
    D, I = index.search(qv, overfetch)

    hits = []
    q_lower = query.lower()
    for score, idx in zip(D[0], I[0]):
        m = chunks[idx]
        text_lower = m["text"].lower()
        kw_bonus = sum(1 for term in FIN_TERMS if term in q_lower and term in text_lower)
        final = float(score) + 0.1 * kw_bonus
        hits.append({"score": final, **m})
    hits.sort(key=lambda x: -x["score"])
    return hits[:k]

# -----------------------------
# Post-processing & Answer Builder (strict)
# -----------------------------
_CANON = {
    "debt": ["debt", "borrowings", "net interest", "interest expense", "interest revenue"],
    "taxation": ["tax", "taxation", "gsp", "gross state product", "taxation as a % of gsp"],
    "gsp": ["gsp", "gross state product"],
    "net assets": ["net assets", "total assets", "total liabilities"],
    "superannuation": ["superannuation", "liabilities", "funded", "percentage funding"],
    "credit rating": ["credit rating", "triple a", "aaa"],
    "balanced budget": ["balanced budget", "operating result", "surplus", "economic cycle"],
    "infrastructure": ["infrastructure", "capital works", "property, plant and equipment"],
}

def _derive_must_terms(query: str):
    ql = query.lower()
    must = set()
    for key, aliases in _CANON.items():
        if any(a in ql for a in [key] + aliases):
            must.add(key)
    # Pairing: taxation vs gsp → require both
    if ("tax" in ql or "taxation" in ql) and ("gsp" in ql or "gross state product" in ql):
        must.update(["taxation", "gsp"])
    toks = set(re.findall(r"[a-z]+", ql))
    return must, toks

def _text_has_key(text_l: str, key: str) -> bool:
    aliases = [key] + _CANON.get(key, [])
    return any(a in text_l for a in aliases)

def _filter_hits_by_must(hits, must_keys):
    if not must_keys:
        return hits
    kept = []
    for h in hits:
        t = h["text"].lower()
        if all(_text_has_key(t, k) for k in must_keys):
            kept.append(h)
    if not kept:
        for h in hits:
            t = h["text"].lower()
            if any(_text_has_key(t, k) for k in must_keys):
                kept.append(h)
    return kept or hits

def _select_spans(text: str, q_tokens: set, must_keys: set, max_chars: int = 700):
    # split by sentences and lines (helps with wrapped bullets/tables)
    parts = re.split(r'(?<=[.!?])\s+|\n', text)
    parts = [p.strip() for p in parts if p.strip()]
    parts = [p for p in parts if len(p.split()) >= 4]
    selected = []

    def _matches(p):
        pl = p.lower()
        if any(t in pl for t in q_tokens if len(t) > 2):
            return True
        if any(_text_has_key(pl, k) for k in must_keys):
            return True
        return False

    for p in parts:
        if _matches(p):
            if sum(len(x) for x in selected) + len(p) + 1 <= max_chars:
                selected.append(p)
    return selected

def _postprocess_answer(question_l: str, selected_spans: list, full_hits: list = None):
    """
    Tailored polish to mimic notebook answers:
      - Strategic priorities: stitch 6 bullets even if line-wrapped.
      - Taxation vs GSP: keep lines that mention both or % line.
      - Superannuation: keep target/funding %; drop assets/liabilities noise.
      - Else: join selected spans.
    """
    def dedupe(seq):
        seen = set(); out = []
        for x in seq:
            if x not in seen:
                seen.add(x); out.append(x)
        return out

    ans = " ".join(selected_spans).strip()
    top_text = full_hits[0]["text"] if (full_hits and len(full_hits) > 0 and "text" in full_hits[0]) else ""
    combined_text = ("\n".join(selected_spans) + ("\n" + top_text if top_text else "")).strip()

    # Strategic priorities
    if "strategic" in question_l and "priorit" in question_l:
        lines = [ln.rstrip() for ln in (top_text or combined_text).splitlines()]
        bullets = []
        i = 0
        while i < len(lines):
            l = lines[i].strip()
            if l.startswith("•") or l.startswith("- "):
                buf = l
                j = i + 1
                while j < len(lines):
                    nxt = lines[j].strip()
                    if not nxt or nxt.startswith("•") or nxt.startswith("- "):
                        break
                    buf += " " + nxt
                    j += 1
                bullets.append(buf.strip())
                i = j
            else:
                i += 1
        bullets = dedupe(bullets)
        if bullets:
            bullets = bullets[:6]
            return "Strategic priorities, as they relate to the Territory’s Budget, are summarised as: " + " ".join(bullets)

    # Taxation vs GSP
    if ("tax" in question_l or "taxation" in question_l) and ("gsp" in question_l or "gross state product" in question_l):
        lines = []
        for s in selected_spans:
            for ln in s.splitlines():
                l = ln.strip()
                ll = l.lower()
                if ("taxation" in ll and "gsp" in ll) or ("taxation as a % of gsp" in ll) or re.search(r"\b\d\.\d%\b", l):
                    lines.append(l)
        if lines:
            return "\n".join(lines)

    # Superannuation (keep targets/funding)
    if "superannuation" in question_l:
        sentences = re.split(r'(?<=[.!?])\s+', combined_text)
        keep = []
        for s in sentences:
            sl = s.lower()
            if any(bad in sl for bad in ["net assets", "total assets", "total liabilities"]):
                continue
            if ("90%" in s) or ("2039" in sl) or ("2040" in sl) or ("percentage funding" in sl):
                keep.append(s.strip())
        keep = dedupe(keep)
        if keep:
            return " ".join(keep[:2])

    return ans

# Memory-aware augmentation
class Mem:
    def __init__(self, max_turns=4):
        self.history = deque(maxlen=max_turns)

    def last_topic(self):
        if not self.history:
            return ""
        prev_user = self.history[-1][0]
        nouns = re.findall(r"\b(debt|tax|taxation|net assets|infrastructure|superannuation|interest|operating result|credit rating|gsp)\b",
                           prev_user.lower())
        return nouns[-1] if nouns else ""

    def augment(self, q):
        topic = self.last_topic()
        if topic and (re.search(r"\b(it|that|those|them|this|one)\b", q.lower()) or "what about" in q.lower()):
            return f"{q} (context topic: {topic})"
        return q

mem = Mem()

def build_answer(question: str, k: int = TOP_K, chunks: List[Dict] = None, index: faiss.IndexFlatIP = None, model=None) -> str:
    q_aug = mem.augment(question)
    must_keys, q_tokens = _derive_must_terms(q_aug)

    hits = hybrid_search(q_aug, k=max(k, 6), chunks=chunks, index=index, model=model)
    hits = _filter_hits_by_must(hits, must_keys)

    selected, cites = [], []
    for h in hits:
        spans = _select_spans(h["text"], q_tokens, must_keys, max_chars=700)
        if spans:
            selected.extend(spans)
            cites.append(f"p.{h['page']}")
            if len(" ".join(selected)) > 600:  # stop when we have enough evidence
                break

    if not selected and hits:
        selected = [hits[0]["text"].strip()]
        cites.append(f"p.{hits[0]['page']}")

    answer = _postprocess_answer(q_aug.lower(), selected, full_hits=hits).strip()
    cite_str = ", ".join(sorted(set(cites), key=lambda x: int(x.split(".")[1])))
    return f"{answer}\n\nSources: {cite_str}"

# -----------------------------
# Cached data/model/index setup
# -----------------------------
@st.cache_data(show_spinner=True)
def _load_and_chunk(pdf_path: str):
    pages = extract_pages(pdf_path)
    chunks = fine_chunks(pages, max_chars=MAX_CHARS_PER_CHUNK)
    return pages, chunks

# Load once (cached)
pages, CHUNKS = _load_and_chunk(PDF_PATH)
MODEL, INDEX, _EMB = build_index(CHUNKS)

# -----------------------------
# UI
# -----------------------------
st.title("💬 JVAI Financial Policy Chatbot By Md. Shoaib Ahmed")
st.caption("Answers are extracted from the bundled policy PDF with semantic + keyword search and strict filtering, then returned with page citations.")

# Small doc summary strip
with st.container():
    left, right = st.columns([3, 1])
    with left:
        st.markdown(f"**Document:** `{os.path.basename(PDF_PATH)}`")
        st.markdown(f"**Pages:** {len(pages)} &nbsp;&nbsp; **Chunks:** {len(CHUNKS)} &nbsp;&nbsp; **Top-k:** {TOP_K}")
    with right:
        st.success("Indexed and ready", icon="✅")

st.divider()
st.subheader("Chat")

# Chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

for role, content in st.session_state.messages:
    with st.chat_message(role):
        st.markdown(content)

user_q = st.chat_input("Ask about strategic priorities, taxation vs GSP, debt, superannuation, infrastructure, etc.")
if user_q:
    # Show user turn
    st.session_state.messages.append(("user", user_q))
    with st.chat_message("user"):
        st.markdown(user_q)

    # Build answer
    with st.chat_message("assistant"):
        with st.spinner("Searching the document..."):
            ans = build_answer(user_q, k=TOP_K, chunks=CHUNKS, index=INDEX, model=MODEL)
            st.markdown(ans)

    # Save to memory & history
    mem.history.append((user_q, ans))
    st.session_state.messages.append(("assistant", ans))
