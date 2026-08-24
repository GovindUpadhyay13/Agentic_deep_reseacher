import streamlit as st
import os
import sys
import io
import re
import time
import operator
from typing import TypedDict, List, Dict, Any, Annotated, Optional
from contextlib import redirect_stdout
from dotenv import load_dotenv

load_dotenv()

# Add current dir to sys.path
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import google.generativeai as genai
from langgraph.graph import StateGraph, END

# Import Hybrid Retrieval Subsystem (BM25 + Qdrant + RRF + Semantic Reranker)
try:
    from hybrid_retriever import fetch_multi_source_documents, hybrid_engine
except ImportError:
    from src.hybrid_retriever import fetch_multi_source_documents, hybrid_engine

# ─────────────────────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Karpathy", page_icon="🔭", layout="wide")

# ─────────────────────────────────────────────────────────────────────────────
# Design system — Perplexity-inspired dark UI with LangGraph Research styling
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:ital,wght@0,300;0,400;0,500;0,600;0,700;1,400&display=swap');

:root {
  --bg:             #0f1010;
  --bg-surface:     #181a1a;
  --bg-card:        #1d1f1f;
  --bg-hover:       #232525;
  --border:         rgba(255,255,255,0.07);
  --border-hi:      rgba(255,255,255,0.13);
  --accent:         #20808d;
  --accent-2:       #25939f;
  --accent-glow:    rgba(32,128,141,0.12);
  --t1:             #efefed;
  --t2:             #9a9a95;
  --t3:             #5c5c58;
  --r-xl:           24px;
  --r-lg:           16px;
  --r-md:           12px;
  --r-sm:           8px;
  /* source palette */
  --c-arxiv:        #20808d;
  --c-crossref:     #e07b2a;
  --c-s2:           #4a8fd4;
  --c-wiki:         #71717a;
  --c-web:          #8b5cf6;
  --c-local:        #10b981;
}

html, body, [class*="css"] {
  font-family: 'Inter', system-ui, -apple-system, sans-serif !important;
  background: var(--bg) !important;
  color: var(--t1) !important;
}
#MainMenu, .stDeployButton, header, footer { visibility: hidden !important; display: none !important; }

/* ── Layout ─────────────────────────────────────────────────────────────── */
.main .block-container {
  max-width: 880px !important;
  margin: 0 auto !important;
  padding: 0 1.5rem 5rem !important;
}

/* ── Sidebar ─────────────────────────────────────────────────────────────── */
section[data-testid="stSidebar"] {
  background: var(--bg-surface) !important;
  border-right: 1px solid var(--border) !important;
}
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] span {
  color: var(--t2) !important;
  font-size: 0.8rem !important;
}
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 {
  color: var(--t1) !important;
  font-size: 0.75rem !important;
  font-weight: 600 !important;
  letter-spacing: 0.07em !important;
  text-transform: uppercase !important;
}

/* ── Animations ──────────────────────────────────────────────────────────── */
@keyframes fadeUp {
  from { opacity: 0; transform: translateY(10px); }
  to   { opacity: 1; transform: translateY(0); }
}
@keyframes pulse-dot {
  0%,100% { opacity:1; transform:scale(1); }
  50%      { opacity:0.5; transform:scale(1.4); }
}
@keyframes slideRight {
  from { opacity: 0; transform: translateX(-10px); }
  to   { opacity: 1; transform: translateX(0); }
}

/* ── Guest Login Card ────────────────────────────────────────────────────── */
.karp-login-wrap {
  max-width: 520px;
  margin: 3.5rem auto 2rem;
  background: var(--bg-surface);
  border: 1px solid var(--border-hi);
  border-radius: var(--r-xl);
  padding: 2.2rem 2.2rem 1.8rem;
  box-shadow: 0 16px 36px rgba(0,0,0,0.35);
  animation: fadeUp 0.4s ease both;
  text-align: center;
}
.karp-login-badge {
  display: inline-block;
  font-size: 0.7rem;
  font-weight: 700;
  color: var(--accent);
  background: var(--accent-glow);
  border: 1px solid rgba(32,128,141,0.3);
  border-radius: 999px;
  padding: 0.2rem 0.65rem;
  margin-bottom: 0.7rem;
  text-transform: uppercase;
  letter-spacing: 0.05em;
}
.karp-login-title {
  font-size: 1.65rem;
  font-weight: 700;
  color: var(--t1);
  letter-spacing: -0.02em;
  margin-bottom: 0.35rem;
}
.karp-login-desc {
  font-size: 0.85rem;
  color: var(--t2);
  line-height: 1.5;
  margin-bottom: 1.6rem;
}
.karp-user-pill {
  display: flex;
  align-items: center;
  justify-content: space-between;
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-radius: var(--r-sm);
  padding: 0.55rem 0.85rem;
  margin-bottom: 1rem;
}
.karp-user-pill-name {
  font-size: 0.82rem;
  font-weight: 600;
  color: var(--t1);
  display: flex;
  align-items: center;
  gap: 0.4rem;
}
.karp-user-pill-badge {
  font-size: 0.65rem;
  font-weight: 700;
  color: var(--accent);
  background: var(--accent-glow);
  padding: 0.15rem 0.45rem;
  border-radius: 4px;
}

/* ── Hero (home state) ───────────────────────────────────────────────────── */
.karp-hero {
  text-align: center;
  padding: 3.5rem 1rem 2rem;
  animation: fadeUp 0.4s ease both;
}
.karp-logo-mark {
  font-size: 2.4rem;
  font-weight: 700;
  letter-spacing: -0.03em;
  color: var(--t1);
  margin-bottom: 0.4rem;
}
.karp-tagline {
  font-size: 0.95rem;
  color: var(--t2);
  margin-bottom: 1.5rem;
}
.karp-source-pills {
  display: flex;
  justify-content: center;
  gap: 0.5rem;
  flex-wrap: wrap;
  margin-bottom: 2rem;
}
.karp-src-pill {
  font-size: 0.72rem;
  font-weight: 600;
  letter-spacing: 0.04em;
  padding: 0.25rem 0.65rem;
  border-radius: 999px;
  border: 1px solid var(--border);
  text-transform: uppercase;
}
.karp-src-pill.arxiv    { color: var(--c-arxiv);    border-color: rgba(32,128,141,0.3); }
.karp-src-pill.s2       { color: var(--c-s2);       border-color: rgba(74,143,212,0.3); }
.karp-src-pill.crossref { color: var(--c-crossref); border-color: rgba(224,123,42,0.3); }
.karp-src-pill.wiki     { color: var(--c-wiki);     border-color: rgba(113,113,122,0.3); }
.karp-src-pill.web      { color: var(--c-web);      border-color: rgba(139,92,246,0.3); }

/* ── Suggested chips ─────────────────────────────────────────────────────── */
.karp-suggest-label {
  font-size: 0.75rem;
  color: var(--t3);
  text-transform: uppercase;
  letter-spacing: 0.06em;
  margin-bottom: 0.6rem;
}
.karp-chips {
  display: flex;
  justify-content: center;
  gap: 0.4rem;
  flex-wrap: wrap;
}
.karp-chip {
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-radius: 999px;
  padding: 0.35rem 0.85rem;
  font-size: 0.78rem;
  color: var(--t2);
}

/* ── Search Input ────────────────────────────────────────────────────────── */
div[data-testid="stTextInput"] input {
  background: var(--bg-surface) !important;
  border: 1px solid var(--border-hi) !important;
  border-radius: var(--r-xl) !important;
  color: var(--t1) !important;
  font-size: 0.95rem !important;
  padding: 0.75rem 1.25rem !important;
}
div[data-testid="stTextInput"] input:focus {
  border-color: var(--accent) !important;
  box-shadow: 0 0 0 3px var(--accent-glow) !important;
}
div[data-testid="stButton"] button {
  background: var(--accent) !important;
  color: #fff !important;
  border: none !important;
  border-radius: var(--r-xl) !important;
  font-weight: 600 !important;
  padding: 0.75rem 1.5rem !important;
}
div[data-testid="stButton"] button:hover {
  background: var(--accent-2) !important;
}

/* ── Query Headline ──────────────────────────────────────────────────────── */
.karp-query-head {
  font-size: 1.45rem;
  font-weight: 700;
  color: var(--t1);
  letter-spacing: -0.02em;
  margin: 1.5rem 0 1rem;
  animation: fadeUp 0.3s ease both;
}

/* ── Live Step Trace ─────────────────────────────────────────────────────── */
.karp-trace {
  display: flex;
  flex-direction: column;
  gap: 0.4rem;
  margin-bottom: 1.2rem;
}
.karp-step {
  display: flex;
  align-items: flex-start;
  gap: 0.65rem;
  font-size: 0.82rem;
  animation: slideRight 0.25s ease both;
}
.karp-step-dot {
  width: 18px;
  height: 18px;
  border-radius: 50%;
  display: flex;
  align-items: center;
  justify-content: center;
  flex-shrink: 0;
  margin-top: 1px;
  font-size: 0.65rem;
  font-weight: 700;
}
.karp-step-dot.done {
  background: rgba(32,128,141,0.2);
  color: var(--accent);
  border: 1px solid var(--accent);
}
.karp-step-dot.active {
  background: var(--accent);
  color: #fff;
  animation: pulse-dot 1.2s infinite;
}
.karp-step-lbl { font-size: 0.82rem; }
.karp-step-lbl.done   { color: var(--t2); }
.karp-step-lbl.active { color: var(--t1); font-weight: 500; }

/* ── Source cards inside step trace ──────────────────────────────────────── */
.karp-src-row {
  display: flex;
  gap: 0.5rem;
  flex-wrap: wrap;
  margin: 0.35rem 0 0.2rem 1.7rem;
}
.karp-src-card {
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-radius: var(--r-sm);
  padding: 0.4rem 0.65rem;
  max-width: 260px;
  text-decoration: none;
  display: flex;
  flex-direction: column;
  gap: 0.2rem;
  animation: fadeUp 0.2s ease both;
}
.karp-src-card:hover {
  border-color: var(--accent);
  background: var(--bg-hover);
}
.karp-src-card-title {
  font-size: 0.72rem;
  font-weight: 500;
  color: var(--t1);
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}
.karp-src-badge {
  font-size: 0.62rem;
  font-weight: 600;
  border-radius: 4px;
  padding: 0.1rem 0.35rem;
  width: fit-content;
}
.badge-arxiv    { background: rgba(32,128,141,0.15); color: var(--c-arxiv); }
.badge-crossref { background: rgba(224,123,42,0.15); color: var(--c-crossref); }
.badge-s2       { background: rgba(74,143,212,0.15); color: var(--c-s2); }
.badge-wiki     { background: rgba(113,113,122,0.15); color: var(--c-wiki); }
.badge-web      { background: rgba(139,92,246,0.15); color: var(--c-web); }
.badge-local    { background: rgba(16,185,129,0.15); color: var(--c-local); }

/* ── Summary Pill ────────────────────────────────────────────────────────── */
.karp-sum-pill {
  display: inline-flex;
  align-items: center;
  gap: 0.4rem;
  background: rgba(32,128,141,0.1);
  border: 1px solid rgba(32,128,141,0.25);
  border-radius: 999px;
  padding: 0.28rem 0.8rem;
  font-size: 0.78rem;
  font-weight: 500;
  color: var(--accent);
  margin-bottom: 1.2rem;
  animation: fadeUp 0.3s ease both;
}

/* ── Sources Index grid (shown above answer) ────────────────────────────── */
.karp-sources-idx { margin: 1rem 0 1.4rem; }
.karp-sources-idx-hdr {
  font-size: 0.7rem; font-weight:700; letter-spacing:0.08em;
  text-transform:uppercase; color:var(--t2); margin-bottom:0.55rem;
  display:flex; align-items:center; gap:0.4rem;
}
.karp-sources-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(190px,1fr));
  gap: 0.45rem;
}
.karp-idx-card {
  background: var(--bg-card); border:1px solid var(--border);
  border-radius: var(--r-sm); padding:0.5rem 0.65rem;
  text-decoration:none; display:flex; gap:0.5rem; align-items:flex-start;
  transition: border-color 0.15s, background 0.15s;
  animation: fadeUp 0.2s ease both;
}
.karp-idx-card:hover { border-color:var(--accent); background:var(--bg-hover); }
.karp-idx-num { font-size:0.62rem; font-weight:700; color:var(--t3); flex-shrink:0; min-width:1.1rem; margin-top:0.12rem; }
.karp-idx-title { font-size:0.74rem; font-weight:500; color:var(--t1); display:-webkit-box; -webkit-line-clamp:2; -webkit-box-orient:vertical; overflow:hidden; line-height:1.3; margin-bottom:0.2rem; }
.karp-idx-meta { display:flex; gap:0.4rem; align-items:center; }
.karp-idx-year { font-size:0.62rem; color:var(--t3); }

/* ── Answer Sections ─────────────────────────────────────────────────────── */
.karp-answer {
  display: flex;
  flex-direction: column;
  gap: 1.2rem;
  animation: fadeUp 0.35s ease both;
}
.karp-sec {
  background: var(--bg-surface);
  border: 1px solid var(--border);
  border-radius: var(--r-lg);
  padding: 1.2rem 1.4rem;
}
.karp-sec-hdr {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  margin-bottom: 0.85rem;
}
.karp-sec-hdr h3 {
  font-size: 0.95rem !important;
  font-weight: 700 !important;
  color: var(--t1) !important;
  margin: 0 !important;
}
.karp-sec-icon { font-size: 1rem; }
.karp-sec-body {
  font-size: 0.88rem;
  line-height: 1.7;
  color: var(--t1);
}
.karp-sec-body p  { margin: 0 0 0.75rem; }
.karp-sec-body p:last-child { margin-bottom: 0; }
.karp-sec-body ul { margin: 0 0 0.75rem; padding-left: 1.2rem; }
.karp-sec-body li { margin-bottom: 0.35rem; }

/* ── Survey section star highlight ───────────────────────────────────────── */
.karp-survey-body {
  background: rgba(32,128,141,0.06);
  border: 1px solid rgba(32,128,141,0.2);
  border-radius: var(--r-md);
  padding: 0.9rem 1.1rem;
}
.karp-survey-star {
  display: inline-flex;
  align-items: center;
  gap: 0.3rem;
  font-size: 0.7rem;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: 0.06em;
  color: var(--accent);
  margin-bottom: 0.5rem;
}

/* ── Visual Vertical Timeline ────────────────────────────────────────────── */
.karp-tl {
  position: relative;
  padding-left: 2rem;
  margin: 0.5rem 0;
}
.karp-tl::before {
  content: '';
  position: absolute;
  left: 7px;
  top: 10px;
  bottom: 10px;
  width: 2px;
  background: linear-gradient(to bottom, var(--accent), rgba(32,128,141,0.15));
}
.karp-tl-item {
  position: relative;
  margin-bottom: 1.4rem;
  animation: slideRight 0.3s ease both;
}
.karp-tl-item:last-child { margin-bottom: 0; }
.karp-tl-item::before {
  content: '';
  position: absolute;
  left: -2rem;
  top: 5px;
  width: 16px;
  height: 16px;
  border-radius: 50%;
  background: var(--bg-surface);
  border: 3px solid var(--accent);
  box-shadow: 0 0 0 2px var(--bg);
}
.karp-tl-item.tl-undated::before { border-color: var(--t3); }
.karp-tl-year {
  font-size: 0.75rem;
  font-weight: 700;
  letter-spacing: 0.06em;
  color: var(--accent);
  text-transform: uppercase;
  margin-bottom: 0.3rem;
}
.karp-tl-item.tl-undated .karp-tl-year { color: var(--t3); }
.karp-tl-card {
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-radius: var(--r-md);
  padding: 0.75rem 1rem;
  font-size: 0.85rem;
  line-height: 1.65;
  color: var(--t1);
}

/* ── SOTA model cards ───────────────────────────────────────────────────── */
.karp-sota-grid { display:flex; flex-direction:column; gap:0.6rem; margin-top:0.5rem; }
.karp-sota-card {
  background: var(--bg-card); border:1px solid var(--border);
  border-left: 3px solid var(--c-s2);
  border-radius: var(--r-md); padding:0.75rem 1rem;
  animation: slideRight 0.3s ease both;
}
.karp-sota-name { font-size:0.9rem; font-weight:700; color:var(--t1); margin-bottom:0.4rem; }
.karp-sota-body { font-size:0.85rem; line-height:1.65; color:var(--t1); }
.karp-sota-body ul { margin:0.2rem 0 0; padding-left:1.1rem; }
.karp-sota-body li { margin-bottom:0.2rem; }

/* ── Research Tables ────────────────────────────────────────────────────── */
.karp-table-wrap {
  overflow-x: auto;
  margin: 0.8rem 0 1.2rem;
  border-radius: var(--r-md);
  border: 1px solid var(--border);
}
.karp-table {
  width: 100%;
  border-collapse: collapse;
  font-size: 0.82rem;
  text-align: left;
  background: var(--bg-card);
}
.karp-table th {
  background: rgba(32,128,141,0.12);
  color: var(--t1);
  font-weight: 700;
  padding: 0.65rem 0.85rem;
  border-bottom: 1px solid var(--border);
  letter-spacing: 0.02em;
}
.karp-table td {
  padding: 0.6rem 0.85rem;
  border-bottom: 1px solid var(--border);
  color: var(--t1);
  line-height: 1.45;
}
.karp-table tr:last-child td { border-bottom: none; }
.karp-table tr:hover td { background: var(--bg-hover); }

/* ── Blockquotes ────────────────────────────────────────────────────────── */
.karp-quote {
  border-left: 3px solid var(--accent);
  background: var(--accent-glow);
  padding: 0.6rem 1rem;
  border-radius: 0 var(--r-sm) var(--r-sm) 0;
  margin: 0.6rem 0;
  color: var(--t2);
  font-style: italic;
  font-size: 0.88rem;
}

/* ── Citations inline ────────────────────────────────────────────────────── */
.karp-cite {
  font-size: 0.72rem;
  font-weight: 600;
  color: var(--accent);
  background: rgba(32,128,141,0.12);
  border: 1px solid rgba(32,128,141,0.25);
  border-radius: 4px;
  padding: 0.05rem 0.35rem;
  margin: 0 0.15rem;
  text-decoration: none;
}
.karp-cite:hover {
  background: rgba(32,128,141,0.25);
  color: var(--t1);
}

/* ── Error container ─────────────────────────────────────────────────────── */
.karp-error {
  background: rgba(239,68,68,0.08);
  border: 1px solid rgba(239,68,68,0.25);
  border-radius: var(--r-md);
  padding: 1rem 1.2rem;
  color: #fca5a5;
  font-size: 0.88rem;
  line-height: 1.6;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Helper: classify source badge CSS class
# ─────────────────────────────────────────────────────────────────────────────
def _src_badge_class(source: str) -> str:
    s = source.lower()
    if "arxiv"   in s: return "badge-arxiv"
    if "crossref" in s: return "badge-crossref"
    if "semantic" in s or " s2" in s: return "badge-s2"
    if "wikipedia" in s: return "badge-wiki"
    if "web" in s or "duckduckgo" in s or "ddg" in s: return "badge-web"
    if "local" in s: return "badge-local"
    return "badge-arxiv"


# ─────────────────────────────────────────────────────────────────────────────
# Helper: render live step trace HTML for LangGraph nodes
# ─────────────────────────────────────────────────────────────────────────────
def _render_trace(steps: list) -> str:
    rows = []
    for i, s in enumerate(steps):
        status = s.get("status", "done")
        icon = "✓" if status == "done" else "·"
        delay = i * 0.06
        src_html = ""
        if s.get("sources"):
            cards = []
            for j, src in enumerate(s["sources"]):
                title  = src.get("title", src.get("arxiv_id", ""))[:80]
                url    = src.get("url", f"https://arxiv.org/abs/{src.get('arxiv_id', '')}")
                label  = src.get("source", "arXiv")
                badge  = _src_badge_class(label)
                cd     = j * 0.07
                cards.append(
                    f'<a class="karp-src-card" href="{url}" target="_blank" style="animation-delay:{cd:.2f}s">'
                    f'<div class="karp-src-card-title">{title}</div>'
                    f'<span class="karp-src-badge {badge}">{label[:22]}</span>'
                    f'</a>'
                )
            src_html = f'<div class="karp-src-row">{"".join(cards)}</div>'
        rows.append(
            f'<div class="karp-step" style="animation-delay:{delay:.2f}s">'
            f'<div class="karp-step-dot {status}">{icon}</div>'
            f'<div style="flex:1"><div class="karp-step-lbl {status}">{s["name"]}</div>{src_html}</div>'
            f'</div>'
        )
    return f'<div class="karp-trace">{"".join(rows)}</div>'


def _summary_pill(n_src: int, n_steps: int) -> str:
    return (
        f'<div class="karp-sum-pill">'
        f'✦ LangGraph completed {n_steps} steps · Indexed {n_src} sources'
        f'</div>'
    )


def _render_sources_index(sources: list) -> str:
    """Numbered grid of ALL sources collected — shown above the answer."""
    if not sources:
        return ''
    cards = []
    for i, src in enumerate(sources, 1):
        title  = src.get('title', src.get('arxiv_id', ''))[:80]
        url    = src.get('url', '#')
        label  = src.get('source', 'arXiv')
        year   = src.get('year')
        badge  = _src_badge_class(label)
        delay  = (i - 1) * 0.04
        year_html = f'<span class="karp-idx-year">{year}</span>' if year else ''
        cards.append(
            f'<a class="karp-idx-card" href="{url}" target="_blank" style="animation-delay:{delay:.2f}s">'
            f'<span class="karp-idx-num">{i}</span>'
            f'<div>'
            f'<div class="karp-idx-title">{title}</div>'
            f'<div class="karp-idx-meta">'
            f'<span class="karp-src-badge {badge}">{label[:18]}</span>'
            f'{year_html}'
            f'</div>'
            f'</div>'
            f'</a>'
        )
    grid = f'<div class="karp-sources-grid">{"" .join(cards)}</div>'
    n = len(sources)
    return (
        f'<div class="karp-sources-idx">'
        f'<div class="karp-sources-idx-hdr">'
        f'<span>📚</span> {n} source{"s" if n != 1 else ""} indexed'
        f'</div>'
        f'{grid}'
        f'</div>'
    )


# ─────────────────────────────────────────────────────────────────────────────
# Helper: markdown parsing and section renderers
# ─────────────────────────────────────────────────────────────────────────────
def _cites(text: str) -> str:
    """Replace [id] citations with styled pill links."""
    def _r(m):
        aid = m.group(1)
        url = f"https://arxiv.org/abs/{aid}" if re.match(r'\d{4}\.\d+', aid) else f"https://doi.org/{aid}"
        return f'<a class="karp-cite" href="{url}" target="_blank">[{aid}]</a>'
    text = re.sub(r'\[(\d{4}\.\d{4,6}(?:v\d+)?)\]', _r, text)
    text = re.sub(r'\[(wiki:[^\]]+)\]', r'<span class="karp-cite">[\1]</span>', text)
    return text


def _inline(text: str) -> str:
    text = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', text)
    text = re.sub(r'\*(.+?)\*',     r'<em>\1</em>',         text)
    text = re.sub(r'`(.+?)`',       r'<code>\1</code>',     text)
    text = _cites(text)
    return text


def _md_to_html(text: str, card_class: str = "karp-sec-body") -> str:
    """Parses markdown text into rich styled HTML including tables, lists, and subheadings."""
    lines = text.strip().split('\n')
    out = []
    in_ul = False
    in_table = False
    table_header_done = False

    for line in lines:
        s = line.strip()
        if not s:
            if in_ul:
                out.append('</ul>'); in_ul = False
            if in_table:
                out.append('</tbody></table></div>'); in_table = False; table_header_done = False
            continue

        if s.startswith('|') and s.endswith('|'):
            if in_ul:
                out.append('</ul>'); in_ul = False
            cells = [c.strip() for c in s[1:-1].split('|')]
            if all(set(c).issubset({'-', ':', ' '}) for c in cells):
                table_header_done = True
                continue
            if not in_table:
                in_table = True
                out.append('<div class="karp-table-wrap"><table class="karp-table"><thead><tr>')
                for c in cells:
                    out.append(f'<th>{_inline(c)}</th>')
                out.append('</tr></thead><tbody>')
            else:
                out.append('<tr>')
                for c in cells:
                    out.append(f'<td>{_inline(c)}</td>')
                out.append('</tr>')
            continue

        if in_table:
            out.append('</tbody></table></div>')
            in_table = False
            table_header_done = False

        if s.startswith('#### '):
            if in_ul: out.append('</ul>'); in_ul = False
            out.append(f'<h5 style="margin:0.7rem 0 0.3rem;font-size:0.85rem;color:var(--t1);font-weight:600;">{_inline(s[5:])}</h5>')
        elif s.startswith('### '):
            if in_ul: out.append('</ul>'); in_ul = False
            out.append(f'<h4 style="margin:0.9rem 0 0.4rem;font-size:0.92rem;color:var(--t1);font-weight:700;">{_inline(s[4:])}</h4>')
        elif s.startswith('> '):
            if in_ul: out.append('</ul>'); in_ul = False
            out.append(f'<blockquote class="karp-quote">{_inline(s[2:])}</blockquote>')
        elif s.startswith('- ') or s.startswith('* '):
            if not in_ul: out.append('<ul>'); in_ul = True
            out.append(f'<li>{_inline(s[2:])}</li>')
        elif re.match(r'^\d+\.\s', s):
            if in_ul: out.append('</ul>'); in_ul = False
            content = re.sub(r'^\d+\.\s', '', s)
            out.append(f'<p style="margin-left:0.5rem"><strong>{s.split(".")[0]}.</strong> {_inline(content)}</p>')
        else:
            if in_ul: out.append('</ul>'); in_ul = False
            out.append(f'<p>{_inline(s)}</p>')

    if in_ul: out.append('</ul>')
    if in_table: out.append('</tbody></table></div>')
    return '\n'.join(out)


def _render_timeline_entries(content: str) -> str:
    """Parse ### YEAR entries into the visual timeline."""
    raw_entries = re.split(r'\n(?=### )', '\n' + content)
    items = []
    for i, entry in enumerate(raw_entries):
        entry = entry.strip()
        if not entry:
            continue
        m = re.match(r'^###\s*(.+)', entry)
        if not m:
            if entry:
                items.append(f'<p style="color:var(--t2);font-size:0.85rem">{_inline(entry)}</p>')
            continue
        year_label = m.group(1).strip()
        rest = entry[m.end():].strip()
        is_undated = any(w in year_label.lower() for w in ('undated', 'web', 'n/a', 'unknown'))
        extra_cls  = 'tl-undated' if is_undated else ''
        delay      = i * 0.08
        items.append(
            f'<div class="karp-tl-item {extra_cls}" style="animation-delay:{delay:.2f}s">'
            f'  <div class="karp-tl-year">{year_label}</div>'
            f'  <div class="karp-tl-card">{_md_to_html(rest)}</div>'
            f'</div>'
        )
    return f'<div class="karp-tl">{"".join(items)}</div>'


def _render_sota_entries(content: str) -> str:
    """Parse ### [Model Name] entries into SOTA model cards with blue-left border."""
    raw_entries = re.split(r'\n(?=### )', '\n' + content)
    cards = []
    preamble = ''
    for i, entry in enumerate(raw_entries):
        entry = entry.strip()
        if not entry:
            continue
        m = re.match(r'^###\s*(.+)', entry)
        if not m:
            preamble = f'<div class="karp-sec-body" style="margin-bottom:0.5rem">{_md_to_html(entry)}</div>'
            continue
        model_name = m.group(1).strip()
        rest = entry[m.end():].strip()
        delay = i * 0.07
        cards.append(
            f'<div class="karp-sota-card" style="animation-delay:{delay:.2f}s">'
            f'<div class="karp-sota-name">{_inline(model_name)}</div>'
            f'<div class="karp-sota-body">{_md_to_html(rest)}</div>'
            f'</div>'
        )
    if not cards:
        return f'<div class="karp-sec-body">{_md_to_html(content)}</div>'
    return preamble + f'<div class="karp-sota-grid">{"".join(cards)}</div>'


def _render_answer(text: str) -> str:
    """Parse structured markdown into styled HTML sections."""
    text = text.strip()
    raw_secs = re.split(r'\n(?=## )', '\n' + text)

    html = '<div class="karp-answer">'
    found_sections = 0

    for sec in raw_secs:
        sec = sec.strip()
        if not sec:
            continue
        m = re.match(r'^##\s*(.+)', sec)
        if not m:
            if sec:
                html += f'<div class="karp-sec"><div class="karp-sec-body">{_md_to_html(sec)}</div></div>'
            continue
        heading = m.group(1).strip()
        body    = sec[m.end():].strip()
        hl = heading.lower()
        found_sections += 1

        if 'timeline' in hl or '📅' in heading or 'chronological' in hl:
            html += (
                '<div class="karp-sec">'
                '<div class="karp-sec-hdr">'
                '<span class="karp-sec-icon">📅</span>'
                '<h3>Chronological Research Timeline</h3>'
                '</div>'
                + _render_timeline_entries(body) +
                '</div>'
            )
        elif 'survey' in hl or 'foundation' in hl or 'overview' in hl or '🔍' in heading:
            html += (
                '<div class="karp-sec">'
                '<div class="karp-sec-hdr">'
                '<span class="karp-sec-icon">🔍</span>'
                '<h3>Survey &amp; Foundation</h3>'
                '</div>'
                f'<div class="karp-sec-body karp-survey-body">'
                f'<div class="karp-survey-star">★ Foundation &amp; Taxonomy</div>'
                + _md_to_html(body) +
                '</div></div>'
            )
        elif 'sota' in hl or 'benchmark' in hl or 'model' in hl or '🤖' in heading:
            html += (
                '<div class="karp-sec">'
                '<div class="karp-sec-hdr">'
                '<span class="karp-sec-icon">🤖</span>'
                '<h3>SOTA Models &amp; Benchmark Comparison</h3>'
                '</div>'
                + _render_sota_entries(body) +
                '</div>'
            )
        elif 'frontier' in hl or 'failure' in hl or 'problem' in hl or 'limitation' in hl or 'state of the art' in hl or '🔬' in heading:
            html += (
                '<div class="karp-sec">'
                '<div class="karp-sec-hdr">'
                '<span class="karp-sec-icon">🔬</span>'
                '<h3>Frontier, Failure Modes &amp; Open Problems</h3>'
                '</div>'
                f'<div class="karp-sec-body karp-sota-body">'
                + _md_to_html(body) +
                '</div></div>'
            )
        elif 'takeaway' in hl or 'synthesis' in hl or 'key' in hl or '💡' in heading:
            html += (
                '<div class="karp-sec">'
                '<div class="karp-sec-hdr">'
                '<span class="karp-sec-icon">💡</span>'
                '<h3>Key Takeaways &amp; Synthesis</h3>'
                '</div>'
                f'<div class="karp-sec-body karp-tips-body">'
                + _md_to_html(body) +
                '</div></div>'
            )
        else:
            clean = re.sub(r'^[^\w\s]+\s*', '', heading).strip()
            html += (
                f'<div class="karp-sec">'
                f'<div class="karp-sec-hdr"><h3>{clean}</h3></div>'
                f'<div class="karp-sec-body">{_md_to_html(body)}</div>'
                f'</div>'
            )

    if found_sections == 0:
        html += f'<div class="karp-sec"><div class="karp-sec-body">{_md_to_html(text)}</div></div>'

    html += '</div>'
    return html


def _stream_words(text: str):
    """Generator: yield answer word-by-word for typewriter effect."""
    words = text.split(' ')
    buf = ''
    for i, w in enumerate(words):
        buf += ('' if i == 0 else ' ') + w
        if (i + 1) % 4 == 0:
            yield buf + ' '
            buf = ''
            time.sleep(0.025)
    if buf:
        yield buf


# ─────────────────────────────────────────────────────────────────────────────
# LangGraph State & Agent Definition
# ─────────────────────────────────────────────────────────────────────────────
class ResearchState(TypedDict):
    question: str
    plan_steps: List[str]
    current_step_idx: int
    current_queries: List[str]
    raw_sources: Annotated[List[Dict[str, Any]], operator.add]
    top_evidence: List[Dict[str, Any]]
    extracted_findings: List[str]
    reflection_notes: str
    is_sufficient: bool
    retry_count: int
    max_retries: int
    verified_evidence: str
    final_answer: str


def call_llm_resilient(model, prompt: str, max_retries: int = 5) -> str:
    """Invokes Gemini LLM with exponential backoff for rate limits."""
    for attempt in range(max_retries):
        try:
            resp = model.generate_content(prompt)
            return resp.text.strip()
        except Exception as e:
            err = str(e)
            if "429" in err or "ResourceExhausted" in err or "quota" in err.lower():
                wait_t = 8 * (attempt + 1)
                time.sleep(wait_t)
            else:
                raise e
    return model.generate_content(prompt).text.strip()


class LangGraphResearchAgent:
    def __init__(self, model=None, max_retries: int = 2):
        self.model = model
        self.max_retries = max_retries
        self.graph = self._build_graph()

    def _planner_node(self, state: ResearchState) -> Dict[str, Any]:
        prompt = f"""You are an elite research planner. The user wants a comprehensive research report for:
"{state['question']}"

Break this research goal into 3 precise, highly specific search angles:
1. Foundational literature reviews, surveys, taxonomy, and historical milestone papers.
2. Technical mechanisms, algorithmic variants, model architectures, and benchmark evaluations.
3. Limitations, failure modes, safety/alignment vulnerabilities, and frontier developments.

Return ONLY a numbered list of 3 search angle descriptions, nothing else."""
        plan_text = call_llm_resilient(self.model, prompt)
        steps = [s.strip() for s in plan_text.split('\n') if s.strip()]
        if len(steps) < 3:
            steps = [
                f"Foundations and survey of {state['question']}",
                f"SOTA architectures and benchmarks for {state['question']}",
                f"Open challenges and failure modes in {state['question']}"
            ]
        return {"plan_steps": steps, "current_step_idx": 0, "reflection_notes": "", "retry_count": 0}

    def _context_node(self, state: ResearchState) -> Dict[str, Any]:
        step_idx = state.get("current_step_idx", 0)
        plan_steps = state.get("plan_steps", [])
        current_focus = plan_steps[step_idx] if step_idx < len(plan_steps) else state["question"]
        reflection = state.get("reflection_notes", "")

        ref_clause = f"\nAddress this gap from previous review: {reflection}" if reflection else ""
        prompt = f"""You are a search query engineer for academic databases.
Research Goal: "{state['question']}"
Current Angle: "{current_focus}"{ref_clause}

Formulate 2 distinct, highly effective search queries for arXiv and academic search engines:
- Query 1: Keyword-rich query for surveys and foundational papers.
- Query 2: Specific query targeting recent SOTA architectures, benchmarks, and mechanisms.

Return ONLY the 2 queries, one per line, no numbering or extra text."""
        raw_queries = call_llm_resilient(self.model, prompt)
        queries = [q.strip().strip('"').strip("'") for q in raw_queries.split('\n') if q.strip()][:2]
        if not queries:
            queries = [state['question'], f"{state['question']} survey benchmark"]
        return {"current_queries": queries}

    def _retriever_node(self, state: ResearchState) -> Dict[str, Any]:
        all_candidates = []
        seen_ids = set()

        for q in state.get("current_queries", [state["question"]]):
            docs = fetch_multi_source_documents(q, n_results=4)
            for d in docs:
                if d["id"] not in seen_ids:
                    seen_ids.add(d["id"])
                    all_candidates.append(d)

        combined_query = " ".join(state.get("current_queries", [state["question"]]))
        top_reranked = hybrid_engine.execute_hybrid_search(
            query=combined_query,
            candidate_docs=all_candidates,
            top_k=6,
        )
        return {"raw_sources": top_reranked, "top_evidence": top_reranked}

    def _reader_node(self, state: ResearchState) -> Dict[str, Any]:
        evidence_text = "\n\n".join([
            f"[Source: {d.get('id')}] ({d.get('year') or 'Web'})\nTitle: {d.get('title')}\nSnippet: {d.get('snippet')}"
            for d in state.get("top_evidence", [])
        ])

        prompt = f"""You are an expert research reader.
Topic: "{state['question']}"

Evidence Chunks:
{evidence_text}

Extract the key technical findings from this evidence:
1. Specific model architectures, loss functions, algorithms, and training techniques.
2. Concrete benchmark evaluations, metric numbers, dataset names, and performance comparisons.
3. Chronological dates and citation tags like [source_id].

Return a dense bullet-point summary of extracted findings:"""
        findings = call_llm_resilient(self.model, prompt)
        current_findings = state.get("extracted_findings", [])
        return {"extracted_findings": current_findings + [findings]}

    def _reflector_node(self, state: ResearchState) -> Dict[str, Any]:
        all_findings = "\n\n".join(state.get("extracted_findings", []))
        prompt = f"""You are a strict research reviewer.
Question: "{state['question']}"

Accumulated Technical Findings:
{all_findings}

Evaluate if there is sufficient multi-year chronological evidence, SOTA benchmarks, and taxonomy to write a publication-grade research dossier.
Respond in this exact format:
SUFFICIENT: [YES or NO]
NOTES: [If NO, explain in 1 sentence what specific angle or benchmark is missing. If YES, write 'Evidence complete.']"""
        review_text = call_llm_resilient(self.model, prompt)
        is_yes = "SUFFICIENT: YES" in review_text.upper()
        notes = review_text.split("NOTES:")[-1].strip() if "NOTES:" in review_text else ""
        retry_count = state.get("retry_count", 0)
        next_step_idx = state.get("current_step_idx", 0) + 1

        return {
            "is_sufficient": is_yes,
            "reflection_notes": notes,
            "retry_count": retry_count if is_yes else retry_count + 1,
            "current_step_idx": next_step_idx,
        }

    def _citation_verifier_node(self, state: ResearchState) -> Dict[str, Any]:
        sources = state.get("raw_sources", [])
        verified_blocks = []
        for s in sources:
            yr = f" ({s['year']})" if s.get('year') else ""
            verified_blocks.append(
                f"[Source: {s.get('id')}]{yr}\nTitle: {s.get('title')}\nSource: {s.get('source')}\nSnippet: {s.get('snippet')}\nURL: {s.get('url')}"
            )
        return {"verified_evidence": "\n\n".join(verified_blocks)}

    def _synthesizer_node(self, state: ResearchState) -> Dict[str, Any]:
        prompt = f"""You are Andrej Karpathy and a Senior Principal AI Research Scientist.
You are writing an authoritative, highly technical, publication-grade Research Dossier for:

"{state['question']}"

Evidence collected across arXiv, Semantic Scholar, CrossRef, Wikipedia, and the web:
{state.get('verified_evidence', '')}

CRITICAL NEGATIVE CONSTRAINTS:
- NEVER output conversational filler (e.g., "Based on the provided evidence", "Here is a summary", "In this report").
- START IMMEDIATELY with the first section header: `## 🔍 Survey & Foundation`.
- Every factual claim MUST cite its exact evidence source tag inline: `[source_id]` (e.g. `[2405.27355v2]` or `[10.1000/xyz]`).
- Do NOT hallucinate papers, dates, or metrics not grounded in the evidence.

MANDATORY DOSSIER STRUCTURE (Follow EXACT Markdown syntax):

## 🔍 Survey & Foundation
Synthesize the foundational paradigm and scope. If survey/review papers or foundational milestone works exist in the evidence, analyze their taxonomy and theoretical underpinnings in 2-3 deep paragraphs. Cite all relevant papers inline [source_id].

## 📅 Chronological Research Timeline
Break down the evolution year-by-year based on the evidence. For EACH year that appears in the evidence, create a `### [Year]` subsection. Under each year, list the key papers/contributions:

### [Year]
**[Paper/System Title]** — [arXiv / CrossRef / Web]
- **Core Mechanism & Objective:** Technical details of the architecture, loss function, algorithm, or methodology. [source_id]
- **Empirical Findings & Metrics:** Concrete benchmark numbers, dataset sizes, score improvements, or ablation results. [source_id]
- **Paradigm Impact:** Why this was a milestone in the trajectory of the field.

(Repeat for all years present in the evidence. Include undated web sources in `### Undated / Web Sources` at the bottom of the timeline.)

## 🤖 SOTA Models & Benchmark Comparison
Analyze the leading state-of-the-art models and algorithmic variants discovered in the evidence.

Include a Markdown comparison table:
| Model / Method | Year | Primary Architecture / Mechanism | Key Benchmark & Result | Primary Citation |
|---|---|---|---|---|
| [Name] | [Year] | [Mechanism] | [Benchmark Score/Metric] | [source_id] |

Below the table, provide deep-dive technical breakdowns for key leading systems:
### [Model/System Name] ([Year])
- **Technical Innovation:** Deep explanation of the novel technique (e.g., training recipe, preference optimization, reward modeling, scaling laws). [source_id]
- **Performance:** Exact benchmark evaluations, Win-rates, or empirical comparisons. [source_id]

## 🔬 Frontier, Failure Modes & Open Problems
Synthesize the critical limitations, vulnerabilities, and open research questions identified in the evidence (e.g., reward tampering, alignment faking, sycophancy, out-of-distribution generalization, compute bottlenecks). 2-3 rigorous paragraphs.

## 💡 Key Takeaways & Synthesis
Provide 5 dense, high-impact, actionable conclusions directly addressing "{state['question']}", each supported by inline citations [source_id].

Generate the full Research Dossier now:"""
        final_report = call_llm_resilient(self.model, prompt)
        return {"final_answer": final_report}

    def _build_graph(self):
        workflow = StateGraph(ResearchState)
        workflow.add_node("planner", self._planner_node)
        workflow.add_node("context", self._context_node)
        workflow.add_node("retriever", self._retriever_node)
        workflow.add_node("reader", self._reader_node)
        workflow.add_node("reflector", self._reflector_node)
        workflow.add_node("citation_verifier", self._citation_verifier_node)
        workflow.add_node("synthesizer", self._synthesizer_node)

        workflow.set_entry_point("planner")
        workflow.add_edge("planner", "context")
        workflow.add_edge("context", "retriever")
        workflow.add_edge("retriever", "reader")
        workflow.add_edge("reader", "reflector")

        def _route_reflector(state: ResearchState) -> str:
            if state.get("is_sufficient", False) or state.get("retry_count", 0) >= state.get("max_retries", self.max_retries):
                return "citation_verifier"
            return "context"

        workflow.add_conditional_edges(
            "reflector",
            _route_reflector,
            {
                "context": "context",
                "citation_verifier": "citation_verifier",
            }
        )
        workflow.add_edge("citation_verifier", "synthesizer")
        workflow.add_edge("synthesizer", END)

        return workflow.compile()

    def run_stream(self, question: str):
        initial_state: ResearchState = {
            "question": question,
            "plan_steps": [],
            "current_step_idx": 0,
            "current_queries": [],
            "raw_sources": [],
            "top_evidence": [],
            "extracted_findings": [],
            "reflection_notes": "",
            "is_sufficient": False,
            "retry_count": 0,
            "max_retries": self.max_retries,
            "verified_evidence": "",
            "final_answer": "",
        }

        node_display_names = {
            "planner": "Planner: Breaking query into research angles",
            "context": "Context: Formulating academic search queries",
            "retriever": "Retriever: Hybrid search (BM25 + Qdrant + RRF + Reranker)",
            "reader": "Reader: Extracting technical findings & metrics",
            "reflector": "Reflector: Critiquing evidence completeness",
            "citation_verifier": "Citation Verifier: Grounding and verifying claims",
            "synthesizer": "Synthesizer: Writing Research Dossier & Timeline",
        }

        accumulated_sources = []
        final_dossier = ""
        current_step = 0

        yield {"type": "step_start", "step_key": "planner", "step_name": node_display_names["planner"]}

        for output in self.graph.stream(initial_state):
            for node_name, node_state in output.items():
                yield {"type": "step_complete", "step_key": f"{node_name}_{current_step}"}

                if node_name == "retriever":
                    sources = node_state.get("top_evidence", [])
                    accumulated_sources.extend(sources)
                    yield {"type": "sources", "sources": sources}

                elif node_name == "reflector":
                    if not node_state.get("is_sufficient", False) and node_state.get("retry_count", 0) <= self.max_retries:
                        yield {"type": "reflector_loop", "notes": node_state.get("reflection_notes", "")}
                        current_step += 1

                elif node_name == "synthesizer":
                    final_dossier = node_state.get("final_answer", "")

                next_key = f"{node_name}_{current_step}"
                yield {"type": "step_start", "step_key": next_key, "step_name": node_display_names.get(node_name, node_name)}

        unique_sources = []
        seen = set()
        for s in accumulated_sources:
            sid = s.get("id") or s.get("arxiv_id")
            if sid and sid not in seen:
                seen.add(sid)
                unique_sources.append(s)

        yield {
            "type": "final_answer",
            "answer": final_dossier,
            "total_sources": len(unique_sources),
            "all_sources": unique_sources,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Session State & Guest Login Authentication
# ─────────────────────────────────────────────────────────────────────────────
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
if "user_name" not in st.session_state:
    st.session_state.user_name = ""
if "gemini_api_key" not in st.session_state:
    st.session_state.gemini_api_key = os.environ.get("GEMINI_API_KEY", "")

# ── Guest Login Screen ────────────────────────────────────────────────────────
if not st.session_state.authenticated:
    st.markdown("""
    <div class="karp-login-wrap">
      <span class="karp-login-badge">✦ Guest Access</span>
      <div class="karp-login-title">Karpathy Deep Research</div>
      <div class="karp-login-desc">
        Welcome! Enter your researcher username and your Google Gemini API key to power autonomous literature synthesis, timeline generation, and hybrid retrieval.
      </div>
    </div>
    """, unsafe_allow_html=True)

    col_l, col_center, col_r = st.columns([1, 2, 1])
    with col_center:
        with st.form("guest_login_form"):
            guest_name_input = st.text_input(
                "Researcher Handle / Name:",
                value=st.session_state.user_name or "Guest Researcher",
                placeholder="e.g. Andrej / Alex / Research Guest",
                help="Your display name during research sessions.",
            )
            api_key_input = st.text_input(
                "Gemini API Key:",
                value=st.session_state.gemini_api_key,
                type="password",
                placeholder="AIzaSy...",
                help="Required for LangGraph agent planning and dossier synthesis.",
            )
            st.caption("🔑 Don't have an API key? Get a free key at [Google AI Studio](https://aistudio.google.com/app/apikey).")

            submit_btn = st.form_submit_button("🚀 Enter Research Console", use_container_width=True, type="primary")

            if submit_btn:
                if not api_key_input.strip():
                    st.error("Please enter a valid Gemini API Key to enable agent computations.")
                else:
                    st.session_state.user_name = guest_name_input.strip() or "Guest Researcher"
                    st.session_state.gemini_api_key = api_key_input.strip()
                    st.session_state.authenticated = True
                    genai.configure(api_key=st.session_state.gemini_api_key)
                    st.rerun()

    st.stop()


# Configure genai with active session key
genai.configure(api_key=st.session_state.gemini_api_key)


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar (Authenticated)
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(f"""
    <div class="karp-user-pill">
      <div class="karp-user-pill-name">
        <span>👤</span> {st.session_state.user_name}
      </div>
      <span class="karp-user-pill-badge">Online</span>
    </div>
    """, unsafe_allow_html=True)

    if st.button("🔑 Switch Key / Sign Out", use_container_width=True):
        st.session_state.authenticated = False
        st.session_state.has_run = False
        st.rerun()

    st.divider()

    st.markdown("### LangGraph Architecture")
    st.markdown("""
    **Workflow Graph:**
    - 📋 `Planner` (Decomposition)
    - 🔍 `Context` (Query Builder)
    - 🗄️ `Retriever` (Hybrid Retrieval)
    - 📖 `Reader` (Findings Extraction)
    - ⚖️ `Reflector` (Retry Loop)
    - 🛡️ `Citation Verifier` (Grounding)
    - ✍️ `Synthesizer` (Dossier)
    """)

    st.divider()

    st.markdown("### Hybrid Retrieval Subsystem")
    st.markdown("""
    - 📄 **BM25 Lexical Index** (`rank-bm25`)
    - 🌀 **Qdrant Vector Index** (`qdrant-client`)
    - 🔀 **RRF Fusion** ($k=60$)
    - 🎯 **Semantic Reranker**
    """)

    st.divider()

    st.markdown("### Engine Configuration")
    selected_model_display = st.selectbox(
        "LLM Backbone:",
        [
            "gemini-flash-lite-latest (Recommended: High Throughput / Stable)",
            "gemini-3.5-flash (Balanced: Speed & Intelligence)",
            "gemini-3-pro-preview (Warning: Strict Rate Limits of 2 RPM)",
        ],
        index=0,
        help="Select the underlying intelligence engine.",
    )
    actual_model_name = selected_model_display.split(" ")[0]


# ─────────────────────────────────────────────────────────────────────────────
# Main — Hero + Search
# ─────────────────────────────────────────────────────────────────────────────
if "has_run" not in st.session_state:
    st.session_state.has_run = False

if not st.session_state.has_run:
    st.markdown(f"""
    <div class="karp-hero">
      <div class="karp-logo-mark">✦ Karpathy</div>
      <div class="karp-tagline">Welcome back, {st.session_state.user_name} · LangGraph Deep Research · Hybrid Retrieval · Multi-Source</div>
      <div class="karp-source-pills">
        <span class="karp-src-pill arxiv">arXiv</span>
        <span class="karp-src-pill s2">Semantic Scholar</span>
        <span class="karp-src-pill crossref">CrossRef</span>
        <span class="karp-src-pill wiki">Wikipedia</span>
        <span class="karp-src-pill web">Web</span>
      </div>
      <div class="karp-suggest-label">Try asking about:</div>
      <div class="karp-chips">
        <span class="karp-chip">LLM agents & tool use</span>
        <span class="karp-chip">Transformer architectures</span>
        <span class="karp-chip">RLHF & alignment</span>
        <span class="karp-chip">Diffusion models</span>
        <span class="karp-chip">RAG systems</span>
        <span class="karp-chip">Chain-of-thought reasoning</span>
      </div>
    </div>
    """, unsafe_allow_html=True)

col_in, col_btn = st.columns([5, 1])
with col_in:
    question = st.text_input(
        "Research Query",
        label_visibility="collapsed",
        placeholder="What do you want to research?",
        key="query_input",
    )
with col_btn:
    run_btn = st.button("→", use_container_width=True, type="primary")


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline execution
# ─────────────────────────────────────────────────────────────────────────────
if run_btn:
    if not question.strip():
        st.warning("Please enter a research question.")
    else:
        st.session_state.has_run = True

        st.markdown(
            f'<div class="karp-query-head">{question}</div>',
            unsafe_allow_html=True,
        )

        trace_ph   = st.empty()
        sources_ph = st.empty()
        answer_ph  = st.empty()

        log_buf = io.StringIO()

        custom_model = genai.GenerativeModel(actual_model_name)
        agent = LangGraphResearchAgent(model=custom_model, max_retries=2)

        steps: list[dict] = []
        key_to_idx: dict[str, int] = {}
        final_answer   = None
        total_sources  = 0
        all_sources: list = []
        error_message  = None

        def _get_or_create(key: str, name: str) -> int:
            if key in key_to_idx:
                steps[key_to_idx[key]]["name"]   = name
                steps[key_to_idx[key]]["status"] = "active"
                return key_to_idx[key]
            idx = len(steps)
            steps.append({"key": key, "name": name, "status": "active", "sources": []})
            key_to_idx[key] = idx
            return idx

        with redirect_stdout(log_buf):
            try:
                for event in agent.run_stream(question):
                    etype = event["type"]

                    if etype == "step_start":
                        _get_or_create(event["step_key"], event["step_name"])
                        trace_ph.markdown(_render_trace(steps), unsafe_allow_html=True)

                    elif etype == "step_complete":
                        k = event["step_key"]
                        if k in key_to_idx:
                            steps[key_to_idx[k]]["status"] = "done"
                        trace_ph.markdown(_render_trace(steps), unsafe_allow_html=True)

                    elif etype == "sources":
                        for s in steps[::-1]:
                            if s["status"] == "active":
                                s["sources"].extend(event["sources"])
                                break
                        trace_ph.markdown(_render_trace(steps), unsafe_allow_html=True)

                    elif etype == "reflector_loop":
                        notes = event.get("notes", "")
                        st.toast(f"Reflector: Retrying retrieval ({notes[:60]}...)" if notes else "Reflector: Refining evidence...", icon="⚖️")

                    elif etype == "final_answer":
                        final_answer  = event["answer"]
                        total_sources = event.get("total_sources", 0)
                        all_sources   = event.get("all_sources", [])

                    elif etype == "error":
                        error_message = event["message"]

            except Exception as e:
                err = str(e).lower()
                if "429" in err or "quota" in err or "exhausted" in err:
                    error_message = f"⚠️ **API Quota Limit Hit.**\n\nPlease wait a moment or switch models. Error: {e}"
                else:
                    error_message = f"**System Error:** {e}"

        for s in steps:
            if s["status"] == "active":
                s["status"] = "done"

        trace_ph.markdown(_summary_pill(total_sources, len(steps)), unsafe_allow_html=True)

        if error_message:
            answer_ph.markdown(
                f'<div class="karp-error">{error_message}</div>',
                unsafe_allow_html=True,
            )
        elif final_answer:
            # 1. Show indexed sources grid at top
            sources_ph.markdown(_render_sources_index(all_sources), unsafe_allow_html=True)

            # 2. Typewriter stream of answer
            stream_ph = answer_ph.empty()
            streamed = ""
            for chunk in _stream_words(final_answer):
                streamed += chunk
                stream_ph.markdown(
                    f'<div class="karp-sec-body" style="padding:1rem">{_cites(streamed)}</div>',
                    unsafe_allow_html=True,
                )

            # 3. Replace with fully parsed sections (Timeline + SOTA + Tables)
            stream_ph.markdown(_render_answer(final_answer), unsafe_allow_html=True)
        else:
            answer_ph.markdown(
                '<div class="karp-error">No answer produced — the search returned no evidence.</div>',
                unsafe_allow_html=True,
            )

        with st.expander("Raw Debug Log"):
            st.code(log_buf.getvalue() or "(no output captured)", language="text")
