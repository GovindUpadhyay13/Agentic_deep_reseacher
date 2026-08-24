import streamlit as st
import io
import re
import time
from contextlib import redirect_stdout
import google.generativeai as genai

# ─────────────────────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Karpathy", page_icon="🔭", layout="wide")

# ─────────────────────────────────────────────────────────────────────────────
# Design system — Perplexity-inspired dark UI
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
@keyframes glow-pulse {
  0%,100% { box-shadow: 0 0 0 0 rgba(32,128,141,0); }
  50%      { box-shadow: 0 0 0 6px rgba(32,128,141,0.15); }
}

/* ── Hero (home state) ───────────────────────────────────────────────────── */
.karp-hero {
  text-align: center;
  padding: 5rem 1rem 1.5rem;
  animation: fadeUp 0.6s ease both;
}
.karp-logo-mark {
  font-size: 3rem;
  font-weight: 800;
  letter-spacing: -0.05em;
  background: linear-gradient(130deg, #20808d 0%, #6ecdd6 45%, #a8e8ed 100%);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
  line-height: 1;
  margin-bottom: 0.6rem;
}
.karp-tagline {
  font-size: 1rem;
  color: var(--t2);
  font-weight: 400;
  margin-bottom: 2.5rem;
  letter-spacing: 0.01em;
}
.karp-source-pills {
  display: flex;
  flex-wrap: wrap;
  gap: 0.4rem;
  justify-content: center;
  margin-bottom: 1.8rem;
}
.karp-src-pill {
  font-size: 0.7rem;
  font-weight: 600;
  letter-spacing: 0.04em;
  text-transform: uppercase;
  padding: 0.2rem 0.6rem;
  border-radius: 100px;
  border: 1px solid;
}
.karp-src-pill.arxiv    { color: var(--c-arxiv);   border-color: rgba(32,128,141,0.3);   background: rgba(32,128,141,0.06); }
.karp-src-pill.crossref { color: var(--c-crossref); border-color: rgba(224,123,42,0.3);   background: rgba(224,123,42,0.06); }
.karp-src-pill.s2       { color: var(--c-s2);       border-color: rgba(74,143,212,0.3);   background: rgba(74,143,212,0.06); }
.karp-src-pill.wiki     { color: var(--c-wiki);     border-color: rgba(113,113,122,0.3);  background: rgba(113,113,122,0.06); }
.karp-src-pill.web      { color: var(--c-web);      border-color: rgba(139,92,246,0.3);   background: rgba(139,92,246,0.06); }

.karp-suggest-label {
  font-size: 0.75rem;
  color: var(--t3);
  margin-bottom: 0.5rem;
}
.karp-chips {
  display: flex;
  flex-wrap: wrap;
  gap: 0.5rem;
  justify-content: center;
  margin-bottom: 1rem;
}
.karp-chip {
  background: var(--bg-card);
  border: 1px solid var(--border-hi);
  border-radius: 100px;
  padding: 0.3rem 0.9rem;
  font-size: 0.8rem;
  color: var(--t2);
  cursor: default;
  transition: border-color 0.15s, color 0.15s;
}
.karp-chip:hover { border-color: var(--accent); color: var(--t1); }

/* ── Search input ────────────────────────────────────────────────────────── */
div[data-testid="stTextInput"] input {
  background: var(--bg-surface) !important;
  border: 1.5px solid var(--border-hi) !important;
  border-radius: 100px !important;
  color: var(--t1) !important;
  font-size: 1rem !important;
  padding: 0.9rem 1.5rem !important;
  transition: border-color 0.2s, box-shadow 0.2s;
}
div[data-testid="stTextInput"] input:focus {
  border-color: var(--accent) !important;
  box-shadow: 0 0 0 4px var(--accent-glow) !important;
  outline: none !important;
}
div[data-testid="stTextInput"] input::placeholder { color: var(--t3) !important; }

/* ── Button ──────────────────────────────────────────────────────────────── */
div[data-testid="stButton"] > button[kind="primary"] {
  background: var(--accent) !important;
  border: none !important;
  border-radius: 100px !important;
  color: #fff !important;
  font-weight: 600 !important;
  font-size: 1rem !important;
  padding: 0.9rem 1.8rem !important;
  transition: background 0.2s, transform 0.1s;
}
div[data-testid="stButton"] > button[kind="primary"]:hover {
  background: var(--accent-2) !important;
  transform: translateY(-1px);
}

/* ── Query headline ──────────────────────────────────────────────────────── */
.karp-query-head {
  font-size: 1.75rem;
  font-weight: 700;
  letter-spacing: -0.03em;
  color: var(--t1);
  line-height: 1.2;
  margin: 1.8rem 0 1.2rem;
  animation: fadeUp 0.2s ease both;
}

/* ── Step trace ──────────────────────────────────────────────────────────── */
.karp-trace {
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-radius: var(--r-lg);
  padding: 0.9rem 1.1rem;
  margin-bottom: 1.2rem;
}
.karp-step { display: flex; align-items: flex-start; gap: 0.55rem; padding: 0.3rem 0; animation: fadeUp 0.2s ease both; }
.karp-step-dot { flex-shrink:0; width:16px; height:16px; border-radius:50%; margin-top:3px; display:flex; align-items:center; justify-content:center; font-size:9px; }
.karp-step-dot.done   { background: rgba(32,128,141,0.18); color: var(--accent); }
.karp-step-dot.active { background: var(--accent); animation: pulse-dot 1.2s ease infinite; }
.karp-step-lbl        { font-size: 0.875rem; font-weight:500; line-height:1.4; }
.karp-step-lbl.done   { color: var(--t2); }
.karp-step-lbl.active { color: var(--t1); }

/* ── Source cards (in trace) ─────────────────────────────────────────────── */
.karp-src-row { display:flex; flex-wrap:wrap; gap:0.4rem; margin-top:0.3rem; margin-left:1.3rem; }
.karp-src-card {
  background: var(--bg);
  border: 1px solid var(--border);
  border-radius: var(--r-sm);
  padding: 0.35rem 0.6rem;
  max-width: 200px;
  text-decoration: none;
  display: block;
  animation: fadeUp 0.3s ease both;
  transition: border-color 0.15s;
}
.karp-src-card:hover { border-color: var(--accent); }
.karp-src-card-title { font-size: 0.73rem; font-weight:500; color:var(--t1); display:-webkit-box; -webkit-line-clamp:2; -webkit-box-orient:vertical; overflow:hidden; margin-bottom:0.2rem; line-height:1.3; }
.karp-src-badge { font-size:0.63rem; font-weight:700; letter-spacing:0.04em; text-transform:uppercase; }
.badge-arxiv    { color: var(--c-arxiv); }
.badge-crossref { color: var(--c-crossref); }
.badge-s2       { color: var(--c-s2); }
.badge-wiki     { color: var(--c-wiki); }
.badge-web      { color: var(--c-web); }
.badge-local    { color: var(--c-local); }

/* ── Summary pill ────────────────────────────────────────────────────────── */
.karp-sum-pill {
  display:inline-flex; align-items:center; gap:0.4rem;
  background: var(--accent-glow);
  border: 1px solid rgba(32,128,141,0.3);
  border-radius: 100px;
  padding: 0.28rem 0.8rem;
  font-size: 0.78rem; color: var(--accent); font-weight:500;
  margin-bottom: 1rem;
}

/* ── Answer wrapper ──────────────────────────────────────────────────────── */
.karp-answer { margin-top: 0.25rem; }

/* Section blocks */
.karp-sec { margin-bottom: 1.6rem; animation: fadeUp 0.35s ease both; }
.karp-sec-hdr {
  display:flex; align-items:center; gap:0.5rem;
  padding-bottom: 0.55rem;
  border-bottom: 1px solid var(--border);
  margin-bottom: 0.8rem;
}
.karp-sec-hdr h3 { margin:0; font-size:0.92rem; font-weight:600; color:var(--t1); letter-spacing:-0.01em; }
.karp-sec-icon { font-size:0.95rem; }
.karp-sec-body { font-size:0.9rem; line-height:1.78; color:var(--t1); }
.karp-sec-body p  { margin:0 0 0.7rem; }
.karp-sec-body ul { margin:0 0 0.7rem; padding-left:1.35rem; }
.karp-sec-body li { margin-bottom:0.35rem; }
.karp-sec-body strong { font-weight:600; }
.karp-sec-body em     { font-style:italic; color:var(--t2); }
.karp-sec-body code   { font-family:'Courier New',monospace; background:rgba(255,255,255,0.06); padding:0.1em 0.35em; border-radius:3px; font-size:0.83em; }

/* Tinted section variants */
.karp-survey-body  { background:rgba(32,128,141,0.06);  border:1px solid rgba(32,128,141,0.14); border-radius:var(--r-md); padding:1rem 1.2rem; }
.karp-sota-body    { background:rgba(74,143,212,0.05);  border:1px solid rgba(74,143,212,0.12); border-radius:var(--r-md); padding:1rem 1.2rem; }
.karp-tips-body    { background:rgba(139,92,246,0.05);  border:1px solid rgba(139,92,246,0.12); border-radius:var(--r-md); padding:1rem 1.2rem; }

/* ── VISUAL TIMELINE ─────────────────────────────────────────────────────── */
.karp-tl {
  position: relative;
  padding-left: 3rem;
  margin: 0.8rem 0 0.5rem;
}
.karp-tl::before {
  content: '';
  position: absolute;
  left: 1.05rem;
  top: 0.3rem;
  bottom: 0.5rem;
  width: 2px;
  background: linear-gradient(to bottom, var(--accent) 0%, rgba(32,128,141,0.08) 100%);
}
.karp-tl-item {
  position: relative;
  margin-bottom: 1.7rem;
  animation: slideRight 0.35s ease both;
}
.karp-tl-item::before {
  content: '';
  position: absolute;
  left: -2.15rem;
  top: 0.55rem;
  width: 11px;
  height: 11px;
  border-radius: 50%;
  background: var(--accent);
  border: 2.5px solid var(--bg);
  box-shadow: 0 0 0 3px rgba(32,128,141,0.22);
}
.karp-tl-item.tl-undated::before { background: var(--c-web); box-shadow: 0 0 0 3px rgba(139,92,246,0.22); }

.karp-tl-year {
  font-size: 0.7rem;
  font-weight: 800;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  color: var(--accent);
  margin-bottom: 0.35rem;
}
.karp-tl-item.tl-undated .karp-tl-year { color: var(--c-web); }

.karp-tl-card {
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-radius: var(--r-md);
  padding: 0.85rem 1rem;
  font-size: 0.875rem;
  line-height: 1.7;
  color: var(--t1);
  transition: border-color 0.15s;
}
.karp-tl-card:hover { border-color: rgba(32,128,141,0.3); }
.karp-tl-card p     { margin: 0 0 0.5rem; }
.karp-tl-card p:last-child { margin-bottom: 0; }
.karp-tl-card ul    { margin: 0.2rem 0 0.5rem; padding-left: 1.15rem; }
.karp-tl-card li    { margin-bottom: 0.2rem; }
.karp-tl-card strong{ font-weight:600; }

/* Survey star badge */
.karp-survey-star {
  display:inline-flex; align-items:center; gap:0.3rem;
  background: rgba(32,128,141,0.12); border:1px solid rgba(32,128,141,0.3);
  border-radius:100px; padding:0.1rem 0.55rem;
  font-size:0.68rem; font-weight:700; color:var(--accent);
  letter-spacing:0.04em; text-transform:uppercase; margin-bottom:0.5rem;
}

/* ── Citation pill ───────────────────────────────────────────────────────── */
.karp-cite {
  display:inline-block;
  background: var(--accent-glow); border:1px solid rgba(32,128,141,0.28);
  border-radius:4px; padding:0.05em 0.32em;
  font-size:0.73em; font-family:'Courier New',monospace; color:var(--accent);
  text-decoration:none; margin:0 1px; vertical-align:middle;
  transition: background 0.15s;
}
.karp-cite:hover { background: rgba(32,128,141,0.22); }

/* ── Error ───────────────────────────────────────────────────────────────── */
.karp-error {
  background:rgba(239,68,68,0.07); border:1px solid rgba(239,68,68,0.22);
  border-radius:var(--r-md); padding:1rem 1.2rem;
  color:#fca5a5; font-size:0.88rem;
}

/* ── Expander ────────────────────────────────────────────────────────────── */
div[data-testid="stExpander"] {
  border:1px solid var(--border) !important;
  border-radius:var(--r-md) !important;
  background:var(--bg-surface) !important;
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

/* ── SOTA model cards ───────────────────────────────────────────────────── */
.karp-sota-grid { display:flex; flex-direction:column; gap:0.6rem; margin-top:0.5rem; }
.karp-sota-card {
  background: var(--bg-card); border:1px solid var(--border);
  border-left: 3px solid var(--c-s2);
  border-radius: var(--r-md); padding:0.75rem 1rem;
  animation: slideRight 0.3s ease both;
  transition: border-color 0.15s;
}
.karp-sota-card:hover { border-color: var(--c-s2); }
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
# Helper: render live step trace HTML
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
                title  = src.get("title", src["arxiv_id"])[:80]
                url    = src.get("url", f"https://arxiv.org/abs/{src['arxiv_id']}")
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
        f'✦ Searched {n_src} sources across {n_steps} steps'
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
# Helper: basic markdown → HTML (for answer sections)
# ─────────────────────────────────────────────────────────────────────────────
def _cites(text: str) -> str:
    """Replace [id] citations with styled pill links."""
    def _r(m):
        aid = m.group(1)
        url = f"https://arxiv.org/abs/{aid}" if re.match(r'\d{4}\.\d+', aid) else f"https://doi.org/{aid}"
        return f'<a class="karp-cite" href="{url}" target="_blank">[{aid}]</a>'
    # arXiv style: [2301.04567]
    text = re.sub(r'\[(\d{4}\.\d{4,6}(?:v\d+)?)\]', _r, text)
    # DOI / wiki style: [source:wiki:XXX] — just bold it
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

        # Markdown Table Row
        if s.startswith('|') and s.endswith('|'):
            if in_ul:
                out.append('</ul>'); in_ul = False
            cells = [c.strip() for c in s[1:-1].split('|')]
            if all(set(c).issubset({'-', ':', ' '}) for c in cells):
                # Divider line
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

        # Subheadings
        if s.startswith('#### '):
            if in_ul:
                out.append('</ul>'); in_ul = False
            out.append(f'<h5 style="margin:0.7rem 0 0.3rem;font-size:0.85rem;color:var(--t1);font-weight:600;">{_inline(s[5:])}</h5>')
        elif s.startswith('### '):
            if in_ul:
                out.append('</ul>'); in_ul = False
            out.append(f'<h4 style="margin:0.9rem 0 0.4rem;font-size:0.92rem;color:var(--t1);font-weight:700;">{_inline(s[4:])}</h4>')
        # Blockquote
        elif s.startswith('> '):
            if in_ul:
                out.append('</ul>'); in_ul = False
            out.append(f'<blockquote class="karp-quote">{_inline(s[2:])}</blockquote>')
        # Bullet list
        elif s.startswith('- ') or s.startswith('* '):
            if not in_ul:
                out.append('<ul>'); in_ul = True
            out.append(f'<li>{_inline(s[2:])}</li>')
        # Numbered list
        elif re.match(r'^\d+\.\s', s):
            if in_ul:
                out.append('</ul>'); in_ul = False
            content = re.sub(r'^\d+\.\s', '', s)
            out.append(f'<p style="margin-left:0.5rem"><strong>{s.split(".")[0]}.</strong> {_inline(content)}</p>')
        else:
            if in_ul:
                out.append('</ul>'); in_ul = False
            out.append(f'<p>{_inline(s)}</p>')

    if in_ul:
        out.append('</ul>')
    if in_table:
        out.append('</tbody></table></div>')
    return '\n'.join(out)


# ─────────────────────────────────────────────────────────────────────────────
# Helper: parse timeline section into visual HTML
# ─────────────────────────────────────────────────────────────────────────────
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
            # Plain content before first year entry
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
            # Text before first model card (e.g. "No SOTA models identified")
            preamble = f'<p class="karp-sec-body" style="margin-bottom:0.5rem">{_inline(entry)}</p>'
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
    """
    Parse the synthesizer's structured markdown (##/### headings) into
    beautiful HTML: Survey section, Visual Timeline, SOTA, Takeaways.
    Falls back to plain rendering if structure is missing.
    """
    text = text.strip()
    # Split on top-level ## headings
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
            # ── Visual Timeline ──────────────────────────────────────────────
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
            # ── Survey / Foundation ──────────────────────────────────────────
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
            # ── SOTA Models & Benchmark Comparison ────────────────────────────
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
            # ── Frontier, Failure Modes & Open Problems ───────────────────────
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
            # ── Takeaways & Synthesis ─────────────────────────────────────────
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
            # ── Generic section ───────────────────────────────────────────────
            clean = re.sub(r'^[^\w\s]+\s*', '', heading).strip()
            html += (
                f'<div class="karp-sec">'
                f'<div class="karp-sec-hdr"><h3>{clean}</h3></div>'
                f'<div class="karp-sec-body">{_md_to_html(body)}</div>'
                f'</div>'
            )

    if found_sections == 0:
        # No ## structure found — render as-is with citation pills
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
# Boot — load agent + index
# ─────────────────────────────────────────────────────────────────────────────
with st.status("Booting Karpathy...", expanded=True) as boot_status:
    st.write("Connecting to knowledge engine...")
    from phase2_agent import ResearchAgent, collection
    st.write("Ready.")
    boot_status.update(label="System Ready", state="complete", expanded=False)


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### Architecture Flags")
    use_planner   = st.checkbox("Planner (Query Rewriting)", value=True)
    use_reflector = st.checkbox("Reflector (Self-Critique)",  value=True)
    use_verifier  = st.checkbox("Citation Verifier",          value=True)

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

    st.divider()

    st.markdown("### System Stats")
    chunk_count  = collection.count() if collection is not None else 0
    index_status = str(chunk_count) if collection is not None else "Live API (no local index)"
    st.text(f"Sources: arXiv · S2 · CrossRef\n         Wikipedia · Web")
    st.text(f"Local Index: {index_status}")
    st.text(f"Engine:\n{actual_model_name}")


# ─────────────────────────────────────────────────────────────────────────────
# Main — Hero + Search
# ─────────────────────────────────────────────────────────────────────────────
if "has_run" not in st.session_state:
    st.session_state.has_run = False

if not st.session_state.has_run:
    # ── Hero (home state) ────────────────────────────────────────────────────
    st.markdown("""
    <div class="karp-hero">
      <div class="karp-logo-mark">✦ Karpathy</div>
      <div class="karp-tagline">Deep Research Intelligence · Timeline Synthesis · Multi-Source</div>
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

        # ── Query headline ────────────────────────────────────────────────────
        st.markdown(
            f'<div class="karp-query-head">{question}</div>',
            unsafe_allow_html=True,
        )

        # ── Placeholders ──────────────────────────────────────────────────────
        trace_ph   = st.empty()
        sources_ph = st.empty()
        answer_ph  = st.empty()

        # ── Log capture ───────────────────────────────────────────────────────
        log_buf = io.StringIO()

        # ── Agent ─────────────────────────────────────────────────────────────
        custom_model = genai.GenerativeModel(actual_model_name)
        agent = ResearchAgent(
            model=custom_model,
            collection=collection,
            max_steps=3,
            use_planner=use_planner,
            use_reflector=use_reflector,
            use_verifier=use_verifier,
        )

        # Step tracking
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

        # ── Stream loop ───────────────────────────────────────────────────────
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

                    elif etype == "final_answer":
                        final_answer  = event["answer"]
                        total_sources = event.get("total_sources", 0)
                        all_sources   = event.get("all_sources", [])

                    elif etype == "error":
                        error_message = event["message"]

            except Exception as e:
                err = str(e).lower()
                if "429" in err or "quota" in err or "exhausted" in err:
                    if "pro" in actual_model_name.lower():
                        error_message = (
                            "⚠️ **Rate Limit Exceeded (2 RPM)**\n\n"
                            "The Gemini Pro free tier allows only 2 requests per minute, "
                            "which is instantly exhausted by the Reflector loop.\n\n"
                            "**Fix:** Wait 60 seconds or switch to `gemini-flash-lite-latest`."
                        )
                    else:
                        error_message = f"⚠️ **API Quota Exceeded.**\n\nRaw error: {e}"
                else:
                    error_message = f"**System Error:** {e}"

        # ── Mark remaining active steps done ──────────────────────────────────
        for s in steps:
            if s["status"] == "active":
                s["status"] = "done"

        # ── Collapse trace to summary pill ────────────────────────────────────
        n_search = sum(1 for s in steps if s["key"].startswith("search_"))
        trace_ph.markdown(_summary_pill(total_sources, n_search), unsafe_allow_html=True)

        # ── Render sources index + answer ────────────────────────────────────
        if error_message:
            answer_ph.markdown(
                f'<div class="karp-error">{error_message}</div>',
                unsafe_allow_html=True,
            )
        elif final_answer:
            # Show indexed sources grid first
            sources_ph.markdown(_render_sources_index(all_sources), unsafe_allow_html=True)
            # 1. Typewriter stream of raw text
            stream_ph = answer_ph.empty()
            streamed = ""
            for chunk in _stream_words(final_answer):
                streamed += chunk
                stream_ph.markdown(
                    f'<div class="karp-sec-body" style="padding:1rem">{_cites(streamed)}</div>',
                    unsafe_allow_html=True,
                )
            # 2. Replace with fully-parsed timeline + SOTA + sections HTML
            stream_ph.markdown(_render_answer(final_answer), unsafe_allow_html=True)
        else:
            answer_ph.markdown(
                '<div class="karp-error">No answer produced — the search returned no evidence.</div>',
                unsafe_allow_html=True,
            )

        # ── Raw debug log ─────────────────────────────────────────────────────
        with st.expander("Raw Debug Log"):
            st.code(log_buf.getvalue() or "(no output captured)", language="text")