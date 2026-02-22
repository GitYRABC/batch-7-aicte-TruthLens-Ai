"""
╔══════════════════════════════════════════════════════════════════╗
║         TruthLens AI – Real-Time Fact Checker  v9.0             ║
║                                                                  ║
║  PIPELINE:                                                       ║
║    1. Input  → paste text OR upload PDF                          ║
║    2. Llama 3 extracts key factual claims                        ║
║    3. DuckDuckGo searches the web for each claim (free, no key)  ║
║    4. Llama 3 reasons over results → verdict + score + why       ║
║    5. Final overall verdict aggregated across all claims         ║
║                                                                  ║
║  Generative AI : Meta Llama 3.2-3B via HF Router (OpenAI SDK)   ║
║  Web Search    : DuckDuckGo — free, zero API key needed          ║
║  Session State : History + JSON export                           ║
╚══════════════════════════════════════════════════════════════════╝

How to run:
  1. Get a FREE HuggingFace token:
       huggingface.co → Settings → Access Tokens
       Permission needed: "Make calls to Inference Providers"
  2. Accept Meta Llama license:
       huggingface.co/meta-llama/Llama-3.2-3B-Instruct → Accept
  3. Create .env in this folder:
       HF_TOKEN=hf_xxxxxxxxxxxxxxxx
  4. pip install -r requirements.txt
  5. streamlit run truthlens_app.py

requirements.txt:
  streamlit>=1.32.0
  openai>=1.0.0
  python-dotenv>=1.0.0
  PyPDF2>=3.0.0
  requests>=2.31.0
  duckduckgo-search>=6.0.0
"""

import os
import json
import datetime
from io import BytesIO

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="TruthLens – Fact Checker",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history = []
# Remove sidebar token logic: always use env var
st.session_state.hf_token = os.environ.get("HF_TOKEN", "")

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG  — change only this line to swap Llama variants
# ─────────────────────────────────────────────────────────────────────────────
LLAMA_MODEL    = "meta-llama/Llama-3.2-3B-Instruct:hyperbolic"
HF_ROUTER_BASE = "https://router.huggingface.co/v1"

# ─────────────────────────────────────────────────────────────────────────────
# STYLES
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;700;800&family=DM+Mono:wght@400;500&family=DM+Sans:wght@300;400;500&display=swap');

html, body, [class*="css"] { font-family:'DM Sans',sans-serif; }
[data-testid="stAppViewContainer"] { background:#0d0f14; color:#e8e6e0; }
[data-testid="stSidebar"]          { background:#111318; border-right:1px solid #1e2029; }

.tl-title  { font-family:'Syne',sans-serif; font-weight:800; font-size:2.6rem;
             letter-spacing:-1px; color:#f0ede6; }
.tl-badge  { font-family:'DM Mono',monospace; font-size:0.65rem; color:#86efac;
             border:1px solid #86efac; padding:2px 8px; border-radius:20px;
             letter-spacing:1px; text-transform:uppercase; margin-left:12px; }
.tl-sub    { color:#6b7280; font-size:0.95rem; margin-bottom:28px; }
.tl-divider{ border:none; border-top:1px solid #1e2029; margin:20px 0; }

.card      { background:#13151c; border:1px solid #1e2029; border-radius:12px;
             padding:22px 26px; margin-bottom:16px; }
.card-label{ font-family:'DM Mono',monospace; font-size:0.68rem; color:#86efac;
             letter-spacing:2px; text-transform:uppercase; margin-bottom:10px; }
.card-body { color:#d1cfc8; font-size:0.95rem; line-height:1.8; }

.verdict-TRUE          { font-family:'Syne',sans-serif; font-size:2.2rem; font-weight:800; color:#86efac; }
.verdict-FALSE         { font-family:'Syne',sans-serif; font-size:2.2rem; font-weight:800; color:#f87171; }
.verdict-PARTIALLY-TRUE{ font-family:'Syne',sans-serif; font-size:2.2rem; font-weight:800; color:#fbbf24; }
.verdict-UNVERIFIABLE  { font-family:'Syne',sans-serif; font-size:2.2rem; font-weight:800; color:#6b7280; }

.bar-track { background:#1e2029; border-radius:999px; height:10px; margin:8px 0 4px; }
.bar-TRUE          { background:linear-gradient(90deg,#86efac,#34d399); border-radius:999px; height:10px; }
.bar-FALSE         { background:linear-gradient(90deg,#f87171,#ef4444); border-radius:999px; height:10px; }
.bar-PARTIALLY-TRUE{ background:linear-gradient(90deg,#fbbf24,#f59e0b); border-radius:999px; height:10px; }
.bar-UNVERIFIABLE  { background:#374151;                                 border-radius:999px; height:10px; }

.score-label { font-family:'DM Mono',monospace; font-size:0.78rem; color:#6b7280; }
.llama-badge { display:inline-block; background:#0f2a1a; border:1px solid #86efac;
               border-radius:6px; padding:2px 8px; font-family:'DM Mono',monospace;
               font-size:0.6rem; color:#86efac; margin-left:8px; }
.sidebar-section { font-family:'DM Mono',monospace; font-size:0.65rem; color:#4b5563;
                   letter-spacing:2px; text-transform:uppercase; margin:20px 0 8px 0; }

.stButton>button { background:linear-gradient(135deg,#86efac,#34d399); color:#0d0f14;
    font-family:'Syne',sans-serif; font-weight:700; border:none;
    border-radius:8px; padding:12px 28px; width:100%; font-size:1rem; }
.stButton>button:hover { opacity:0.85; }

[data-testid="stMetricValue"] { font-family:'Syne',sans-serif !important; color:#f0ede6 !important; }
[data-testid="stMetricLabel"] { font-family:'DM Mono',monospace !important; color:#6b7280 !important;
    font-size:0.7rem !important; letter-spacing:1px !important; text-transform:uppercase !important; }
textarea, input[type="text"], input[type="password"] {
    background:#13151c !important; border:1px solid #1e2029 !important;
    color:#e8e6e0 !important; border-radius:8px !important; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# OPTIONAL IMPORTS
# ─────────────────────────────────────────────────────────────────────────────
try:
    from openai import OpenAI
    OPENAI_OK = True
except ImportError:
    OPENAI_OK = False

try:
    import PyPDF2
    PDF_OK = True
except ImportError:
    PDF_OK = False

try:
    from duckduckgo_search import DDGS
    DDG_OK = True
except ImportError:
    DDG_OK = False

# ─────────────────────────────────────────────────────────────────────────────
# LLAMA 3 — via HuggingFace Router, OpenAI-compatible SDK
# ─────────────────────────────────────────────────────────────────────────────

def call_llama(system: str, user: str, max_tokens: int = 500) -> str:
    """Call Meta Llama 3 via HF Router using the OpenAI SDK."""
    if not OPENAI_OK:
        return "❌ Run: pip install openai"
    token = st.session_state.hf_token
    if not token:
        return "❌ Service unavailable, contact admin."
    try:
        client = OpenAI(base_url=HF_ROUTER_BASE, api_key=token)
        resp   = client.chat.completions.create(
            model=LLAMA_MODEL,
            messages=[
                {"role": "system", "content": system},
                {"role": "user",   "content": user},
            ],
            max_tokens=max_tokens,
            temperature=0.3,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        err = str(e)
        # Hide token if accidentally present in error message
        if token and token in err:
            err = err.replace(token, "[HIDDEN]")
        if "401" in err: return "❌ Invalid HF token."
        if "403" in err: return "❌ Accept Meta's Llama license at huggingface.co/meta-llama first."
        if "429" in err: return "⚠️ Rate limited — wait a moment and retry."
        return f"❌ API error: {err}"


# ─────────────────────────────────────────────────────────────────────────────
# PIPELINE STEP 1 — Extract key claims
# ─────────────────────────────────────────────────────────────────────────────

def extract_claims(text: str) -> list:
    raw = call_llama(
        system=(
            "You are a fact-checking assistant. "
            "Extract the 3 to 5 most important factual claims from the given text that can be verified online. "
            "Return ONLY a numbered list, one claim per line, no commentary. "
            "Each claim must be a short, self-contained, searchable statement."
        ),
        user=f"Text:\n\n{text[:2500]}",
        max_tokens=250,
    )
    claims = []
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        # Strip list markers
        for marker in ["1.","2.","3.","4.","5.","1)","2)","3)","4)","5)","-","*","•"]:
            if line.startswith(marker):
                line = line[len(marker):].strip()
                break
        if len(line) > 10:
            claims.append(line)
    return claims[:5]


# ─────────────────────────────────────────────────────────────────────────────
# PIPELINE STEP 2 — Search web via DuckDuckGo
# ─────────────────────────────────────────────────────────────────────────────

def search_web(claim: str, n: int = 4) -> str:
    """Search DuckDuckGo and return formatted results as a string."""
    if not DDG_OK:
        return "duckduckgo-search not installed. Run: pip install duckduckgo-search"
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(claim, max_results=n))
        if not results:
            return "No results found."
        lines = []
        for i, r in enumerate(results, 1):
            lines.append(
                f"[{i}] {r.get('title','')}\n"
                f"    {r.get('body','')[:300]}\n"
                f"    URL: {r.get('href','')}"
            )
        return "\n\n".join(lines)
    except Exception as e:
        return f"Search error: {e}"


# ─────────────────────────────────────────────────────────────────────────────
# PIPELINE STEP 3 — Llama reasons over search results for ONE claim
# ─────────────────────────────────────────────────────────────────────────────

def verify_claim(claim: str, search_results: str) -> dict:
    """Returns { verdict, score, reasoning } for a single claim."""
    raw = call_llama(
        system=(
            "You are an expert fact-checker. Given a claim and web search results, "
            "reason over the evidence carefully and respond in this EXACT format — "
            "no extra text before or after:\n\n"
            "VERDICT: [TRUE / FALSE / PARTIALLY TRUE / UNVERIFIABLE]\n"
            "SCORE: [0-100]  (100=definitely true, 0=definitely false, 50=uncertain)\n"
            "REASONING: [2-4 sentences explaining your verdict citing the evidence]"
        ),
        user=(
            f"CLAIM:\n{claim}\n\n"
            f"WEB SEARCH RESULTS:\n{search_results}"
        ),
        max_tokens=400,
    )

    verdict   = "UNVERIFIABLE"
    score     = 50
    reasoning = raw  # fallback

    for line in raw.splitlines():
        l = line.strip()
        if l.upper().startswith("VERDICT:"):
            v = l.split(":", 1)[1].strip().upper()
            if "FALSE" in v and "PARTIAL" not in v:
                verdict = "FALSE"
            elif "PARTIAL" in v:
                verdict = "PARTIALLY TRUE"
            elif "TRUE" in v:
                verdict = "TRUE"
            else:
                verdict = "UNVERIFIABLE"
        elif l.upper().startswith("SCORE:"):
            try:
                score = int("".join(c for c in l.split(":",1)[1] if c.isdigit())[:3])
                score = max(0, min(100, score))
            except Exception:
                pass
        elif l.upper().startswith("REASONING:"):
            reasoning = l.split(":", 1)[1].strip()

    return {"claim": claim, "verdict": verdict, "score": score, "reasoning": reasoning}


# ─────────────────────────────────────────────────────────────────────────────
# PIPELINE STEP 4 — Aggregate all claims → final overall verdict
# ─────────────────────────────────────────────────────────────────────────────

def aggregate_verdict(claim_results: list, original_text: str) -> dict:
    """Ask Llama to produce one overall verdict from all individual results."""
    summary = "\n\n".join(
        f"Claim {i}: {r['claim']}\n"
        f"  Verdict: {r['verdict']} | Score: {r['score']}/100\n"
        f"  Reasoning: {r['reasoning']}"
        for i, r in enumerate(claim_results, 1)
    )

    raw = call_llama(
        system=(
            "You are a senior fact-checker. Based on the individual claim verdicts below, "
            "give ONE overall verdict for the entire piece of content. "
            "Respond in this EXACT format:\n\n"
            "VERDICT: [TRUE / FALSE / PARTIALLY TRUE / UNVERIFIABLE]\n"
            "SCORE: [0-100]\n"
            "REASONING: [3-5 sentences summarising your conclusion, "
            "referencing which claims were true or false and why.]"
        ),
        user=(
            f"ORIGINAL TEXT (first 400 chars):\n{original_text[:400]}\n\n"
            f"INDIVIDUAL CLAIM RESULTS:\n{summary}"
        ),
        max_tokens=500,
    )

    verdict   = "UNVERIFIABLE"
    score     = 50
    reasoning = raw

    for line in raw.splitlines():
        l = line.strip()
        if l.upper().startswith("VERDICT:"):
            v = l.split(":", 1)[1].strip().upper()
            if "FALSE" in v and "PARTIAL" not in v:
                verdict = "FALSE"
            elif "PARTIAL" in v:
                verdict = "PARTIALLY TRUE"
            elif "TRUE" in v:
                verdict = "TRUE"
            else:
                verdict = "UNVERIFIABLE"
        elif l.upper().startswith("SCORE:"):
            try:
                score = int("".join(c for c in l.split(":",1)[1] if c.isdigit())[:3])
                score = max(0, min(100, score))
            except Exception:
                pass
        elif l.upper().startswith("REASONING:"):
            reasoning = l.split(":", 1)[1].strip()

    return {"verdict": verdict, "score": score, "reasoning": reasoning}


# ─────────────────────────────────────────────────────────────────────────────
# PDF TEXT EXTRACTION
# ─────────────────────────────────────────────────────────────────────────────

def extract_pdf_text(f) -> str:
    if not PDF_OK:
        return "PyPDF2 not installed. Run: pip install PyPDF2"
    try:
        reader = PyPDF2.PdfReader(BytesIO(f.read()))
        return "\n".join(p.extract_text() or "" for p in reader.pages).strip()
    except Exception as e:
        return f"PDF error: {e}"


# ─────────────────────────────────────────────────────────────────────────────
# SESSION HISTORY
# ─────────────────────────────────────────────────────────────────────────────

def save_to_history(text, overall, claims):
    st.session_state.history.insert(0, {
        "timestamp":  datetime.datetime.now().strftime("%H:%M:%S"),
        "date":       datetime.datetime.now().strftime("%d %b %Y"),
        "excerpt":    text[:120] + "...",
        "verdict":    overall["verdict"],
        "score":      overall["score"],
        "reasoning":  overall["reasoning"],
        "claims":     claims,
        "word_count": len(text.split()),
    })


# ─────────────────────────────────────────────────────────────────────────────
# UI HELPERS
# ─────────────────────────────────────────────────────────────────────────────

EMOJI = {
    "TRUE": "✅", "FALSE": "❌",
    "PARTIALLY TRUE": "⚠️", "UNVERIFIABLE": "❓"
}
COLOR = {
    "TRUE": "#86efac", "FALSE": "#f87171",
    "PARTIALLY TRUE": "#fbbf24", "UNVERIFIABLE": "#6b7280"
}
# CSS class key — spaces replaced with dash for class name
def vcss(v): return f"verdict-{v.replace(' ','-')}"
def bcss(v): return f"bar-{v.replace(' ','-')}"


def render_overall_card(overall):
    v     = overall["verdict"]
    score = overall["score"]
    col   = COLOR.get(v, "#6b7280")
    st.markdown(f"""
    <div class="card" style="border-left:4px solid {col};">
        <div class="card-label">Overall Verdict</div>
        <div class="{vcss(v)}">{EMOJI.get(v,'')} {v}</div>
        <div style="margin-top:16px;">
            <div class="score-label">Accuracy / Confidence Score: {score} / 100</div>
            <div class="bar-track"><div class="{bcss(v)}" style="width:{score}%;"></div></div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_reasoning_card(reasoning):
    st.markdown(f"""
    <div class="card">
        <div class="card-label">Why This Verdict
            <span class="llama-badge">🦙 Llama 3 · HF Router</span>
        </div>
        <div class="card-body">{reasoning}</div>
    </div>
    """, unsafe_allow_html=True)


def render_claim_card(i, r):
    v   = r["verdict"]
    col = COLOR.get(v, "#6b7280")
    st.markdown(f"""
    <div class="card" style="border-left:3px solid {col};">
        <div class="score-label" style="margin-bottom:6px;">CLAIM {i}</div>
        <div style="color:#e8e6e0;font-size:0.92rem;margin-bottom:10px;">
            {r['claim']}
        </div>
        <div style="font-family:Syne,sans-serif;font-weight:700;
                    font-size:1.05rem;color:{col};">
            {EMOJI.get(v,'')} {v}
        </div>
        <div style="margin-top:8px;">
            <div class="score-label">Score: {r['score']} / 100</div>
            <div class="bar-track"><div class="{bcss(v)}" style="width:{r['score']}%;"></div></div>
        </div>
        <div style="margin-top:10px;font-size:0.86rem;color:#6b7280;line-height:1.65;">
            {r['reasoning']}
        </div>
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────

def render_sidebar():
    with st.sidebar:
        st.markdown("""
        <div style="font-family:Syne,sans-serif;font-weight:800;font-size:1.3rem;
                    color:#86efac;margin-bottom:4px;">TruthLens AI</div>
        <div style="font-family:'DM Mono',monospace;font-size:0.65rem;color:#4b5563;
                    letter-spacing:2px;text-transform:uppercase;margin-bottom:20px;">
            Real-Time Fact Checker v9.0
        </div>
        """, unsafe_allow_html=True)

        # Pipeline overview
        st.markdown('<div class="sidebar-section">Pipeline</div>', unsafe_allow_html=True)
        for step in [
            "① Extract claims — Llama 3",
            "② Search web   — DuckDuckGo",
            "③ Verify each  — Llama 3",
            "④ Final verdict — Llama 3",
        ]:
            st.markdown(
                f'<div style="font-family:DM Mono,monospace;font-size:0.72rem;'
                f'color:#4b5563;margin:4px 0;">{step}</div>', unsafe_allow_html=True)

        # Session stats
        st.markdown('<div class="sidebar-section">Session</div>', unsafe_allow_html=True)
        n = len(st.session_state.history)
        st.markdown(
            f'<div style="font-family:DM Mono,monospace;font-size:0.75rem;color:#6b7280;">'
            f'📊 {n} checks this session</div>', unsafe_allow_html=True)

        # Status
        st.markdown('<div class="sidebar-section">Status</div>', unsafe_allow_html=True)
        for name, ok in [("OpenAI SDK", OPENAI_OK),
                         ("PyPDF2",     PDF_OK),
                         ("DuckDuckGo", DDG_OK)]:
            st.markdown(
                f'<div style="font-family:DM Mono,monospace;font-size:0.72rem;'
                f'color:#6b7280;margin:3px 0;">{"🟢" if ok else "🔴"} {name}</div>',
                unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    render_sidebar()

    # Header
    st.markdown("""
    <div style="margin-bottom:4px;">
        <span class="tl-title">TruthLens AI</span>
        <span class="tl-badge">Fact Checker · v9.0</span>
    </div>
    <div class="tl-sub">
        Paste text or upload a PDF → Llama 3 extracts claims →
        DuckDuckGo searches the web → Verdict + Score + Reasoning
    </div>
    """, unsafe_allow_html=True)

    tab_check, tab_history = st.tabs(["🔍 Fact Check", "📚 History"])

    # ══════════════════════════════════════════════════════════════════════════
    # FACT CHECK TAB
    # ══════════════════════════════════════════════════════════════════════════
    with tab_check:

        input_mode = st.radio(
            "Input", ["📝 Paste Text", "📄 Upload PDF"],
            horizontal=True, label_visibility="collapsed"
        )

        text = ""

        if input_mode == "📝 Paste Text":
            text = st.text_area(
                "Text to fact-check:",
                height=220,
                placeholder="Paste a news article, social media post, claim, or any text…",
            )

        else:
            if not PDF_OK:
                st.error("Run: pip install PyPDF2")
            else:
                f = st.file_uploader("Upload PDF", type=["pdf"])
                if f:
                    with st.spinner("Extracting text from PDF…"):
                        text = extract_pdf_text(f)
                    if text:
                        with st.expander("📄 Extracted Text"):
                            st.text(text[:3000])

        st.markdown('<hr class="tl-divider">', unsafe_allow_html=True)

        run = st.button("🔍  Fact Check Now")

        if not run:
            st.markdown("""
            <div style="text-align:center;padding:60px 0;">
                <div style="font-size:3rem;">🔍</div>
                <div style="font-family:'DM Mono',monospace;font-size:0.8rem;
                            letter-spacing:2px;color:#2a2d3a;text-transform:uppercase;
                            margin-top:12px;">Paste text or upload a PDF and click Fact Check</div>
            </div>
            """, unsafe_allow_html=True)
            return

        if not text or len(text.strip()) < 30:
            st.warning("⚠️ Please provide at least 30 characters of content.")
            return

        # Remove sidebar token check
        # if not st.session_state.hf_token:
        #     st.error("❌ Add your HuggingFace token in the sidebar first.")
        #     return

        # ── STEP 1: Extract claims ─────────────────────────────────────────
        st.markdown("## Results")
        with st.spinner("🦙 Step 1/4 — Llama 3 is extracting key claims…"):
            claims = extract_claims(text)

        if not claims:
            st.error("Could not extract claims. Try more specific text.")
            return

        st.markdown(f"""
        <div class="card">
            <div class="card-label">Claims Identified ({len(claims)})</div>
            {"".join(f'<div style="padding:6px 0;border-bottom:1px solid #1e2029;'
                     f'font-size:0.9rem;color:#a1a09a;">▸ {c}</div>' for c in claims)}
        </div>
        """, unsafe_allow_html=True)

        # ── STEP 2 + 3: Search + verify each claim ─────────────────────────
        claim_results = []
        progress = st.progress(0)
        status   = st.empty()

        for i, claim in enumerate(claims):
            status.markdown(
                f'<div style="font-family:DM Mono,monospace;font-size:0.8rem;color:#6b7280;">'
                f'🔎 Step 2-3/4 — Searching &amp; verifying claim {i+1}/{len(claims)}…</div>',
                unsafe_allow_html=True,
            )
            search_results = search_web(claim)
            result         = verify_claim(claim, search_results)
            claim_results.append(result)
            progress.progress((i + 1) / len(claims))

        status.empty()
        progress.empty()

        # ── STEP 4: Aggregate final verdict ───────────────────────────────
        with st.spinner("🦙 Step 4/4 — Llama 3 is producing final verdict…"):
            overall = aggregate_verdict(claim_results, text)

        st.markdown('<hr class="tl-divider">', unsafe_allow_html=True)

        # ── Render results ─────────────────────────────────────────────────
        left, right = st.columns([2, 3])

        with left:
            render_overall_card(overall)

            # Quick metric row
            c1, c2 = st.columns(2)
            with c1:
                st.metric("Score", f"{overall['score']} / 100")
            with c2:
                st.metric("Claims Checked", len(claim_results))

            render_reasoning_card(overall["reasoning"])

        with right:
            st.markdown(
                '<div class="card-label" style="margin-bottom:12px;">'
                'Claim-by-Claim Breakdown</div>',
                unsafe_allow_html=True,
            )
            for i, r in enumerate(claim_results, 1):
                render_claim_card(i, r)

        # ── Save to history ────────────────────────────────────────────────
        save_to_history(text, overall, claim_results)
        st.success(
            f"✅ Fact check complete · saved to history "
            f"({len(st.session_state.history)} total)"
        )

    # ══════════════════════════════════════════════════════════════════════════
    # HISTORY TAB
    # ══════════════════════════════════════════════════════════════════════════
    with tab_history:
        st.markdown("## 📚 History")
        n = len(st.session_state.history)

        if not st.session_state.history:
            st.markdown("""
            <div style="text-align:center;padding:60px 0;color:#2a2d3a;">
                <div style="font-size:2.5rem;">📭</div>
                <div style="font-family:DM Mono,monospace;font-size:0.8rem;
                            letter-spacing:2px;text-transform:uppercase;margin-top:12px;">
                    No fact checks yet
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            c1, c2 = st.columns([3, 1])
            with c1:
                st.download_button(
                    "⬇️ Export as JSON",
                    data=json.dumps(st.session_state.history, indent=2),
                    file_name=f"truthlens_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}.json",
                    mime="application/json",
                )
            with c2:
                if st.button("🗑️ Clear"):
                    st.session_state.history = []
                    st.rerun()

            st.markdown('<hr class="tl-divider">', unsafe_allow_html=True)

            for entry in st.session_state.history:
                v   = entry["verdict"]
                col = COLOR.get(v, "#6b7280")
                with st.expander(
                    f"{EMOJI.get(v,'')} {v}  ({entry['score']}/100)  ·  "
                    f"{entry['date']} {entry['timestamp']}  ·  "
                    f"{entry['word_count']} words"
                ):
                    st.metric("Verdict", f"{EMOJI.get(v,'')} {v}")
                    st.markdown(
                        f'<div class="card-label" style="margin-top:12px;">Overall Reasoning</div>'
                        f'<div class="card-body">{entry["reasoning"]}</div>',
                        unsafe_allow_html=True,
                    )
                    st.markdown(
                        f'<div class="card-label" style="margin-top:12px;">Excerpt</div>'
                        f'<div class="card-body" style="color:#6b7280;font-size:0.85rem;">'
                        f'{entry["excerpt"]}</div>',
                        unsafe_allow_html=True,
                    )
                    st.markdown(
                        f'<div class="card-label" style="margin-top:12px;">'
                        f'Claims ({len(entry["claims"])})</div>',
                        unsafe_allow_html=True,
                    )
                    for i, r in enumerate(entry["claims"], 1):
                        render_claim_card(i, r)

    # Footer
    st.markdown('<hr class="tl-divider">', unsafe_allow_html=True)
    st.markdown("""
    <div style="text-align:center;font-family:'DM Mono',monospace;font-size:0.65rem;
                color:#2a2d3a;letter-spacing:1px;padding:8px 0;">
        TRUTHLENS AI v9  ·  META LLAMA 3 (HF ROUTER)  ·  DUCKDUCKGO SEARCH  ·  FOR EDUCATIONAL USE
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__" or True:
    main()
