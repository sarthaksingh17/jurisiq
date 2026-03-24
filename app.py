import streamlit as st
import requests
import time
import json

API = "http://localhost:8000"

st.set_page_config(page_title="JURIS-IQ", page_icon="⚖️", layout="wide")

st.markdown("""
    <h1 style='text-align:center;'>⚖️ JURIS-IQ</h1>
    <p style='text-align:center; color:gray;'>Multi-Agent Legal Contract Intelligence</p>
    <hr/>
""", unsafe_allow_html=True)

# ── sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.header("Upload Contract")

    if st.button("🔄 Reset"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()

    mode = st.radio("Input mode", ["Upload PDF", "Paste Text"])

    if mode == "Upload PDF":
        pdf = st.file_uploader("Drop your contract PDF", type=["pdf"])
        if pdf and st.button("Analyze", type="primary"):
            with st.spinner("Submitting..."):
                resp = requests.post(
                    f"{API}/upload",
                    files={"file": (pdf.name, pdf.getvalue(), "application/pdf")}
                )
                if resp.status_code == 200:
                    st.session_state["job_id"]  = resp.json()["job_id"]
                    st.session_state["verdict"] = None
                    st.success("Job started!")
                else:
                    st.error(f"Error: {resp.text}")

    else:
        text     = st.text_area("Paste contract text", height=300)
        doc_name = st.text_input("Document name", value="contract.txt")
        if st.button("Analyze", type="primary"):
            if text.strip():
                with st.spinner("Submitting..."):
                    resp = requests.post(
                        f"{API}/analyze",
                        json={"text": text, "doc_name": doc_name}
                    )
                    if resp.status_code == 200:
                        st.session_state["job_id"]  = resp.json()["job_id"]
                        st.session_state["verdict"] = None
                        st.success("Job started!")
                    else:
                        st.error(f"Error: {resp.text}")
            else:
                st.warning("Paste some contract text first.")

    if "job_id" in st.session_state:
        st.caption(f"Job ID: `{st.session_state['job_id'][:8]}...`")


# ── main area ─────────────────────────────────────────────────────────────────

if "job_id" not in st.session_state:
    st.info("Upload a contract PDF or paste text in the sidebar to begin.")
    st.stop()

job_id = st.session_state["job_id"]

# poll until verdict appears in DuckDB
if not st.session_state.get("verdict"):
    with st.spinner("Agents analyzing contract... this takes 30-60 seconds."):
        for _ in range(120):
            try:
                r = requests.get(f"{API}/verdict/{job_id}")
                if r.status_code == 200:
                    st.session_state["verdict"] = r.json()
                    break
            except:
                pass
            time.sleep(2)

verdict = st.session_state.get("verdict")

if not verdict:
    st.error("Analysis timed out or failed. Check the backend terminal.")
    st.stop()

# parse final_verdict JSON safely
try:
    final = json.loads(verdict["final_verdict"]) if verdict.get("final_verdict") else {}
except:
    final = {}

# ── scores ────────────────────────────────────────────────────────────────────

def score_color(score):
    if score is None:  return "⚪"
    if score >= 75:    return "🔴"
    if score >= 50:    return "🟡"
    return "🟢"

overall    = verdict.get("overall_score")    or 0
risk       = verdict.get("risk_score")       or 0
finance    = verdict.get("finance_score")    or 0
compliance = verdict.get("compliance_score") or 0

col1, col2, col3, col4 = st.columns(4)
col1.metric(f"{score_color(overall)} Overall Risk",  f"{overall}/100")
col2.metric(f"{score_color(risk)} Legal Risk",       f"{risk}/100")
col3.metric(f"{score_color(finance)} Finance Risk",  f"{finance}/100")
col4.metric(f"{score_color(compliance)} Compliance", f"{compliance}/100")

st.divider()

# ── recommendation ────────────────────────────────────────────────────────────

rec = final.get("final_recommendation") if final else None
if not rec:
    raw = verdict.get("final_verdict", "")
    if "reject" in raw.lower():
        rec = "reject"
    elif "approve_with_conditions" in raw.lower() or "conditions" in raw.lower():
        rec = "approve_with_conditions"
    elif "approve" in raw.lower():
        rec = "approve"
    else:
        rec = "unknown"

conf  = final.get("confidence", "N/A") if final else "N/A"
color = {"approve": "🟢", "approve_with_conditions": "🟡", "reject": "🔴"}.get(rec, "⚪")
st.markdown(f"### {color} Recommendation: `{rec.replace('_', ' ').upper()}`")
st.markdown(f"**Confidence:** {conf}%")

st.divider()

# ── top issues ────────────────────────────────────────────────────────────────

issues = final.get("top_issues", []) if final else []
if issues:
    st.subheader("Top Issues")
    for issue in issues:
        sev  = issue.get("severity", "medium")
        icon = {"high": "🔴", "medium": "🟡", "low": "🟢"}.get(sev, "⚪")
        with st.expander(f"{icon} {issue.get('issue', '')}"):
            st.write(f"**Severity:** {sev.upper()}")
            st.write(f"**Recommendation:** {issue.get('recommendation', '')}")

st.divider()

# ── judge reasoning ───────────────────────────────────────────────────────────

reasoning = final.get("reasoning") if final else verdict.get("final_verdict", "")
if reasoning:
    st.subheader("Judge's Reasoning")
    st.write(reasoning)

st.divider()

# ── debate log ────────────────────────────────────────────────────────────────

debate = verdict.get("debate_log")
if debate:
    try:
        rounds = json.loads(debate) if isinstance(debate, str) else debate
        st.subheader(f"Debate Log — {len(rounds)} arguments")
        for arg in rounds:
            agent     = arg.get("agent", "Agent")
            round_num = arg.get("round", "?")
            delta     = arg.get("confidence_delta", 0)
            arrow     = "↑" if delta > 0 else "↓"
            with st.expander(f"Round {round_num} — {agent} {arrow}{abs(delta)}"):
                st.write(arg.get("argument", ""))
    except:
        st.write(debate)

st.divider()

# ── audit footer ──────────────────────────────────────────────────────────────

st.caption(f"🔒 Audit hash: `{verdict.get('audit_hash', 'N/A')}`")
st.caption(f"📄 Document: `{verdict.get('doc_name', 'N/A')}`")
st.caption(f"🕐 Analyzed: `{verdict.get('created_at', 'N/A')}`")