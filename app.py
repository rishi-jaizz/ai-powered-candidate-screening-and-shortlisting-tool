import streamlit as st
import pandas as pd
import pdfplumber

from backend.scorer import score_candidates
from backend.explainer import generate_explanation, generate_domain_section
from backend.bias_audit import run_bias_audit
from backend.charts import weight_chart

# ─────────────────────────────────────────────
st.set_page_config(
    page_title="AI Resume Shortlister",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Custom CSS ──────────────────────────────
st.markdown("""
<style>
    .metric-card {
        background: linear-gradient(135deg, #1e3a5f, #2d6a9f);
        border-radius: 12px;
        padding: 16px;
        color: white;
        text-align: center;
        margin: 4px;
    }
    .metric-card h2 { margin: 0; font-size: 2rem; }
    .metric-card p  { margin: 0; font-size: 0.85rem; opacity: 0.85; }

    .rank-badge {
        display: inline-block;
        background: #2d6a9f;
        color: white;
        border-radius: 50%;
        width: 30px; height: 30px;
        line-height: 30px;
        text-align: center;
        font-weight: bold;
        margin-right: 8px;
    }
    .score-bar-wrap { background: #e0e0e0; border-radius: 8px; height: 10px; margin: 4px 0; }
    .score-bar      { border-radius: 8px; height: 10px; }

    .shortlisted   { border-left: 4px solid #2ecc71; }
    .not-shortlist { border-left: 4px solid #e74c3c; }

    [data-testid="stExpander"] { border-radius: 10px; }
    .stAlert { border-radius: 10px; }
</style>
""", unsafe_allow_html=True)

# ─── Title ───────────────────────────────────
st.title("🚀 AI Resume Shortlister")
st.caption("Upload resumes → Score against JD → Adjust weights → Get ranked shortlist + explanations + bias audit")

# ═══════════════════════════════════════════
# SIDEBAR — Weights & Threshold
# ═══════════════════════════════════════════
with st.sidebar:
    st.header("⚖️ Scoring Weights")
    st.caption("Drag sliders to re-prioritise criteria. Weights auto-normalize to 100%.")

    skill_w    = st.slider("🛠️  Skills Match",     0, 100, 40)
    exp_w      = st.slider("💼  Experience",        0, 100, 30)
    edu_w      = st.slider("🎓  Education",         0, 100, 20)
    semantic_w = st.slider("🔍  Semantic Relevance",0, 100, 10)

    total_w = skill_w + exp_w + edu_w + semantic_w
    if total_w == 0:
        st.error("All weights are 0 — set at least one above 0.")
    else:
        st.info(f"Total weight: **{total_w}** (auto-normalized to 100%)")

    st.divider()
    st.header("🎯 Shortlist Threshold")
    threshold = st.slider("Min ATS Score to shortlist", 0, 100, 60)
    st.caption(f"Candidates scoring **≥ {threshold}** will be shortlisted.")

    st.divider()
    st.markdown("**Weight Preview**")
    if total_w > 0:
        weight_chart({
            "Skills": skill_w, "Experience": exp_w,
            "Education": edu_w, "Semantic": semantic_w
        })

# ═══════════════════════════════════════════
# MAIN — Input
# ═══════════════════════════════════════════
st.header("📥 Input")
col1, col2 = st.columns([1, 1])

with col1:
    jd_text = st.text_area(
        "📌 Paste Job Description",
        height=260,
        placeholder="e.g. We are looking for a Python developer with 3+ years experience in ML/AI, FastAPI, Docker..."
    )

with col2:
    resumes = st.file_uploader(
        "📂 Upload Resumes (PDF)",
        type=["pdf"],
        accept_multiple_files=True,
        help="Upload 1–15 PDF resumes"
    )
    if resumes:
        st.success(f"✅ {len(resumes)} resume(s) uploaded")
        for f in resumes:
            st.caption(f"• {f.name}")

# ═══════════════════════════════════════════
# RUN BUTTON
# ═══════════════════════════════════════════
st.divider()
run = st.button("🚀 Screen Candidates", use_container_width=True, type="primary")

if run:

    # ── Validation ────────────────────────
    if not jd_text.strip():
        st.warning("⚠️ Please paste a Job Description first.")
        st.stop()
    if not resumes:
        st.warning("⚠️ Please upload at least one resume PDF.")
        st.stop()
    if total_w == 0:
        st.error("⚠️ All scoring weights are 0. Adjust the sliders in the sidebar.")
        st.stop()

    # ── Process resumes ───────────────────
    results = []
    progress = st.progress(0, text="Processing resumes…")

    for i, file in enumerate(resumes):
        progress.progress((i) / len(resumes), text=f"Processing: {file.name}")

        # Extract text from PDF
        try:
            with pdfplumber.open(file) as pdf:
                text = ""
                for page in pdf.pages:
                    text += (page.extract_text() or "")
        except Exception as e:
            st.warning(f"Could not read {file.name}: {e}")
            continue

        if not text.strip():
            st.warning(f"No text extracted from {file.name} — skipping.")
            continue

        # Score
        score_data = score_candidates(
            text, jd_text,
            skill_w / 100, exp_w / 100, edu_w / 100, semantic_w / 100
        )

        # Generate explanation
        explanation = generate_explanation(score_data, file.name, jd_text)

        domain_section = generate_domain_section(score_data.get("_domain_result", {}))
        results.append({
            "Name":          file.name.replace(".pdf", ""),
            "ATS Score":     score_data["final"],
            "Skills":        score_data["skills"],
            "Experience":    score_data["experience"],
            "Education":     score_data["education"],
            "Semantic":      score_data["semantic"],
            "Domain":        score_data["domain"],
            "Shortlisted":   score_data["final"] >= threshold,
            "Explanation":   explanation,
            "DomainSection": domain_section,
            "DomainSummary": score_data.get("_domain_result", {}).get("summary", ""),
            "Breakdown":     score_data,
            "Raw Text":      text[:300]
        })

    progress.progress(1.0, text="Done!")

    if not results:
        st.error("No resumes could be processed.")
        st.stop()

    # ── Build DataFrame ───────────────────
    df = pd.DataFrame(results).sort_values("ATS Score", ascending=False).reset_index(drop=True)
    df.index = df.index + 1  # 1-based rank

    shortlisted_df = df[df["Shortlisted"] == True]
    rejected_df    = df[df["Shortlisted"] == False]

    # ═══════════════════════════════════════
    # SECTION 1 — Summary KPIs
    # ═══════════════════════════════════════
    st.header("📊 Summary")

    k1, k2, k3, k4 = st.columns(4)
    k1.markdown(f"""<div class="metric-card"><h2>{len(df)}</h2><p>Total Resumes</p></div>""", unsafe_allow_html=True)
    k2.markdown(f"""<div class="metric-card"><h2>{len(shortlisted_df)}</h2><p>Shortlisted ✅</p></div>""", unsafe_allow_html=True)
    k3.markdown(f"""<div class="metric-card"><h2>{len(rejected_df)}</h2><p>Not Shortlisted ❌</p></div>""", unsafe_allow_html=True)
    avg = round(df["ATS Score"].mean(), 1)
    k4.markdown(f"""<div class="metric-card"><h2>{avg}</h2><p>Avg ATS Score</p></div>""", unsafe_allow_html=True)

    # ═══════════════════════════════════════

    # ═══════════════════════════════════════
    # SECTION 3 — Ranked Shortlist Table
    # ═══════════════════════════════════════
    st.header("🏆 Ranked Shortlist")

    display_df = df[["Name", "ATS Score", "Skills", "Experience", "Education", "Semantic", "Domain", "Shortlisted"]].copy()
    display_df["Rank"] = range(1, len(display_df) + 1)
    display_df["Status"] = display_df["Shortlisted"].map({True: "✅ Shortlisted", False: "❌ Rejected"})
    display_df = display_df[["Rank", "Name", "ATS Score", "Skills", "Experience", "Education", "Semantic", "Domain", "Status"]]

    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "ATS Score":  st.column_config.ProgressColumn("ATS Score", min_value=0, max_value=100, format="%.1f"),
            "Skills":     st.column_config.ProgressColumn("Skills",    min_value=0, max_value=100, format="%.1f"),
            "Experience": st.column_config.ProgressColumn("Experience",min_value=0, max_value=100, format="%.1f"),
            "Education":  st.column_config.ProgressColumn("Education", min_value=0, max_value=100, format="%.1f"),
            "Semantic":   st.column_config.ProgressColumn("Semantic",  min_value=0, max_value=100, format="%.1f"),
            "Domain":    st.column_config.ProgressColumn("Domain",    min_value=0, max_value=100, format="%.1f"),
        }
    )

    # Download button
    csv = display_df.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Download Results CSV", csv, "shortlist_results.csv", "text/csv")

    # ═══════════════════════════════════════
    # SECTION 4 — Detailed Candidate Cards
    # ═══════════════════════════════════════
    st.header("🧠 Candidate Explanations")

    tab1, tab2 = st.tabs([f"✅ Shortlisted ({len(shortlisted_df)})", f"❌ Not Shortlisted ({len(rejected_df)})"])

    def render_candidate_card(row, rank):
        with st.expander(f"#{rank}  {row['Name']}  —  ATS Score: **{row['ATS Score']}**"):
            c1, c2 = st.columns([1, 2])
            with c1:
                st.metric("🏆 ATS Score", f"{row['ATS Score']}/100")
                st.metric("🛠️ Skills",     f"{row['Skills']}/100")
                st.metric("💼 Experience", f"{row['Experience']}/100")
                st.metric("🎓 Education",  f"{row['Education']}/100")
                st.metric("🔍 Semantic",   f"{row['Semantic']}/100")
                st.metric("🎯 Domain",     f"{row['Domain']}/100")
            with c2:
                st.markdown("**📋 AI Explanation**")
                st.info(row["Explanation"])
                st.markdown("---")
                st.markdown(row["DomainSection"])

    with tab1:
        if shortlisted_df.empty:
            st.info("No candidates met the shortlist threshold. Try lowering it in the sidebar.")
        for rank, (_, row) in enumerate(shortlisted_df.iterrows(), 1):
            render_candidate_card(row, rank)

    with tab2:
        if rejected_df.empty:
            st.success("All candidates were shortlisted! 🎉")
        for rank, (_, row) in enumerate(rejected_df.iterrows(), 1):
            render_candidate_card(row, len(shortlisted_df) + rank)

    # ═══════════════════════════════════════
    # SECTION 5 — Bias Audit
    # ═══════════════════════════════════════
    st.header("⚠️ Bias Audit")

    bias_report = run_bias_audit(df, skill_w, exp_w, edu_w, semantic_w, threshold)

    if not bias_report["issues"]:
        st.success("✅ No major bias patterns detected in this screening run.")
    else:
        for issue in bias_report["issues"]:
            st.warning(f"⚠️ {issue}")

    with st.expander("📘 Bias Audit Details"):
        st.markdown(bias_report["details"])