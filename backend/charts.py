import streamlit as st
import pandas as pd


def score_chart(df: pd.DataFrame, threshold: int = 60):
    """Render score breakdown chart for all candidates."""

    chart_df = df[["Name", "Skills", "Experience", "Education", "Semantic"]].copy()
    chart_df = chart_df.set_index("Name")

    st.bar_chart(chart_df, height=350, use_container_width=True)
    st.caption(f"Each component score (0–100) shown per candidate. Shortlist threshold: **{threshold}**")


def weight_chart(weights: dict):
    """Render a small sidebar weight visualization."""
    df = pd.DataFrame({"Weight (%)": list(weights.values())}, index=list(weights.keys()))
    st.bar_chart(df, height=180, use_container_width=True)