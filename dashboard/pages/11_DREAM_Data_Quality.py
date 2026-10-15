"""DREAM aggregate coverage review; no model inference."""
from pathlib import Path
import json
import pandas as pd
import streamlit as st

st.title("DREAM Dataset: Data Quality")
path = Path(__file__).resolve().parents[2] / "docs/DREAM_DATA_QUALITY_EVIDENCE.json"
if not path.is_file():
    st.info("Run the DREAM audit documentation stage first.")
    st.stop()
evidence = json.loads(path.read_text(encoding="utf-8"))
st.caption("Separate external-data investigation; no diagnostic model evaluation")
cols = st.columns(3)
cols[0].metric("Parsed sessions", evidence["parsed_sessions"])
cols[1].metric("Participants", evidence["participants"])
cols[2].metric("Parsing failures", evidence["parsing_failures"])
st.subheader("Coordinate quality findings")
quality = pd.DataFrame(list(evidence["quality_status_counts"].items()), columns=["Category", "Sessions"])
st.dataframe(quality, hide_index=True, width="stretch")
st.bar_chart(quality.set_index("Category"), horizontal=True)
st.subheader("Corrected therapy-task counts")
st.dataframe(pd.DataFrame(list(evidence["task_ability_counts"].items()), columns=["Ability code", "Sessions"]), hide_index=True, width="stretch")
cohort = evidence["provisional_activity_cohort"]
st.write(f"Provisional activity cohort: {cohort['sessions']} sessions from {cohort['participants']} participants.")
st.caption(cohort["rule"])
st.info("Missing coordinates remain missing. The provisional cohort is not a validated training set. All participants have ASD; therapy conditions are not autism-negative labels. No diagnosis or severity predictions are made here.")
st.caption("Team: Muhammad Danial, Hamza Rasheed, Muhammad Freeh")
