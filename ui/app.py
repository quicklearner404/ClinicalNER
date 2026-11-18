"""Streamlit demo: paste a note, see drugs/chemicals and diseases highlighted.

Run:
    streamlit run ui/app.py   (expects the API at API_URL, default http://localhost:8000)
"""

from __future__ import annotations

import html
import json
import os

import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000")
COLORS = {"Chemical": "#cfe8ff", "Disease": "#ffd9d2"}
SAMPLE_NOTE = """DISCHARGE SUMMARY (synthetic example)
History: 64-year-old man with type 2 diabetes mellitus and hypertension, on metformin and lisinopril.
Hospital course: Admitted with community-acquired pneumonia and started on ceftriaxone and azithromycin.
On day 3 he developed acute kidney injury, attributed to dehydration; lisinopril was held.
He also reported nausea after azithromycin. Creatinine improved with IV fluids.
Discharge medications: metformin 500 mg twice daily, amoxicillin for 5 days. Lisinopril to be restarted at follow-up."""


def highlight(text: str, entities: list[dict]) -> str:
    parts, cursor = [], 0
    for ent in sorted(entities, key=lambda e: e["start"]):
        parts.append(html.escape(text[cursor : ent["start"]]))
        border = "border-bottom: 2px dashed #b45309;" if ent["needs_review"] else ""
        parts.append(
            f'<mark style="background:{COLORS[ent["label"]]};color:#111;padding:0 3px;border-radius:3px;{border}" '
            f'title="{ent["label"]} · {ent["score"]:.0%}">{html.escape(text[ent["start"] : ent["end"]])}'
            f'<sup style="font-size:0.65em;margin-left:2px">{ent["label"][0]}</sup></mark>'
        )
        cursor = ent["end"]
    parts.append(html.escape(text[cursor:]))
    return "<div style='white-space:pre-wrap;line-height:1.9'>" + "".join(parts) + "</div>"


st.set_page_config(page_title="ClinicalNER", page_icon="🩺", layout="wide")
st.title("ClinicalNER")
st.caption(
    "Extracts chemical/drug (C) and disease (D) mentions from free text and returns structured JSON. "
    "Runs on CPU; text never leaves the server. Dashed underline = low confidence, needs human review. "
    "Research prototype, not for diagnosis."
)

text = st.text_area("Clinical note", SAMPLE_NOTE, height=180)
min_score = st.slider("Minimum confidence", 0.0, 1.0, 0.5, 0.05)

if st.button("Extract entities", type="primary") and text.strip():
    try:
        response = requests.post(f"{API_URL}/v1/extract", json={"text": text, "min_score": min_score}, timeout=30)
        response.raise_for_status()
    except requests.RequestException as exc:
        st.error(f"Could not reach the API at {API_URL}: {exc}")
        st.stop()

    data = response.json()
    entities = data["entities"]
    st.markdown(highlight(text, entities), unsafe_allow_html=True)
    st.caption(f"{len(entities)} entities · {data['processing_ms']:.0f} ms · model {data['model']['version']}")

    if entities:
        st.dataframe(
            [{k: e[k] for k in ("text", "label", "score", "needs_review", "start", "end")} for e in entities],
            use_container_width=True,
            hide_index=True,
        )
    st.download_button("Download JSON", json.dumps(data, indent=2), "entities.json", "application/json")
