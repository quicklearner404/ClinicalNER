import streamlit as st
import requests

st.set_page_config(page_title="Clinical NER Agent", page_icon="🏥")

st.title("🏥 Clinical NER AI Agent")

user_input = st.text_area("Enter Medical Notes:", "Patient was prescribed 50mg of Advil for heart health.")

if st.button("Run Clinical Extraction"):
    if user_input:
        try:
            # Call your FastAPI
            response = requests.post("http://127.0.0.1:8000/extract", json={"text": user_input})
            data = response.json()
            raw_entities = data.get("entities", [])
            
            # 60% Confidence & Blacklist
            # Look for this line in src/app.py and update it:
            blacklist = ["patient", "health", "hospital", "doctor", "clinic", "heart", "was", "for"]
            filtered = [e for e in raw_entities if e['score'] >= 0.60 and e['word'].lower().strip() not in blacklist]
            
            if filtered:
                st.subheader("Verified Clinical Results")
                for ent in filtered:
                    st.success(f"**{ent['word'].capitalize()}** | Confidence: {ent['score']:.2%}")
            else:
                st.warning("No entities met the 60% confidence threshold.")
        except Exception as e:
            st.error(f"Backend Offline! Error: {e}")