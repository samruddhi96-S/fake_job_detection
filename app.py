"""
Streamlit web app for Fake Job Detection.
Run:  streamlit run app.py
"""
import os

import streamlit as st

from predict import check_job, load_model

st.set_page_config(page_title="Fake Job Detection", page_icon="🔍")
st.title("🔍 Fake Job Detection")
st.write("Paste a job posting below. The system checks it using Machine Learning and scam red-flag rules.")

if not os.path.exists("model/model.pkl"):
    st.error("Model not found. Please run `python train.py` first.")
    st.stop()


@st.cache_resource
def get_model():
    return load_model()


model, tfidf = get_model()

example_fake = ("Urgent hiring! Earn Rs 50000 per week from home. No experience needed, no interview. "
                "Pay a small registration fee and contact us on WhatsApp. Email: hr.jobs@gmail.com")
example_real = ("We are looking for a Python Developer to join our engineering team. You will design and test "
                "features, take part in code reviews and sprint planning. Requirements: BTech in CS, 2 years "
                "experience, two technical interviews and one HR round.")

col1, col2 = st.columns(2)
if col1.button("Load fake example"):
    st.session_state["text"] = example_fake
if col2.button("Load real example"):
    st.session_state["text"] = example_real

text = st.text_area("Job description", key="text", height=250)

if st.button("Check Job", type="primary"):
    if not text.strip():
        st.warning("Please paste some text first.")
    else:
        label, prob, flags = check_job(text, model, tfidf)
        if label == "FAKE":
            st.error(f"⚠️ This job looks FAKE (ML fake probability: {prob:.1f}%)")
        else:
            st.success(f"✅ This job looks REAL (ML fake probability: {prob:.1f}%)")
        st.progress(min(int(prob), 100))

        if flags:
            st.subheader("Red flags found")
            for f in flags:
                st.write("🚩", f)
        else:
            st.write("No obvious red flags found.")

st.caption("Mini project demo. Always verify the company on its official website before applying.")
