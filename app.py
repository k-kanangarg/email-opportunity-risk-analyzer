import streamlit as st
import joblib
import numpy as np
from scipy.sparse import hstack

from opportunity_rules import (
    detect_opportunity,
    detect_paid_risk,
    extract_links,
    extract_deadline
)

from opportunity_log import log_opportunity


# -----------------------------
# Load saved ML artifacts
# -----------------------------
model = joblib.load("spam_classifier.joblib")
vectorizer = joblib.load("tfidf_vectorizer.joblib")
threshold = joblib.load("decision_threshold.joblib")


# -----------------------------
# Feature extractor (same as training)
# -----------------------------
def extract_features(texts):
    return np.array([
        [len(t), sum(c.isdigit() for c in t)]
        for t in texts
    ])


# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="Spam & Opportunity Detector", layout="centered")

st.title("📨 Email Opportunity Analyzer")
st.write("Paste an email to analyze internship opportunities, risks, and spam likelihood.")

user_input = st.text_area("Message text", height=200)


# -----------------------------
# Button action
# -----------------------------
if st.button("Analyze email"):
    if user_input.strip() == "":
        st.warning("Please enter an email.")
    else:
        st.divider()
        st.subheader("📌 Opportunity Analysis")

        # ---- Opportunity logic ----
        is_opportunity = detect_opportunity(user_input)
        paid_risk = detect_paid_risk(user_input)
        links = extract_links(user_input)
        deadline = extract_deadline(user_input)

        if not is_opportunity:
            st.info("This email does not look like an internship or job opportunity.")
        else:
            st.success("Likely Internship / Career Opportunity")

            if paid_risk == "probably_paid":
                st.warning("⚠️ Possibly a paid or training-based program. Read terms carefully.")
            else:
                st.success("No obvious payment indicators detected.")

            if deadline:
                st.write(f"**Deadline hint:** {deadline}")

            st.markdown("### 📄 Opportunity Summary")
            st.markdown(f"""
            - **Type:** Internship / Career Program  
            - **Cost Risk:** {paid_risk.replace('_', ' ').title()}  
            - **Links Found:** {len(links)}  
            """)

            if links:
                st.write("**Application Links:**")
                for link in links:
                    st.markdown(f"- [{link}]({link})")
            else:
                st.write("No application links detected.")

            # 🔹 Log opportunity ONLY here
            log_opportunity(user_input, paid_risk, links)

        # -----------------------------
        # Spam probability (context, not gatekeeping)
        # -----------------------------
        st.divider()
        st.subheader("🛡️ Spam Analysis")

        text_vec = vectorizer.transform([user_input])
        struct_feat = extract_features([user_input])
        final_vec = hstack([text_vec, struct_feat])

        spam_prob = model.predict_proba(final_vec)[0][1]
        is_spam = spam_prob >= threshold

        st.progress(min(int(spam_prob * 100), 100))

        if is_spam:
            st.error(f"🚨 Spam detected (probability: {spam_prob:.2f})")
        else:
            st.success(f"✅ Not spam (spam probability: {spam_prob:.2f})")

        st.caption(f"Decision threshold: {threshold}")
