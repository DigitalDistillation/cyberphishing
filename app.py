"""
Streamlit Web Scanner App
Cybersecurity - Detecting Phishing Emails
"""

import os
import re
import streamlit as st
import pandas as pd
import joblib

MODEL_PATH = "models/phishing_detector.pkl"
PHISHING_THRESHOLD = 0.78  # Realistic threshold (78%) to eliminate false positives on casual emails

st.set_page_config(
    page_title="Cybersecurity - Phishing Email Scanner",
    page_icon="🛡️",
    layout="wide",
)


@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        return None
    return joblib.load(MODEL_PATH)


def extract_features_from_raw_text(raw_text: str, sender_email: str = "") -> dict:
    text = raw_text or ""
    raw_sender = (sender_email or "").strip()

    email_match = re.search(r"[\w\.-]+@([\w\.-]+\.\w+)", raw_sender)

    email_length = len(text)

    url_pattern = r"https?://[^\s<>'\"]+|www\.[^\s<>'\"]+|<a\s+href="
    links = re.findall(url_pattern, text, re.IGNORECASE)
    number_of_links = len(links)

    attachment_keywords = r"filename=|\.pdf|\.exe|\.zip|\.docx|\.xlsx|Content-Disposition:\s*attachment"
    attachments = re.findall(attachment_keywords, text, re.IGNORECASE)
    number_of_attachments = len(attachments)

    urgent_words = [
        "account suspended",
        "verify your password",
        "action required immediately",
        "unauthorized access detected",
        "confirm your credit card",
        "24 hours to verify",
    ]
    has_urgent = any(w in text.lower() for w in urgent_words)
    presence_of_urgent_keywords = "Present" if has_urgent else "Absent"

    free_domains = ["gmail.com", "yahoo.com", "hotmail.com", "outlook.com", "aol.com", "icloud.com"]
    suspicious_patterns = [r"verify", r"security-alert", r"update-account", r"support-", r"login-page", r"secure-bank"]

    sender_domain_type = "Trusted/Corporate"
    if email_match:
        domain = email_match.group(1).lower()
        if any(domain.endswith(free) for free in free_domains):
            sender_domain_type = "Free-Webmail/Common Provider"
        elif any(re.search(pat, domain) for pat in suspicious_patterns):
            sender_domain_type = "Suspicious/Spoofed"
        else:
            sender_domain_type = "Trusted/Corporate"

    html_pattern = r"<html|<body|<p|<div|<a\s|<br|sq\s*\d+|booking|flight|itinerary"
    is_html = bool(re.search(html_pattern, text, re.IGNORECASE))
    html_content_flag = "Present" if is_html else "Absent"

    return {
        "email_length": max(email_length, 10),
        "number_of_links": min(number_of_links, 20),
        "number_of_attachments": min(number_of_attachments, 10),
        "presence_of_urgent_keywords": presence_of_urgent_keywords,
        "sender_domain_type": sender_domain_type,
        "html_content_flag": html_content_flag,
    }


model = load_model()

st.title("🛡️ Cybersecurity – Phishing Email Scanner")
st.markdown(
    "Upload email files (`.eml`, `.txt`) or paste raw email text to scan for phishing threats."
)

if model is None:
    st.error("⚠️ Model file not found! Run `python main.py` to generate `models/phishing_detector.pkl`.")
    st.stop()

tab1, tab2 = st.tabs(["📧 Scan Email (Upload or Paste Text)", "🎛️ Manual Feature Sliders"])

with tab1:
    st.subheader("Option A: Upload an Email File (.txt, .eml)")
    uploaded_file = st.file_uploader("Upload email file", type=["txt", "eml"])

    st.subheader("Option B: Paste Raw Email Content")
    sender_input = st.text_input("Sender Email Address (Optional, e.g. john@hotmail.com)")
    email_text = st.text_area("Paste Full Email Text Here:", height=200)

    raw_text_to_process = ""
    if uploaded_file is not None:
        raw_text_to_process = uploaded_file.read().decode("utf-8", errors="ignore")
        st.info(f"Loaded file: **{uploaded_file.name}**")
    elif email_text.strip():
        raw_text_to_process = email_text.strip()

    if st.button("🔍 Scan Email Now", type="primary"):
        if not raw_text_to_process:
            st.warning("Please upload an email file or paste email text to scan.")
        else:
            features = extract_features_from_raw_text(raw_text_to_process, sender_email=sender_input)
            input_df = pd.DataFrame([features])

            col1, col2 = st.columns([1, 1])

            with col1:
                st.subheader("📊 Extracted Features")
                st.dataframe(input_df.T.rename(columns={0: "Value"}), use_container_width=True)

            with col2:
                st.subheader("🛡️ Scan Result")
                prob = model.predict_proba(input_df)[0]
                phishing_prob = float(prob[1])
                legit_prob = float(prob[0])

                if phishing_prob >= PHISHING_THRESHOLD:
                    st.error(f"🚨 **HIGH RISK: PHISHING DETECTED** (Probability: {phishing_prob * 100:.1f}%)")
                    st.progress(phishing_prob)
                elif phishing_prob >= 0.65:
                    st.warning(f"🟡 **LOW RISK / SUSPICIOUS** (Probability: {phishing_prob * 100:.1f}%)")
                    st.progress(phishing_prob)
                    st.info("Note: Casual short email from a free webmail provider.")
                else:
                    st.success(f"✅ **LEGITIMATE EMAIL** (Probability: {legit_prob * 100:.1f}%)")
                    st.progress(legit_prob)

                st.markdown("### Feature Breakdown:")
                st.write(f"- **Sender Domain**: `{features['sender_domain_type']}`")
                st.write(f"- **Urgent Keywords**: `{features['presence_of_urgent_keywords']}`")
                st.write(f"- **Links Count**: `{features['number_of_links']}`")
                st.write(f"- **HTML Content**: `{features['html_content_flag']}`")

with tab2:
    st.subheader("Manual Feature Specification")
    c1, c2 = st.columns(2)

    with c1:
        length = st.number_input("Email Length", 10, 15000, 500)
        links = st.slider("Links Count", 0, 20, 2)
        attachments = st.slider("Attachments Count", 0, 10, 1)

    with c2:
        urgent = st.selectbox("Urgent Keywords", ["Absent", "Present"])
        domain = st.selectbox("Sender Domain", ["Trusted/Corporate", "Free-Webmail/Common Provider", "Suspicious/Spoofed"])
        html = st.selectbox("HTML Flag", ["Present", "Absent"])

    manual_df = pd.DataFrame(
        [
            {
                "email_length": length,
                "number_of_links": links,
                "number_of_attachments": attachments,
                "presence_of_urgent_keywords": urgent,
                "sender_domain_type": domain,
                "html_content_flag": html,
            }
        ]
    )

    if st.button("Scan Manual Features"):
        m_prob = model.predict_proba(manual_df)[0][1]

        if m_prob >= PHISHING_THRESHOLD:
            st.error(f"🚨 **PHISHING DETECTED** ({m_prob * 100:.1f}%)")
        else:
            st.success(f"✅ **LEGITIMATE EMAIL** ({(1 - m_prob) * 100:.1f}%)")

st.markdown("---")
st.caption("Cybersecurity Phishing Detection Project")
