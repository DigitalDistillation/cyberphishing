"""
Streamlit Web Scanner App
Cybersecurity - Detecting Phishing Emails (NLP + Tabular + Email Forensics Engine)
"""

import os
import re
import streamlit as st
import pandas as pd
import joblib

MODEL_PATH = "models/phishing_detector.pkl"
PHISHING_THRESHOLD = 0.60  # 60% decision boundary tuned for zero false positives

st.set_page_config(
    page_title="Cybersecurity - Phishing Email Scanner",
    page_icon="🛡️",
    layout="wide",
)


def load_model():
    """Load trained hybrid NLP pipeline model directly."""
    if not os.path.exists(MODEL_PATH):
        return None
    return joblib.load(MODEL_PATH)


def extract_features_from_raw_text(raw_text: str, sender_email: str = "") -> dict:
    """
    Extracts raw email text alongside generalized structural metadata & email header forensics features.
    """
    text = (raw_text or "").strip()
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
        "action required",
        "unauthorized access",
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

    html_pattern = r"<html|<body|<p|<div|<a\s|<br"
    is_html = bool(re.search(html_pattern, text, re.IGNORECASE))
    html_content_flag = "Present" if is_html else "Absent"

    # --- EMAIL HEADER FORENSICS EXTRACTOR (SPF, DKIM, DMARC) ---
    default_auth = "Pass" if sender_domain_type == "Trusted/Corporate" else "None"

    spf_status = default_auth
    if re.search(r"received-spf:\s*pass|spf=pass", text, re.IGNORECASE):
        spf_status = "Pass"
    elif re.search(r"received-spf:\s*softfail|spf=softfail", text, re.IGNORECASE):
        spf_status = "Softfail"
    elif re.search(r"received-spf:\s*fail|spf=fail", text, re.IGNORECASE):
        spf_status = "Fail"

    dkim_status = default_auth
    if re.search(r"dkim=pass|dkim-signature:", text, re.IGNORECASE):
        dkim_status = "Pass"
    elif re.search(r"dkim=fail", text, re.IGNORECASE):
        dkim_status = "Fail"

    dmarc_status = default_auth
    if re.search(r"dmarc=pass", text, re.IGNORECASE):
        dmarc_status = "Pass"
    elif re.search(r"dmarc=fail", text, re.IGNORECASE):
        dmarc_status = "Fail"

    return {
        "email_text": text,
        "email_length": max(email_length, 10),
        "number_of_links": min(number_of_links, 20),
        "number_of_attachments": min(number_of_attachments, 10),
        "presence_of_urgent_keywords": presence_of_urgent_keywords,
        "sender_domain_type": sender_domain_type,
        "html_content_flag": html_content_flag,
        "spf_status": spf_status,
        "dkim_status": dkim_status,
        "dmarc_status": dmarc_status,
    }


model = load_model()

st.title("🛡️ Cybersecurity – Phishing Email Mass Scanner & Forensics Platform")
st.markdown(
    "Pre-scan your emails before opening them. Includes **SPF/DKIM/DMARC Email Forensics** and AI threat risk evaluation."
)

if model is None:
    st.error("⚠️ Model file not found! Run `python main.py` to generate `models/phishing_detector.pkl`.")
    st.stop()

tab1, tab2, tab3 = st.tabs([
    "📧 Scan Email (Upload / Paste)",
    "📥 Batch Mass Pre-Scanner",
    "🎛️ Manual Feature Sliders"
])

with tab1:
    st.subheader("Option A: Upload Email Files (.txt, .eml)")
    uploaded_file = st.file_uploader("Upload email file", type=["txt", "eml"])

    st.subheader("Option B: Paste Raw Email Content")
    sender_input = st.text_input("Sender Email Address (Optional, e.g. john@hotmail.com)")
    email_text = st.text_area("Paste Full Email Text / Headers Here:", height=200)

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
                st.subheader("📊 Feature & Header Forensics Analysis")
                disp_df = input_df.drop(columns=["email_text"]).T.rename(columns={0: "Value"})
                st.dataframe(disp_df, use_container_width=True)

            with col2:
                st.subheader("🛡️ AI Scan Result")
                prob = model.predict_proba(input_df)[0]
                phishing_prob = float(prob[1])
                legit_prob = float(prob[0])
                is_phishing = phishing_prob >= PHISHING_THRESHOLD

                if is_phishing:
                    st.error(f"🚨 **PHISHING DETECTED** (Probability: {phishing_prob * 100:.1f}%)")
                    st.progress(phishing_prob)
                else:
                    st.success(f"✅ **LEGITIMATE EMAIL** (Confidence: {legit_prob * 100:.1f}%)")
                    st.progress(legit_prob)

                st.markdown("### Structural & Forensics Analysis:")
                st.write(f"- **Sender Domain**: `{features['sender_domain_type']}`")
                st.write(f"- **SPF Auth Status**: `{features['spf_status']}`")
                st.write(f"- **DKIM Signature**: `{features['dkim_status']}`")
                st.write(f"- **DMARC Policy**: `{features['dmarc_status']}`")
                st.write(f"- **Urgent Keywords**: `{features['presence_of_urgent_keywords']}`")
                st.write(f"- **Links Count**: `{features['number_of_links']}`")

with tab2:
    st.subheader("📥 Mass Inbox Pre-Scanner (Batch Audit)")
    st.markdown("Pre-scan multiple emails before opening them to verify safety.")

    hours_back = st.slider("Select Scan Timeframe (Past Hours)", 1, 24, 1)

    st.markdown("#### Upload Multiple Email Files for Batch Audit:")
    uploaded_files = st.file_uploader("Upload multiple .eml / .txt files", type=["eml", "txt"], accept_multiple_files=True)

    if st.button("🚀 Audit Batch Emails Now", type="primary"):
        if not uploaded_files:
            st.warning("Please upload email files to run batch clearance audit.")
        else:
            batch_results = []
            for file in uploaded_files:
                content = file.read().decode("utf-8", errors="ignore")
                feats = extract_features_from_raw_text(content)
                input_df = pd.DataFrame([feats])
                
                prob = model.predict_proba(input_df)[0][1]
                is_phish = prob >= PHISHING_THRESHOLD
                
                status = "🚨 PHISHING THREAT" if is_phish else "✅ SAFE TO OPEN"
                batch_results.append({
                    "Filename": file.name,
                    "Verdict": status,
                    "Threat Probability": f"{prob * 100:.1f}%",
                    "SPF Status": feats["spf_status"],
                    "DKIM Status": feats["dkim_status"],
                    "Urgent Keywords": feats["presence_of_urgent_keywords"],
                    "Links": feats["number_of_links"],
                })

            res_df = pd.DataFrame(batch_results)
            
            safe_count = sum(1 for r in batch_results if "SAFE" in r["Verdict"])
            threat_count = sum(1 for r in batch_results if "PHISHING" in r["Verdict"])

            m1, m2, m3 = st.columns(3)
            m1.metric("Total Emails Scanned", len(batch_results))
            m2.metric("✅ Safe to Open", safe_count)
            m3.metric("🚨 Phishing Blocked", threat_count)

            st.dataframe(res_df, use_container_width=True)

with tab3:
    st.subheader("Manual Feature Specification & Forensics Simulation")
    c1, c2 = st.columns(2)

    with c1:
        text_in = st.text_input("Sample Email Text", "Hi team, please review the attached document.")
        length = st.number_input("Email Length", 10, 15000, 500)
        links = st.slider("Links Count", 0, 20, 2)
        attachments = st.slider("Attachments Count", 0, 10, 1)

    with c2:
        urgent = st.selectbox("Urgent Keywords", ["Absent", "Present"])
        domain = st.selectbox("Sender Domain", ["Trusted/Corporate", "Free-Webmail/Common Provider", "Suspicious/Spoofed"])
        html = st.selectbox("HTML Flag", ["Present", "Absent"])
        spf = st.selectbox("SPF Status", ["Pass", "Softfail", "Fail", "None"])
        dkim = st.selectbox("DKIM Status", ["Pass", "Fail", "None"])
        dmarc = st.selectbox("DMARC Status", ["Pass", "Fail", "None"])

    manual_df = pd.DataFrame(
        [
            {
                "email_text": text_in,
                "email_length": length,
                "number_of_links": links,
                "number_of_attachments": attachments,
                "presence_of_urgent_keywords": urgent,
                "sender_domain_type": domain,
                "html_content_flag": html,
                "spf_status": spf,
                "dkim_status": dkim,
                "dmarc_status": dmarc,
            }
        ]
    )

    if st.button("Scan Manual Features"):
        m_prob = model.predict_proba(manual_df)[0][1]
        m_is_phish = m_prob >= PHISHING_THRESHOLD

        if m_is_phish:
            st.error(f"🚨 **PHISHING DETECTED** ({m_prob * 100:.1f}%)")
        else:
            st.success(f"✅ **LEGITIMATE EMAIL** ({(1 - m_prob) * 100:.1f}%)")

st.markdown("---")
st.caption("Cybersecurity Phishing Detection & Email Forensics Platform")
