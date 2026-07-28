"""
Email Scanner Inference Script (Supports CLI text/file parsing)
Cybersecurity - Detecting Phishing Emails

Scans emails by passing raw email text, uploading email files (.eml, .txt),
or manually specifying feature parameters.
"""

import os
import sys
import re
import argparse
import joblib
import pandas as pd

MODEL_PATH = "models/phishing_detector.pkl"


def load_scanner_model():
    """Load trained pipeline model."""
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Error: Model artifact '{MODEL_PATH}' not found.")
        print("   Please run 'python main.py' first to train and save the model!")
        sys.exit(1)
    return joblib.load(MODEL_PATH)


def extract_features_from_raw_text(raw_text: str, sender_email: str = "") -> dict:
    text = raw_text or ""
    sender = (sender_email or "").strip().lower()

    email_length = len(text)
    url_pattern = r"https?://[^\s<>'\"]+|www\.[^\s<>'\"]+|<a\s+href="
    links = re.findall(url_pattern, text, re.IGNORECASE)
    number_of_links = len(links)

    attachment_keywords = r"filename=|\.pdf|\.exe|\.zip|\.docx|\.xlsx|Content-Disposition:\s*attachment"
    attachments = re.findall(attachment_keywords, text, re.IGNORECASE)
    number_of_attachments = len(attachments)

    urgent_words = [
        "urgent",
        "immediately",
        "account suspended",
        "verify your password",
        "action required",
        "security alert",
        "unauthorized access",
        "bank",
        "confirm your identity",
        "24 hours",
    ]
    has_urgent = any(w in text.lower() for w in urgent_words)
    presence_of_urgent_keywords = "Present" if has_urgent else "Absent"

    free_domains = ["gmail.com", "yahoo.com", "hotmail.com", "outlook.com", "aol.com"]
    suspicious_patterns = [r"verify", r"security", r"update", r"support-", r"service-", r"account-", r"login", r"secure"]

    sender_domain_type = "Trusted/Corporate"
    if sender:
        domain = sender.split("@")[-1] if "@" in sender else sender
        if domain in free_domains:
            sender_domain_type = "Free-Webmail/Common Provider"
        elif any(re.search(pat, domain) for pat in suspicious_patterns):
            sender_domain_type = "Suspicious/Spoofed"
    else:
        if has_urgent and number_of_links > 2:
            sender_domain_type = "Suspicious/Spoofed"

    html_pattern = r"<html|<body|<p|<div|<a\s|<br"
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


def scan_email_dict(features: dict):
    model = load_scanner_model()
    input_data = pd.DataFrame([features])

    pred = model.predict(input_data)[0]
    prob = model.predict_proba(input_data)[0]

    phishing_prob = prob[1] * 100
    legit_prob = prob[0] * 100

    print("\n" + "=" * 60)
    print(" 🛡️  CYBERSECURITY EMAIL SCANNER REPORT")
    print("=" * 60)
    print("📧 Scanned Features:")
    for k, v in features.items():
        print(f"  • {k:28s}: {v}")
    print("-" * 60)

    if pred == 1:
        print(f"🚨 SCAN RESULT: PHISHING DETECTED!")
        print(f"   Confidence Score: {phishing_prob:.1f}% Phishing (Legitimate: {legit_prob:.1f}%)")
    else:
        print(f"✅ SCAN RESULT: LEGITIMATE EMAIL")
        print(f"   Confidence Score: {legit_prob:.1f}% Legitimate (Phishing: {phishing_prob:.1f}%)")
    print("=" * 60 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Scan email file or text for Phishing detection.")
    parser.add_argument("--file", type=str, help="Path to email file (.txt, .eml)")
    parser.add_argument("--text", type=str, help="Raw email content string")
    parser.add_argument("--sender", type=str, help="Sender email address")

    args = parser.parse_args()

    if args.file:
        if not os.path.exists(args.file):
            print(f"File '{args.file}' not found.")
            sys.exit(1)
        with open(args.file, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()
        features = extract_features_from_raw_text(content, sender_email=args.sender)
        scan_email_dict(features)
    elif args.text:
        features = extract_features_from_raw_text(args.text, sender_email=args.sender)
        scan_email_dict(features)
    else:
        print("\nPaste raw email text below (press Ctrl+D or Ctrl+Z on new line when done):\n")
        try:
            content = sys.stdin.read()
            if content.strip():
                features = extract_features_from_raw_text(content)
                scan_email_dict(features)
            else:
                print("No text provided.")
        except KeyboardInterrupt:
            print("\nCancelled.")


if __name__ == "__main__":
    main()
