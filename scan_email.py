"""
Single Email CLI Scanner
Cybersecurity - Detecting Phishing Emails & Email Forensics
"""

import os
import sys
import re
import argparse
import pandas as pd
import joblib

MODEL_PATH = "models/phishing_detector.pkl"


def load_model():
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model file '{MODEL_PATH}' not found. Run 'python main.py' first.")
        sys.exit(1)
    return joblib.load(MODEL_PATH)


def extract_features_from_raw_text(raw_text: str, sender_email: str = "") -> dict:
    text = (raw_text or "").strip()
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
            sender_domain_type = "Trusted/Corporate"

    html_pattern = r"<html|<body|<p|<div|<a\s|<br"
    is_html = bool(re.search(html_pattern, text, re.IGNORECASE))
    html_content_flag = "Present" if is_html else "Absent"

    # --- EMAIL HEADER FORENSICS EXTRACTOR (SPF, DKIM, DMARC) ---
    spf_status = "None"
    if re.search(r"received-spf:\s*pass|spf=pass", text, re.IGNORECASE):
        spf_status = "Pass"
    elif re.search(r"received-spf:\s*softfail|spf=softfail", text, re.IGNORECASE):
        spf_status = "Softfail"
    elif re.search(r"received-spf:\s*fail|spf=fail", text, re.IGNORECASE):
        spf_status = "Fail"

    dkim_status = "None"
    if re.search(r"dkim=pass|dkim-signature:", text, re.IGNORECASE):
        dkim_status = "Pass"
    elif re.search(r"dkim=fail", text, re.IGNORECASE):
        dkim_status = "Fail"

    dmarc_status = "None"
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


def main():
    parser = argparse.ArgumentParser(description="Scan email text or file for phishing risks.")
    parser.add_argument("--file", type=str, help="Path to email text file (.txt or .eml)")
    parser.add_argument("--text", type=str, help="Raw email text content")
    parser.add_argument("--sender", type=str, default="", help="Sender email address")
    parser.add_argument("--length", type=int, help="Email character length")
    parser.add_argument("--links", type=int, help="Number of links")
    parser.add_argument("--attachments", type=int, help="Number of attachments")
    parser.add_argument("--urgent", type=str, choices=["yes", "no"], help="Urgent keywords present?")
    parser.add_argument("--domain", type=str, choices=["trusted", "freemail", "suspicious"], help="Domain type")
    parser.add_argument("--html", type=str, choices=["yes", "no"], help="HTML content present?")

    args = parser.parse_args()
    model = load_model()

    if args.file and os.path.exists(args.file):
        with open(args.file, "r", encoding="utf-8", errors="ignore") as f:
            raw = f.read()
        feats = extract_features_from_raw_text(raw, sender_email=args.sender)
    elif args.text:
        feats = extract_features_from_raw_text(args.text, sender_email=args.sender)
    elif args.length is not None:
        domain_map = {"trusted": "Trusted/Corporate", "freemail": "Free-Webmail/Common Provider", "suspicious": "Suspicious/Spoofed"}
        feats = {
            "email_text": "Manual feature override scan.",
            "email_length": args.length,
            "number_of_links": args.links or 0,
            "number_of_attachments": args.attachments or 0,
            "presence_of_urgent_keywords": "Present" if args.urgent == "yes" else "Absent",
            "sender_domain_type": domain_map.get(args.domain, "Trusted/Corporate"),
            "html_content_flag": "Present" if args.html == "yes" else "Absent",
            "spf_status": "Pass",
            "dkim_status": "Pass",
            "dmarc_status": "Pass",
        }
    else:
        print("\n=== Interactive CLI Email Scanner & Forensics ===")
        sample_text = input("Enter or paste email text: ")
        sender_in = input("Enter sender email (optional): ")
        feats = extract_features_from_raw_text(sample_text, sender_email=sender_in)

    input_df = pd.DataFrame([feats])
    pred = model.predict(input_df)[0]
    prob = model.predict_proba(input_df)[0]

    print("\n" + "="*50)
    print("📊 EXTRACTED FEATURES & FORENSICS:")
    for k, v in feats.items():
        if k != "email_text":
            print(f"  • {k:30s}: {v}")

    print("="*50)
    if pred == 1:
        print(f"🚨 VERDICT: PHISHING DETECTED (Risk Probability: {prob[1]*100:.1f}%)")
    else:
        print(f"✅ VERDICT: LEGITIMATE EMAIL (Confidence: {prob[0]*100:.1f}%)")
    print("="*50 + "\n")


if __name__ == "__main__":
    main()
