"""
Inbox Mass Scanner Module
Cybersecurity - Detecting Phishing Emails & Forensics

Connects to an IMAP email account (Gmail, Outlook, Yahoo) or scans a directory of email files,
auditing all emails received within the last X hours using the trained hybrid AI model.
"""

import os
import sys
import re
import argparse
import imaplib
import email
from email.header import decode_header
from datetime import datetime, timedelta, timezone
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


def scan_imap_inbox(server: str, email_user: str, email_pass: str, hours: int = 1):
    model = load_model()
    print(f"\n📡 Connecting to IMAP server '{server}' for '{email_user}'...")
    
    try:
        mail = imaplib.IMAP4_SSL(server)
        mail.login(email_user, email_pass)
        mail.select("inbox")

        since_date = (datetime.now(timezone.utc) - timedelta(hours=hours)).strftime("%d-%b-%Y")
        status, data = mail.search(None, f'(SINCE "{since_date}")')

        if status != "OK" or not data[0]:
            print(f"✅ No emails found in the last {hours} hour(s). Your inbox is clean!")
            return

        mail_ids = data[0].split()
        print(f"🔍 Found {len(mail_ids)} email(s) received in the last {hours} hour(s). Auditing now...\n")

        results = []
        for m_id in mail_ids:
            res, msg_data = mail.fetch(m_id, "(RFC822)")
            for response_part in msg_data:
                if isinstance(response_part, tuple):
                    msg = email.message_from_bytes(response_part[1])
                    
                    subject = decode_header(msg.get("Subject", ""))[0][0]
                    if isinstance(subject, bytes):
                        subject = subject.decode("utf-8", errors="ignore")
                    
                    sender = msg.get("From", "")
                    body = ""
                    if msg.is_multipart():
                        for part in msg.walk():
                            if part.get_content_type() == "text/plain":
                                body = part.get_payload(decode=True).decode("utf-8", errors="ignore")
                                break
                    else:
                        body = msg.get_payload(decode=True).decode("utf-8", errors="ignore")

                    full_text = f"Subject: {subject}\n{body}"
                    feats = extract_features_from_raw_text(full_text, sender_email=sender)
                    input_df = pd.DataFrame([feats])
                    
                    pred = model.predict(input_df)[0]
                    prob = model.predict_proba(input_df)[0][1] * 100

                    status_str = "🚨 PHISHING" if pred == 1 else "✅ SAFE"
                    results.append({
                        "Subject": subject[:35],
                        "Sender": sender[:25],
                        "Status": status_str,
                        "SPF": feats["spf_status"],
                        "DKIM": feats["dkim_status"],
                        "Risk Score": f"{prob:.1f}%"
                    })

        mail.logout()
        res_df = pd.DataFrame(results)
        print("=== INBOX MASS SAFETY & FORENSICS AUDIT REPORT ===")
        print(res_df.to_string(index=False))

    except Exception as e:
        print(f"⚠️ Connection error: {e}")


def main():
    parser = argparse.ArgumentParser(description="Mass Inbox Email Pre-Scanner (Past X Hours)")
    parser.add_argument("--server", type=str, default="imap.gmail.com", help="IMAP server (e.g. imap.gmail.com, outlook.office365.com)")
    parser.add_argument("--user", type=str, help="Email address")
    parser.add_argument("--password", type=str, help="App password")
    parser.add_argument("--hours", type=int, default=1, help="Hours back to scan (default: 1)")

    args = parser.parse_args()

    if args.user and args.password:
        scan_imap_inbox(args.server, args.user, args.password, hours=args.hours)
    else:
        print("\n📥 Batch Inbox Mass Pre-Scanner (CLI Mode)")
        print("Usage Example:")
        print("  python inbox_scanner.py --user yourname@gmail.com --password 'your-app-password' --hours 1\n")


if __name__ == "__main__":
    main()
