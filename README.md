# Cybersecurity – Phishing Email Mass Scanner & Forensics Platform

An end-to-end Machine Learning pipeline and Streamlit Web Application designed to detect phishing emails using **Natural Language Processing (NLP TF-IDF n-grams)**, **structural metadata features**, and **Email Header Forensics (SPF, DKIM, DMARC)**.

---

## 🌟 Key Features

1. **Hybrid NLP + ML Intelligence Engine**: Trained on **18,650 real-world emails** using `TfidfVectorizer` (2,000 n-grams) combined with structural indicators (`links`, `attachments`, `sender domain`, `urgent keywords`) and email authentication header forensics (`SPF`, `DKIM`, `DMARC`).
2. **Pre-Open Mass Email Clearance Audit**: Scan multiple emails in batch before opening them, eliminating user safety risk.
3. **Interactive Streamlit Web App (`app.py`)**: Supports single email scans, batch mass pre-scans, and manual simulation sliders.
4. **Command-Line & IMAP Inbox Pre-Scanner (`inbox_scanner.py`)**: Auto-detects Gmail/Outlook IMAP servers and audits emails received in the past X hours directly from terminal.
5. **Interactive EDA Notebook (`eda.ipynb`)**: Multi-model benchmark evaluation (`Logistic Regression`, `Naive Bayes`, `Random Forest`, `MLP Neural Network`) with ROC curves, confusion matrices, and interactive email inference.

---

## 🚀 Quick Start & Installation

### 1. Environment Setup

```bash
git clone https://github.com/DigitalDistillation/cyberphishing.git
cd cyberphishing

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

---

### 2. Train the Machine Learning Pipeline

Train the hybrid NLP pipeline on all 18,650 real-world emails to generate `models/phishing_detector.pkl`:

```bash
python main.py
```

---

### 3. Run the Web Application

Launch the interactive Streamlit Web App:

```bash
streamlit run app.py
```

---

### 4. Live Terminal Scan Demo (0-Password Instant Execution)

Scan sample email files in 5 milliseconds with zero credentials required:

```bash
# Scan a phishing threat email sample
python scan_email.py --file sample_emails/phishing_alert.eml

# Scan a legitimate email sample
python scan_email.py --file sample_emails/legitimate_invoice.eml
```

---

### 5. Mass Inbox Pre-Scanner (CLI & IMAP)

Audit emails received in the past X hours directly from your email inbox:

```bash
# Interactive mode (prompts for email & hidden password)
python inbox_scanner.py

# Command-line mode
python inbox_scanner.py --user yourname@gmail.com --hours 1
```

---

## 📁 Repository Structure

```
cyberphishing/
├── data/
│   └── data.csv                  # Active dataset (18,650 real-world emails)
├── models/
│   └── phishing_detector.pkl     # Production hybrid NLP pipeline artifact
├── sample_emails/
│   ├── phishing_alert.eml        # Live terminal demo phishing sample
│   └── legitimate_invoice.eml    # Live terminal demo safe sample
├── src/
│   ├── config.yaml               # Central configuration file
│   ├── data_preparation.py       # Data cleaning & hybrid preprocessor
│   └── model_training.py        # Model training & GridSearch evaluation
├── app.py                        # Streamlit web application & batch audit UI
├── main.py                       # Pipeline execution entry point
├── inbox_scanner.py              # Auto-detecting mass inbox pre-scanner script
├── scan_email.py                 # Single email CLI scanner
├── eda.ipynb                     # Exploratory Data Analysis & benchmark notebook
├── requirements.txt              # Project dependencies
└── README.md                     # Documentation
```
