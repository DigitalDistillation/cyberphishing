# Cybersecurity – Detecting Phishing Emails

An end-to-end Machine Learning pipeline and Streamlit Web Application designed to detect phishing emails using **Natural Language Processing (NLP TF-IDF n-grams)** and **structural metadata features**.

---

## 🌟 Key Features

1. **Hybrid NLP + ML Intelligence Engine**: Extracts vocabulary context (`TfidfVectorizer`) alongside structural email indicators (`links`, `attachments`, `sender domain`, `urgent keywords`).
2. **Pre-Open Mass Email Clearance Audit**: Scan multiple emails in batch before opening them, eliminating user safety risk.
3. **Interactive Streamlit Web App (`app.py`)**: Supports single email scans, batch mass pre-scans, and manual simulation sliders.
4. **Command-Line & IMAP Inbox Scanner (`inbox_scanner.py`)**: Scriptable CLI scanner for automated inbox pre-scanning over the past X hours.

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

Train the hybrid NLP model and generate `models/phishing_detector.pkl`:

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

### 4. Mass Inbox Pre-Scanner (CLI & Script)

Audit emails received in the past X hours directly from the command line:

```bash
# Scan emails from your IMAP inbox received in the past 1 hour
python inbox_scanner.py --user yourname@gmail.com --password 'your-app-password' --hours 1
```

---

## 📁 Repository Structure

```
cyberphishing/
├── data/
│   └── data.csv                  # Project dataset (1,050 samples)
├── models/
│   └── phishing_detector.pkl     # Trained hybrid NLP pipeline artifact
├── src/
│   ├── config.yaml               # Central configuration file
│   ├── data_preparation.py       # Data cleaning & hybrid preprocessor
│   └── model_training.py        # Model training & GridSearch evaluation
├── app.py                        # Streamlit web application & batch audit UI
├── main.py                       # Pipeline execution entry point
├── inbox_scanner.py              # Mass inbox pre-scanner script
├── scan_email.py                 # Single email CLI scanner
├── eda.ipynb                     # Exploratory Data Analysis notebook
├── requirements.txt              # Project dependencies
└── README.md                     # Documentation
```
