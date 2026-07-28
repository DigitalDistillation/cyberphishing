# AIAP Cybersecurity – Detecting Phishing Emails

![Python Version](https://img.shields.io/badge/python-3.9%2B-blue)
![Framework](https://img.shields.io/badge/scikit--learn-1.3%2B-orange)
![License](https://img.shields.io/badge/license-MIT-green)

A modular, production-ready machine learning project designed to detect phishing emails using tabular structural and content cues. Built strictly according to the **AI Singapore AIAP Foundation Learning Companion** guidelines and software engineering standards.

---

## 📌 Problem Overview

Phishing attacks remain a primary vector for credential theft, data loss, and unauthorized system access. This project formulates phishing detection as a **Binary Classification** task predicting whether an incoming email is legitimate or phishing based on six key features:

- `email_length`: Length of email body in characters (Numerical: 10–10,000)
- `number_of_links`: Number of hyperlinked URLs (Numerical/Integer: 0–20)
- `number_of_attachments`: Count of file attachments (Numerical/Integer: 0–10)
- `presence_of_urgent_keywords`: Categorical/Binary (`Absent` / `Present`)
- `sender_domain_type`: Categorical (`Suspicious/Spoofed`, `Trusted/Corporate`, `Free-Webmail/Common Provider`)
- `html_content_flag`: Categorical/Binary (`Present` / `Absent`)
- **Target Variable (`phishing_email`)**: Binary target (`Legitimate` = `0`, `Phishing` = `1`)

---

## 📁 Repository Structure

```
cyberphishing/
|--- eda.ipynb                # Comprehensive Exploratory Data Analysis notebook
|--- main.py                  # Pipeline execution entry point script (trains & saves model)
|--- scan_email.py            # CLI script to scan emails and display risk reports
|--- app.py                   # Streamlit Web UI Scanner app
|--- README.md                # Project documentation and architectural overview
|--- requirements.txt         # Python package dependencies
|--- data/
|    |--- data.csv            # Synthetic phishing email dataset (1,000 samples)
|--- models/
|    |--- phishing_detector.pkl # Trained pipeline artifact
|--- src/
|    |--- data_preparation.py # DataPreparation OOP class & ColumnTransformer preprocessor
|    |--- model_training.py   # ModelTraining OOP class & GridSearchCV tuning pipeline
|    |--- config.yaml         # Centralized configuration parameters
```

---

## 🚀 Setup & Execution Guide

### 1. Environment Setup

Activate the Python virtual environment:

```bash
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Train Models & Save Pipeline

Run `main.py` to clean data, perform stratified splitting, train baseline & tuned models, and export `models/phishing_detector.pkl`:

```bash
python main.py
```

### 3. Scan Emails (CLI Scanner)

You can scan individual emails directly in your terminal using flag parameters or interactive prompts:

#### Command Line Flags Example:
```bash
python scan_email.py --length 450 --links 6 --attachments 1 --urgent yes --domain suspicious --html yes
```

#### Interactive Mode:
```bash
python scan_email.py
```

### 4. Interactive Web Scanner App (Browser UI)

Launch the Streamlit Web UI:

```bash
streamlit run app.py
```

---

## 📊 Model Evaluation Results

### Selected Best Model Test Set Results (Holdout 150 samples)

- **Selected Model**: Logistic Regression with `class_weight='balanced'`
- **Test Accuracy**: `73.33%`
- **Test Precision**: `67.86%`
- **Test Recall**: `63.33%`
- **Test F1-Score**: `0.6552`
- **Test ROC-AUC**: `0.8030`
- **Confusion Matrix**: `[[72, 18], [22, 38]]`
