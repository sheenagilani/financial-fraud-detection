# 🛡️ FinShield: AI-Powered Financial Fraud Detection Engine

An end-to-end Machine Learning pipeline and surveillance dashboard designed to detect fraudulent financial transactions in real time. Built to address extreme class imbalance using a calibrated Balanced Random Forest classifier and domain-specific balance-error engineering.

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://finshield-fraud-detection.streamlit.app)
[![GitHub License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)

---

## 🔗 Live Application
- **Live Interactive Dashboard:** [finshield-fraud-detection.streamlit.app](https://finshield-fraud-detection.streamlit.app)
- **Source Repository:** [github.com/sheenagilani/financial-fraud-detection](https://github.com/sheenagilani/financial-fraud-detection)

---

## 📌 Problem Statement & Core Challenge
Financial fraud detection datasets (such as the PaySim mobile money benchmark) suffer from **severe class imbalance**, where fraudulent transactions typically comprise less than 0.2% of the overall volume. Relying on standard accuracy leads to naive majority-class classification. FinShield optimizes specifically for high recall and precision via ensemble learning, custom threshold calibration, and engineered balance delta features.

---

## ⚙️ Key Technical Features

- **Balance Discrepancy Engineering:** Computes origin and destination balance errors ($Orig_{diff}$ and $Dest_{diff}$) to identify unauthorized account-draining anomalies.
- **Imbalance Handling:** Utilizes a Balanced Random Forest ensemble with downsampling on majority classes across decision trees.
- **Calibrated Risk Engine:** Employs an adjustable decision threshold to mitigate false negatives in high-risk transaction types (`TRANSFER`, `CASH_OUT`).
- **Real-Time Surveillance UI:** Single-screen diagnostic dashboard built with Streamlit for single-transaction scoring and multi-feature evaluation.

---

## 📂 Repository Structure

```text
├── app.py                      # Interactive Streamlit surveillance dashboard
├── fraud_detection_model.pkl   # Serialized Scikit-Learn / Balanced Random Forest model
├── requirements.txt            # Python environment dependencies
└── README.md                   # Project documentation & architecture overview
