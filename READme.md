# 🏒 NHL Game Predictor & Tracker

A machine learning application that predicts daily NHL game outcomes using a Random Forest Classifier. This project features an automated data pipeline that fetches live stats from the official NHL API, generates predictions, and tracks historical accuracy in a persistent CSV log.

## 🚀 Project Overview

* **Goal:** Build a sustainable system to predict sports outcomes and validate model performance over time.
* **Tech Stack:** Python, scikit-learn, pandas, NumPy, Requests (API), Excel (Dashboard).
* **Model:** Random Forest Classifier (Ensemble Learning).

## 🧠 How It Works (The Machine Learning Logic)

This tool treats hockey prediction as a **binary classification problem** (Win vs. Loss).

1.  **Data Ingestion:**
    * Connects to the `api-web.nhle.com` endpoint to fetch real-time team statistics (Points, Goal Differential, L10 Record, etc.).
2.  **Feature Engineering:**
    * The model evaluates matchups based on *relative* differences rather than raw stats.
    * **Key Features:** `Points Diff`, `Goal Differential Diff`, `Win % Diff`, `Last 10 Games Momentum`.
3.  **Inference:**
    * A Random Forest model calculates a "Win Probability" (Confidence Score).
    * Predictions >55% confidence are flagged as "High Confidence."
4.  **Validation Loop:**
    * The system remembers past predictions. Every time the script runs, it checks yesterday's results to auto-grade the model as `Correct` or `Incorrect`, creating a feedback loop for accuracy tracking.

## 🛠️ Installation & Setup

### Prerequisites
* Python 3.10+
* pip (Python Package Manager)

### 1. Clone the Repository
```bash
git clone [https://github.com/YOUR_USERNAME/nhl-predictor.git](https://github.com/YOUR_USERNAME/nhl-predictor.git)
cd nhl-predictor