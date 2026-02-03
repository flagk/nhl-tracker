## Overview
This Python script connects to the official NHL API to fetch real-time standings and stats. It uses a Random Forest Classifier to predict the winner of any hypothetical matchup based on current team performance (Points, Goal Differential, Home/Away records).

## 🚀 Features
* **Live Data:** Fetches up-to-the-minute stats from `api-web.nhle.com`.
* **Machine Learning:** Uses a Random Forest model to weigh factors like Goal Differential vs. Points.
* **Interactive:** User can input any two teams to see a win probability.

## 🛠️ How to Run
1. Install dependencies:
   ```bash
   pip install requests pandas scikit-learn