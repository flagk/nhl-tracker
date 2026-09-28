# NHL Prediction Model

> **⚠️ Correction (Phase 1 audit):** the "55.8% accuracy / 11.69% ROI / proven edge" figures below
> are **not valid evidence of betting edge**. They assume even-money payouts with no vig and no real
> odds, the model does no better than always picking the home team, and its probabilities are
> overconfident. See [AUDIT.md](AUDIT.md) for the corrected analysis. This project is being rebuilt.
> Research and educational use only; no model guarantees profit.

A production-grade machine learning system for predicting NHL game outcomes with walk-forward backtesting and continuous learning.

## Overview

This model predicts NHL game winners using 11 engineered features including momentum, volatility, rest days, and home/away splits. Walk-forward backtesting on 2,987 historical games demonstrates a **55.8% win rate with 11.69% ROI** — proving a measurable edge over random guessing.

### Key Results

| Metric | Value |
|--------|-------|
| Test Accuracy | 55.8% |
| Games Backtested | 1,805 |
| Profit (100 bets) | $21,100 |
| ROI | 11.69% |
| Edge vs Random | +5.8% |

## Architecture
src/
├── fetcher.py # NHL data retrieval
├── feature_engineer.py # 11 predictive features
├── trainer.py # RandomForest with time-series CV
├── backtester_realistic.py # Walk-forward validation
├── predictor.py # Live game predictions
├── tracker.py # Performance logging
├── generate_chart.py # Performance visualization
└── main.py # Pipeline orchestrator
## Features (11 Total)

1. **PPG Difference** — Points per game differential
2. **Goal Differential** — Goals for/against differential
3. **Recent Form** — Last 5 games momentum
4. **Win Streak** — Current streak
5. **Home/Away Split** — Home court advantage
6. **Rest Advantage** — Days of rest
7. **Win Percentage** — Historical win % gap
8. **Goals For/Against** — Offensive/defensive strength
9-11. **Recent metrics** — Trend analysis

## Methodology

### Walk-Forward Backtesting (No Look-Ahead Bias)

Traditional backtesting is unrealistic:
- Train on ALL history
- Test on SAME history
- Model "cheats" by seeing the future

**Walk-forward validation is correct:**
1. Train on games 1-500
2. Test on games 501-600 (unseen)
3. Train on games 1-600
4. Test on games 601-700
5. Repeat...

This simulates real prediction: train on past, test on future.

### Time-Series Cross-Validation

- Respects chronological order
- Prevents temporal leakage
- Realistic performance estimates

### RandomForest Classifier

- 100 trees, no depth limit
- Handles non-linear patterns
- Built-in feature importance
- Fast training and inference

## Results

| Approach | Accuracy | ROI |
|----------|----------|-----|
| Random guessing | 50.0% | 0% |
| **This model** | **55.8%** | **11.69%** |

**Proven edge:** 55.8% win rate beats both random chance and professional baselines.

## Setup

### Requirements

```bash
pip install -r requirements.txt
```

### Historical Data

Requires `data/history/nhl_history.csv`:
- `Date` (YYYY-MM-DD)
- `Home`, `Away` (team abbreviations)
- `HomeScore`, `AwayScore` (goals)
- `Winner` (team that won)

## Usage

### Train & Backtest

```bash
python3.11 test_realistic.py
```

Output shows win rate and ROI on held-out test data.

### Live Predictions

```python
from src.main import NHLPredictorPipeline

pipeline = NHLPredictorPipeline()
history = pipeline.load_history()

pred = pipeline.predictor.predict_game(
    home='NYR', away='BOS', 
    current_history=history, 
    game_date='2026-02-03'
)

print(f"Pick: {pred['predicted_winner']}")
print(f"Confidence: {pred['confidence_pct']}%")
```

## Automation

GitHub Actions runs daily:
1. Retrain model on latest data
2. Make predictions for today's games
3. Generate performance chart
4. Update README stats
5. Auto-commit results

## Performance Tracking

![Performance Chart](performance_chart.png)

Chart updates daily showing:
- **Rolling win rate** (should stay >55%)
- **Cumulative profit** (should trend up)

## Key Design Decisions

**1. Walk-Forward Over Standard Backtesting**
- Prevents look-ahead bias
- Proves real edge exists
- Realistic performance metrics

**2. 11 Features Over Many**
- Interpretable (explainable to stakeholders)
- Non-redundant (low correlation)
- Proven predictive value

**3. RandomForest Over Deep Learning**
- Interpretable feature importance
- Fast training
- Works with smaller datasets
- No hyperparameter tuning needed

**4. Time-Series CV Over Random Split**
- Respects temporal order
- Prevents future data leakage
- Realistic validation

## Limitations

- Daily data only (no intraday)
- No injury/roster integration
- No strength-of-schedule weighting
- No playoff adjustments

## Future Work

- [ ] Injury report integration
- [ ] Strength of schedule weighting
- [ ] Player-level statistics
- [ ] Ensemble methods
- [ ] Real-time updates

## Tech Stack

- **Data**: Pandas, SQLite
- **ML**: scikit-learn (RandomForest)
- **Automation**: GitHub Actions
- **Visualization**: Matplotlib
- **Python**: 3.11+

## Repository Stats

- 2,987 historical games
- 1,805 predictions tested
- 11 engineered features
- 5-fold time-series CV
- **55.8% accuracy on unseen data**
- **11.69% ROI proven**

## Author

Built as a portfolio project demonstrating:
- End-to-end ML pipeline design
- Production Python code
- Walk-forward backtesting
- Automated CI/CD
- Real performance measurement

## License

MIT
