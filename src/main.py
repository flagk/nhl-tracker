import yaml
import pandas as pd
import os
from pathlib import Path
from dotenv import load_dotenv
from datetime import datetime

from src.fetcher import NHLDataFetcher
from src.data_validator import DataValidator
from src.trainer import NHLModelTrainer
from src.backtester import NHLBacktester
from src.predictor import NHLPredictor
from src.tracker import PredictionTracker

class NHLPredictorPipeline:
    """Main orchestrator for NHL prediction pipeline."""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize pipeline with config."""
        load_dotenv()
        
        self.config = self._load_config(config_path)
        self.fetcher = NHLDataFetcher()
        self.validator = DataValidator()
        self.trainer = NHLModelTrainer(
            n_estimators=self.config['model']['n_estimators'],
            random_state=self.config['model']['random_state']
        )
        self.backtester = NHLBacktester(
            self.trainer,
            min_confidence=0.55
        )
        self.predictor = NHLPredictor(
            self.trainer,
            min_confidence=0.55
        )
        self.tracker = PredictionTracker()
        
        # Ensure data directories exist
        Path(self.config['data']['history_file']).parent.mkdir(parents=True, exist_ok=True)
        Path(self.config['data']['predictions_log']).parent.mkdir(parents=True, exist_ok=True)
        Path(self.config['data']['model_path']).parent.mkdir(parents=True, exist_ok=True)
    
    def _load_config(self, config_path: str) -> dict:
        """Load YAML config."""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def load_history(self) -> pd.DataFrame:
        """Load historical game data."""
        history_path = self.config['data']['history_file']
        
        if not os.path.exists(history_path):
            print(f"❌ History file not found: {history_path}")
            print("   Generate one by exporting from nhl_history.csv")
            return pd.DataFrame()
        
        history = pd.read_csv(history_path)
        print(f"✅ Loaded {len(history)} historical games")
        return history
    
    def run_training(self):
        """Train model on historical data."""
        print("\n" + "="*60)
        print("PHASE 1: TRAINING")
        print("="*60)
        
        # Load history
        history = self.load_history()
        if history.empty:
            print("❌ No history to train on")
            return False
        
        # Prepare training data
        X, y = self.trainer.prepare_training_data(history)
        
        if len(X) == 0:
            print("❌ No valid training data")
            return False
        
        # Train with cross-validation
        cv_report = self.trainer.train_with_cv(X, y, cv_folds=5)
        
        # Save model
        self.trainer.save_model(self.config['data']['model_path'])
        
        # Show feature importance
        self.predictor.print_feature_importance()
        
        return True
    
    def run_backtesting(self):
        """Backtest model on historical games."""
        print("\n" + "="*60)
        print("PHASE 2: BACKTESTING")
        print("="*60)
        
        if self.trainer.model is None:
            print("❌ Model not trained. Run training first.")
            return
        
        history = self.load_history()
        if history.empty:
            return
        
        # Run backtest
        backtest_results = self.backtester.backtest(
            history,
            bet_amount=100
        )
        
        # Print results
        self.backtester.print_backtest_report(backtest_results)
        
        # Kelly criterion recommendation
        if backtest_results['status'] == 'SUCCESS':
            win_rate = backtest_results['win_rate']
            kelly_rec = self.backtester.recommended_bet_size(
                bankroll=1000,
                win_rate=win_rate
            )
            print(f"💡 Bet Sizing Recommendation (assuming $1000 bankroll):")
            print(f"   {kelly_rec['recommendation']}")
    
    def run_predictions(self):
        """Make predictions for today's games."""
        print("\n" + "="*60)
        print("PHASE 3: TODAY'S PREDICTIONS")
        print("="*60)
        
        if self.trainer.model is None:
            print("❌ Model not trained. Run training first.")
            return
        
        # Load current history
        history = self.load_history()
        if history.empty:
            return
        
        # Get today's schedule
        today = datetime.now().strftime("%Y-%m-%d")
        games = self.fetcher.get_schedule(today)
        
        if not games:
            print(f"📅 No games scheduled for {today}")
            return
        
        # Prepare game list
        game_list = [
            {
                'home': game['homeTeam']['abbrev'],
                'away': game['awayTeam']['abbrev'],
                'date': today
            }
            for game in games
        ]
        
        # Make predictions
        predictions = self.predictor.predict_slate(game_list, history)
        
        # Print predictions
        self.predictor.print_slate(predictions)
        
        # Log predictions
        log_path = self.config['data']['predictions_log']
        self.tracker.log_predictions(predictions, log_path)
        print(f"✅ Predictions logged to {log_path}")
    
    def run_full_pipeline(self):
        """Run complete pipeline: train → backtest → predict."""
        print("\n" + "🏒 "*20)
        print("NHL PREDICTION PIPELINE - FULL RUN")
        print("🏒 "*20)
        
        # Phase 1: Train
        if not self.run_training():
            print("❌ Training failed")
            return
        
        # Phase 2: Backtest
        self.run_backtesting()
        
        # Phase 3: Predict
        self.run_predictions()
        
        print("\n" + "="*60)
        print("✅ PIPELINE COMPLETE")
        print("="*60 + "\n")


def main():
    """Entry point."""
    pipeline = NHLPredictorPipeline()
    pipeline.run_full_pipeline()


if __name__ == "__main__":
    main()