import pandas as pd
from typing import Dict, List, Tuple
from datetime import datetime

from src.trainer import NHLModelTrainer

class NHLPredictor:
    """Make live predictions for upcoming NHL games."""
    
    def __init__(self, trainer: NHLModelTrainer, min_confidence: float = 0.55):
        """
        Initialize predictor.
        
        min_confidence: Only recommend bets above this confidence
        """
        self.trainer = trainer
        self.min_confidence = min_confidence
    
    def predict_game(self, home: str, away: str, current_history: pd.DataFrame, game_date: str) -> Dict:
        """
        Predict outcome of a single game.
        
        Returns detailed prediction with confidence and recommendation.
        """
        try:
            predicted_winner, confidence = self.trainer.predict_game(
                home, away, current_history, game_date
            )
            
            # Determine bet recommendation
            should_bet = confidence >= self.min_confidence
            
            # Calculate expected value (simplified, assuming -110 odds = 1.91 decimal)
            odds_decimal = 1.91
            implied_probability = 1 / odds_decimal
            edge = (confidence - implied_probability) * 100
            
            result = {
                'home': home,
                'away': away,
                'game_date': game_date,
                'predicted_winner': predicted_winner,
                'confidence': round(confidence, 4),
                'confidence_pct': round(confidence * 100, 1),
                'should_bet': should_bet,
                'confidence_threshold': self.min_confidence,
                'edge_pct': round(edge, 2),
                'recommendation': self._get_recommendation(predicted_winner, confidence, edge)
            }
            
            return result
        
        except Exception as e:
            return {
                'error': str(e),
                'home': home,
                'away': away
            }
    
    def predict_slate(self, games: List[Dict], current_history: pd.DataFrame) -> List[Dict]:
        """
        Predict multiple games for a given day.
        
        games: List of dicts with 'home', 'away', 'date'
        """
        predictions = []
        
        for game in games:
            pred = self.predict_game(
                game['home'],
                game['away'],
                current_history,
                game.get('date', datetime.now().strftime("%Y-%m-%d"))
            )
            predictions.append(pred)
        
        return predictions
    
    def print_prediction(self, prediction: Dict):
        """Print formatted prediction for a single game."""
        if 'error' in prediction:
            print(f"❌ Error predicting {prediction['home']} vs {prediction['away']}: {prediction['error']}")
            return
        
        matchup = f"{prediction['away']} @ {prediction['home']}"
        winner = prediction['predicted_winner']
        conf = prediction['confidence_pct']
        edge = prediction['edge_pct']
        
        # Color coding for recommendation
        if prediction['should_bet']:
            status = "✅ BET"
        else:
            status = "⏸️  SKIP (low confidence)"
        
        print(f"\n{matchup}")
        print(f"   Pick: {winner} ({conf}% confidence)")
        print(f"   Edge: {edge:.2f}%")
        print(f"   Action: {status}")
    
    def print_slate(self, predictions: List[Dict]):
        """Print all predictions for a slate of games."""
        print(f"\n{'='*60}")
        print(f"GAME PREDICTIONS ({datetime.now().strftime('%Y-%m-%d')})")
        print(f"{'='*60}")
        
        bet_recommendations = [p for p in predictions if p.get('should_bet', False)]
        skip_recommendations = [p for p in predictions if not p.get('should_bet', True)]
        
        print(f"\n🎯 RECOMMENDED BETS ({len(bet_recommendations)}):")
        for pred in bet_recommendations:
            self.print_prediction(pred)
        
        if skip_recommendations:
            print(f"\n⏸️  SKIPPING ({len(skip_recommendations)}):")
            for pred in skip_recommendations:
                matchup = f"{pred['away']} @ {pred['home']}"
                conf = pred['confidence_pct']
                print(f"   {matchup} ({conf}% - below {self.min_confidence*100:.0f}% threshold)")
        
        print(f"\n{'='*60}\n")
    
    def _get_recommendation(self, winner: str, confidence: float, edge: float) -> str:
        """Generate text recommendation based on confidence and edge."""
        if confidence < self.min_confidence:
            return "SKIP - Low confidence"
        
        if confidence >= 0.65 and edge >= 5:
            return "STRONG BET - High confidence + positive edge"
        elif confidence >= 0.60 and edge >= 0:
            return "BET - Moderate confidence + edge"
        elif confidence >= self.min_confidence and edge >= 0:
            return "WEAK BET - Threshold met but marginal edge"
        else:
            return "PASS - Negative edge"
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get which features influence predictions most."""
        return self.trainer.get_feature_importance()
    
    def print_feature_importance(self):
        """Print feature importance in readable format."""
        importance = self.get_feature_importance()
        
        print(f"\n📊 Feature Importance:")
        sorted_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)
        
        for feature, importance_val in sorted_features:
            bar = "█" * int(importance_val * 50)
            print(f"   {feature:20} {bar} {importance_val:.4f}")