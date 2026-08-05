import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from datetime import datetime

from src.trainer import NHLModelTrainer
from src.feature_engineer import FeatureEngineer

class NHLBacktester:
    """Simulate betting performance on historical games."""
    
    def __init__(self, trainer: NHLModelTrainer, min_confidence: float = 0.55):
        """
        Initialize backtester.
        
        min_confidence: Only bet if model is this confident (0.5 = coin flip, 0.6 = strong)
        """
        self.trainer = trainer
        self.min_confidence = min_confidence
        self.feature_engineer = FeatureEngineer()
    
    def backtest(self, history: pd.DataFrame, bet_amount: float = 100) -> Dict:
        """
        Backtest model on historical games.
        
        Args:
            history: Historical game data
            bet_amount: Amount to bet per game (for ROI calculation)
        
        Returns:
            Backtest results with ROI, win rate, edge, etc.
        """
        # Sort by date
        history['Date'] = pd.to_datetime(history['Date'])
        history = history.sort_values('Date').reset_index(drop=True)
        
        print(f"\n🎲 Backtesting on {len(history)} historical games...")
        print(f"   Minimum confidence threshold: {self.min_confidence*100:.1f}%")
        print(f"   Bet amount per game: ${bet_amount}")
        
        predictions = []
        bets_placed = 0
        correct_bets = 0
        total_wagered = 0
        total_won = 0
        
        for idx, row in history.iterrows():
            # Only use data BEFORE this game (no look-ahead bias)
            historical_subset = history[:idx]
            
            if len(historical_subset) < 20:  # Need minimum history
                continue
            
            home = row['Home']
            away = row['Away']
            game_date = row['Date'].strftime("%Y-%m-%d")
            actual_winner = row['Winner']
            
            try:
                # Make prediction
                predicted_winner, confidence = self.trainer.predict_game(
                    home, away, historical_subset, game_date
                )
                
                # Only bet if confident enough
                if confidence >= self.min_confidence:
                    bets_placed += 1
                    total_wagered += bet_amount
                    
                    # Did we win?
                    is_correct = (predicted_winner == actual_winner)
                    if is_correct:
                        correct_bets += 1
                        total_won += bet_amount * 2  # Win doubles your money
                    
                    predictions.append({
                        'date': game_date,
                        'home': home,
                        'away': away,
                        'prediction': predicted_winner,
                        'confidence': confidence,
                        'actual_winner': actual_winner,
                        'result': 'WIN' if is_correct else 'LOSS'
                    })
            
            except Exception as e:
                continue
        
        # Calculate metrics
        if bets_placed == 0:
            print("❌ No bets placed (confidence threshold too high)")
            return {
                'status': 'FAILED',
                'bets_placed': 0,
                'message': 'No bets placed'
            }
        
        win_rate = correct_bets / bets_placed
        profit = total_won - total_wagered
        roi = (profit / total_wagered) * 100 if total_wagered > 0 else 0
        
        # Calculate edge (expected value per bet)
        edge = (win_rate - 0.5) * bet_amount  # Simplified: assumes -110 odds
        
        results = {
            'status': 'SUCCESS',
            'bets_placed': bets_placed,
            'correct': correct_bets,
            'incorrect': bets_placed - correct_bets,
            'win_rate': round(win_rate, 4),
            'total_wagered': total_wagered,
            'total_returned': total_won,
            'profit': profit,
            'roi_pct': round(roi, 2),
            'edge_per_bet': round(edge, 2),
            'predictions': predictions
        }
        
        return results
    
    def print_backtest_report(self, results: Dict):
        """Print formatted backtest results."""
        if results['status'] == 'FAILED':
            print(f"❌ Backtest failed: {results['message']}")
            return
        
        print(f"\n{'='*60}")
        print(f"BACKTEST RESULTS")
        print(f"{'='*60}")
        print(f"\n📊 Overall Performance:")
        print(f"   Bets placed: {results['bets_placed']}")
        print(f"   Wins: {results['correct']} | Losses: {results['incorrect']}")
        print(f"   Win rate: {results['win_rate']*100:.1f}%")
        print(f"\n💰 Financial Results:")
        print(f"   Total wagered: ${results['total_wagered']:,.0f}")
        print(f"   Total returned: ${results['total_returned']:,.0f}")
        print(f"   Profit: ${results['profit']:,.0f}")
        print(f"   ROI: {results['roi_pct']:.2f}%")
        print(f"\n⚡ Edge Analysis:")
        print(f"   Edge per bet: ${results['edge_per_bet']:.2f}")
        
        # Interpretation
        if results['win_rate'] > 0.55:
            print(f"   ✅ Positive edge detected (>55% win rate)")
        elif results['win_rate'] > 0.50:
            print(f"   ⚠️ Slight edge (50-55% win rate)")
        else:
            print(f"   ❌ No edge (≤50% win rate)")
        
        print(f"\n{'='*60}\n")
    
    def kelly_criterion(self, win_rate: float, odds: float = 1.91) -> float:
        """
        Calculate optimal bet size using Kelly Criterion.
        
        Kelly % = (bp - q) / b
        where: b = decimal odds - 1, p = win probability, q = 1 - p
        
        This maximizes long-term bankroll growth.
        """
        if win_rate <= 0 or win_rate >= 1:
            return 0
        
        b = odds - 1  # Typical -110 odds = 1.91 decimal
        p = win_rate
        q = 1 - p
        
        kelly = (b * p - q) / b
        kelly = max(0, min(kelly, 0.25))  # Cap at 25% to be conservative
        
        return kelly
    
    def recommended_bet_size(self, bankroll: float, win_rate: float) -> Dict:
        """
        Recommend bet sizing based on Kelly Criterion.
        """
        kelly_pct = self.kelly_criterion(win_rate)
        bet_size = bankroll * kelly_pct
        
        return {
            'bankroll': bankroll,
            'win_rate': win_rate,
            'kelly_pct': round(kelly_pct * 100, 2),
            'recommended_bet': round(bet_size, 2),
            'recommendation': f"Bet ${bet_size:.2f} per game (Kelly: {kelly_pct*100:.1f}%)"
        }