import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, TimeSeriesSplit
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
import joblib
from typing import Dict, Tuple, List

from src.feature_engineer import FeatureEngineer
from src.data_validator import DataValidator

class NHLModelTrainer:
    """Train betting-grade NHL prediction model."""
    
    def __init__(self, n_estimators: int = 100, random_state: int = 42):
        """Initialize trainer."""
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.model = None
        self.feature_engineer = FeatureEngineer()
        self.validator = DataValidator()
        self.training_report = {}
    
    def prepare_training_data(self, history: pd.DataFrame) -> Tuple[List, List]:
        """
        Prepare feature vectors and labels for training.
        
        CRITICAL: Only use data available BEFORE each game date
        (prevents look-ahead bias)
        """
        # Validate data quality
        report = self.validator.validate_dataset(history)
        print(f"\n📊 Data Quality Report:")
        print(f"   Status: {report['status']}")
        print(f"   Valid records: {report['valid_records']}/{report['total_records']}")
        print(f"   Quality score: {report['quality_score']}%")
        
        if report['status'] == 'FAILED':
            raise ValueError("Data validation failed")
        
        # Clean dataset
        history = self.validator.clean_dataset(history)
        
        # Sort by date
        history['Date'] = pd.to_datetime(history['Date'])
        history = history.sort_values('Date')
        
        # Sufficient history check
        sufficient, msg = self.validator.validate_sufficient_history(history)
        if not sufficient:
            raise ValueError(msg)
        
        print(f"✅ {msg}")
        
        X = []
        y = []
        
        print(f"\n🔧 Engineering features from {len(history)} games...")
        
        for idx, row in history.iterrows():
            try:
                home = row['Home']
                away = row['Away']
                game_date = row['Date'].strftime("%Y-%m-%d")
                
                # Create features using ONLY data before this game
                features = self.feature_engineer.create_training_features(
                    home, away, history[:idx], game_date
                )
                
                # Label: did home team win?
                label = 1 if row['Winner'] == home else 0
                
                X.append(features)
                y.append(label)
            except Exception as e:
                print(f"   ⚠️ Error processing game {idx}: {e}")
                continue
        
        print(f"✅ Created {len(X)} training examples")
        
        return X, y
    
    def train_with_cv(self, X: List, y: List, cv_folds: int = 5) -> Dict:
        """
        Train model with time-series cross-validation.
        
        Time-series CV is CRITICAL for betting:
        - Prevents look-ahead bias
        - Simulates real-world prediction (always predicting future)
        - More realistic performance estimate
        """
        print(f"\n🎓 Training RandomForest with {cv_folds}-fold Time-Series CV...")
        
        # Convert to numpy arrays
        X_array = np.array(X)
        y_array = np.array(y)
        
        # Time-series cross-validation
        tscv = TimeSeriesSplit(n_splits=cv_folds)
        
        all_scores = []
        fold_results = []
        
        for fold, (train_idx, test_idx) in enumerate(tscv.split(X_array)):
            X_train, X_test = X_array[train_idx], X_array[test_idx]
            y_train, y_test = y_array[train_idx], y_array[test_idx]
            
            # Train on this fold
            fold_model = RandomForestClassifier(
                n_estimators=self.n_estimators,
                random_state=self.random_state,
                n_jobs=-1
            )
            fold_model.fit(X_train, y_train)
            
            # Evaluate
            y_pred = fold_model.predict(X_test)
            y_pred_proba = fold_model.predict_proba(X_test)[:, 1]
            
            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred, zero_division=0)
            rec = recall_score(y_test, y_pred, zero_division=0)
            auc = roc_auc_score(y_test, y_pred_proba)
            
            fold_results.append({
                'fold': fold + 1,
                'accuracy': acc,
                'precision': prec,
                'recall': rec,
                'auc': auc
            })
            
            all_scores.append(acc)
            
            print(f"   Fold {fold + 1}: Accuracy={acc:.3f}, Precision={prec:.3f}, AUC={auc:.3f}")
        
        # Train final model on ALL data
        print(f"\n📌 Training final model on all {len(X_array)} games...")
        self.model = RandomForestClassifier(
            n_estimators=self.n_estimators,
            random_state=self.random_state,
            n_jobs=-1
        )
        self.model.fit(X_array, y_array)
        
        # Calculate overall metrics
        cv_accuracy = np.mean(all_scores)
        cv_std = np.std(all_scores)
        
        self.training_report = {
            'cv_folds': cv_folds,
            'cv_accuracy': round(cv_accuracy, 4),
            'cv_std': round(cv_std, 4),
            'fold_results': fold_results,
            'total_training_games': len(X_array)
        }
        
        print(f"\n✅ Cross-Validation Accuracy: {cv_accuracy:.3f} ± {cv_std:.3f}")
        
        return self.training_report
    
    def predict_game(self, home: str, away: str, current_history: pd.DataFrame, game_date: str) -> Tuple[str, float]:
        """
        Make prediction for a specific game.
        Returns: (predicted_winner, confidence_0_to_1)
        """
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        features = self.feature_engineer.create_training_features(
            home, away, current_history, game_date
        )
        
        prediction = self.model.predict([features])[0]
        confidence = self.model.predict_proba([features])[0]
        
        predicted_winner = home if prediction == 1 else away
        confidence_value = max(confidence)  # Confidence of predicted outcome
        
        return predicted_winner, confidence_value
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get which features matter most."""
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        feature_names = self.feature_engineer.get_feature_names()
        importances = self.model.feature_importances_
        
        return dict(zip(feature_names, importances))
    
    def save_model(self, path: str):
        """Save trained model to disk."""
        if self.model is None:
            raise ValueError("No model to save")
        
        joblib.dump(self.model, path)
        print(f"✅ Model saved to {path}")
    
    def load_model(self, path: str):
        """Load trained model from disk."""
        self.model = joblib.load(path)
        print(f"✅ Model loaded from {path}")