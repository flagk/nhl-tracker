from src.main import NHLPredictorPipeline

print("🏒 NHL Predictor - Backtest Run\n")

pipeline = NHLPredictorPipeline()
pipeline.run_training()
pipeline.run_backtesting()
