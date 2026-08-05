from src.main import NHLPredictorPipeline
from src.backtester_realistic import RealisticBacktester

print("🏒 Realistic Walk-Forward Backtest\n")

pipeline = NHLPredictorPipeline()
history = pipeline.load_history()

if not history.empty:
    backtester = RealisticBacktester(pipeline.trainer)
    results = backtester.walk_forward_backtest(history, train_size=500, test_size=100)
    backtester.print_results(results)