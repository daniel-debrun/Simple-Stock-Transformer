# Simple Stock Transformer Configuration

# Model Hyperparameters
EMBED_DIM = 128
NUM_HEADS = 8
FF_DIM = 512
NUM_LAYERS = 6
DROPOUT = 0.1
SEQUENCE_LENGTH = 60

# Training Parameters
BATCH_SIZE = 32
LEARNING_RATE = 0.001
EPOCHS = 100
PATIENCE = 5

# Data Parameters
TRAIN_SPLIT = 0.7
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15

# Stock Symbols (add more as needed)
DEFAULT_SYMBOLS = ["AAPL", "SPY", "TSLA", "MSFT", "GOOGL"]

# Feature Engineering
MOVING_AVERAGE_WINDOWS = [5, 10, 20, 50]
RSI_WINDOW = 14
BOLLINGER_WINDOW = 20
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9

# Paths
MODEL_SAVE_PATH = "models/"
FEATURE_SAVE_PATH = "feature_dataframes/"
LOG_PATH = "logs/"
