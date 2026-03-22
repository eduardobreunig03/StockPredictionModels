# --- Data ---
TICKER = "AAPL"
START_DATE = "2018-01-01"
END_DATE = "2025-01-01"
INTERVAL = "1d"
WINDOW_SIZE = 90

# --- Train/test split (static fallback) ---
TRAIN_RATIO = 0.9

# --- Walk-Forward Validation ---
N_WFV_SPLITS = 5      # number of folds
WFV_TEST_RATIO = 0.1  # fraction of the full series used as the test window per fold

# --- Feature engineering ---
USE_FEATURES = True
RSI_PERIOD = 14
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9
BB_PERIOD = 20
BB_STD = 2.0

# --- Paths ---
DATA_DIR = "data"
MODEL_DIR = "models"
RESULTS_DIR = "results"

# --- LSTM ---
LSTM_UNITS = 64
LSTM_DROPOUT = 0.3
LSTM_EPOCHS = 200
LSTM_BATCH_SIZE = 32
LSTM_PATIENCE = 10

# --- CNN ---
CNN_FILTERS = 64
CNN_KERNEL_SIZE = 3
CNN_EPOCHS = 300
CNN_BATCH_SIZE = 32
CNN_PATIENCE = 10

# --- Transformer ---
TRANSFORMER_HEADS = 2
TRANSFORMER_KEY_DIM = 64
TRANSFORMER_EPOCHS = 400
TRANSFORMER_PATIENCE = 10

# --- XGBoost ---
XGB_N_ESTIMATORS = 100
XGB_OBJECTIVE = "reg:squarederror"
