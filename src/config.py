# base line configuartion for entire project
import pandas as pd
import os

# Base directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "..", "Data","Code_run")

# File paths
RAW_DATA_PATH = os.path.join(DATA_DIR, "..","online_retail_II.xlsx")
RAW_PATH_parquet = os.path.join(DATA_DIR, "raw_data.parquet")
RAW_PATH_csv = os.path.join(DATA_DIR, "raw_data.csv")

# Normalize the path (makes sure .. and / are handled correctly)
RAW_DATA_PATH = os.path.normpath(RAW_DATA_PATH)

# Check existence before loading
if not os.path.exists(RAW_DATA_PATH):
    raise FileNotFoundError(f"Data file not found: {RAW_DATA_PATH}")

TRAIN_FILE = os.path.join(DATA_DIR, "train_baskets.parquet")
VALID_FILE = os.path.join(DATA_DIR, "validation_baskets.parquet")
TEST_FILE  = os.path.join(DATA_DIR, "test_baskets.parquet")

Processed_PATH = os.path.join(DATA_DIR, "data_processed.parquet")
Processed_Data_PATH = os.path.join(DATA_DIR, "data_with_Time.parquet")
Customer_data_PATH = os.path.join(DATA_DIR, "customer_history.pkl")
Item_data_PATH = os.path.join(DATA_DIR, "item_summary.pkl")
Baskets_data_PATH = os.path.join(DATA_DIR, "baskets.pkl")
Rules_data_PATH = os.path.join(DATA_DIR, "rules.pkl")

# Model paths
Model_dir = os.path.join(BASE_DIR, "..", "Models")

Vectorizer_data_PATH = os.path.join(DATA_DIR, "vectorizer.joblib")
Vocabulary_data_PATH = os.path.join(DATA_DIR, "vocabulary.joblib")
Best_model_path = os.path.join(Model_dir, "Final_model.joblib")

# Experiements results
Exp_dir = os.path.join(BASE_DIR, "..", "experiments")

Best_hyperparameters_path = os.path.join(Exp_dir, "Best_result.pkl")

Optimal_model_path = os.path.join(BASE_DIR, '..', 'Notebooks', "Model.py")
Best_model_path = os.path.join(Exp_dir, "retail_model.py")

final_model_path = os.path.join(Exp_dir, "final_model.joblib")
final_weights_path = os.path.join(Exp_dir, "final_weights.pkl")

# Database path
DB_path =  os.path.join(DATA_DIR, "..","Database", "Retail.db")

