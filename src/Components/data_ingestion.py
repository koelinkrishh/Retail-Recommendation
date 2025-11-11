# File to convert excel file into dataset
import os
import sys
import subprocess
import shutil
import zipfile
from src.logger import logging
from src.exception import CustomException
import pandas as pd
import sqlite3
import joblib 

# File to note all paths
from src.config import *

def ensure_dataset_available():
    """
    Ensure 'online_retail_II.xlsx' exists locally; if not, attempt to download
    it from the Kaggle dataset 'koelin/retail-purchase-history'.
    Also ensures the directory structure: Data/, Data/Code_run/, Data/Database/.
    """

    excel_path = RAW_DATA_PATH  # Expected: "Data/online_retail_II.xlsx"
    dataset_dir = os.path.dirname(excel_path)
    kaggle_dataset = "koelin/retail-purchase-history"
    expected_file = "online_retail_II.xlsx"

    # ✅ Step 1: Ensure folder structure
    try:
        os.makedirs(dataset_dir, exist_ok=True)
        os.makedirs(os.path.join(DATA_DIR, "Code_run"), exist_ok=True)
        os.makedirs(os.path.join(DATA_DIR, "Database"), exist_ok=True)
        logging.info("✅ Verified folder structure: Data/, Data/Code_run/, Data/Database/")
    except Exception as e:
        raise CustomException(f"Failed to create data folders: {e}", sys)

    # ✅ Step 2: Check if file already exists
    if os.path.exists(excel_path):
        logging.info(f"✅ Dataset already available locally at: {excel_path}")
        return True

    logging.warning(f"⚠️ Dataset not found locally at: {excel_path}")

    # ✅ Step 3: Attempt to use Kaggle CLI
    kaggle_exists = shutil.which("kaggle") is not None
    if not kaggle_exists:
        logging.warning("⛔ Kaggle CLI not found — skipping dataset download (Render/cloud-safe mode).")
        logging.warning(f"Please manually place '{expected_file}' in {dataset_dir}")
        return False

    # ✅ Step 4: Download from Kaggle
    try:
        logging.info("📦 Downloading dataset from Kaggle...")
        subprocess.run(
            ["kaggle", "datasets", "download", "-d", kaggle_dataset, "-p", dataset_dir],
            check=True
        )

        # ✅ Step 5: Extract the correct file
        for file in os.listdir(dataset_dir):
            if file.endswith(".zip"):
                zip_path = os.path.join(dataset_dir, file)
                with zipfile.ZipFile(zip_path, "r") as zip_ref:
                    for member in zip_ref.namelist():
                        if expected_file in member:
                            zip_ref.extract(member, dataset_dir)
                            extracted_path = os.path.join(dataset_dir, expected_file)
                            # Handle nested folders (Kaggle zips sometimes include subfolders)
                            if not os.path.exists(excel_path) and os.path.exists(os.path.join(dataset_dir, member)):
                                os.rename(os.path.join(dataset_dir, member), excel_path)
                            logging.info(f"✅ Extracted {expected_file} → {excel_path}")
                os.remove(zip_path)
                break

        if not os.path.exists(excel_path):
            raise FileNotFoundError(f"❌ Expected file '{expected_file}' not found after extraction.")

        logging.info("✅ Dataset successfully downloaded and ready for use.")
        return True

    except subprocess.CalledProcessError as e:
        logging.error("❌ Kaggle download failed — check if Kaggle API credentials are set.")
        raise CustomException(e, sys)
    except Exception as e:
        logging.error("❌ Unexpected error while downloading dataset.")
        raise CustomException(e, sys)

def Read_into_Excel_sheets():
    """ Function to read excel sheet and store it as dataframe """
    logging.info("Reading excel file - sheets into dataframes")
    try:
        # Ensure dataset exists
        ensure_dataset_available()
        
        excel_path = RAW_DATA_PATH
        df1 = pd.read_excel(excel_path, sheet_name="Year 2009-2010")
        df2 = pd.read_excel(excel_path, sheet_name="Year 2010-2011")
        
        # combining dataframe
        df = pd.concat([df1,df2], ignore_index=True)
        logging.info("Sucessfully read excel files & combined them together.")
        
        return df  
            
    except Exception as e:
        logging.error("Failed to read excel file")
        raise CustomException(e, sys)

    

def load_data(data_path=None,form="excel"):
    """ Load dataset from the given file. """
    try:
        if form=="parquet":
            logging.info("Reading dataset from parquet file")
            pro = Processed_Data_PATH if data_path is None else data_path
            df = pd.read_parquet(pro)
            return df
        elif form=="excel": # base for start, all other methods are for after preprocessing
            logging.info("Reading dataset from excel file")
            df = Read_into_Excel_sheets()
            return df
        elif form=="csv":
            logging.info("Reading dataset from csv file")
            df = pd.read_csv(data_path)
            return df
        elif form == "json":
            logging.info("Reading dataset from json file")
            df = pd.read_json(data_path)
            return df
        elif form=="pickle":
            logging.info("Reading dataset from pickle file")
            df = pd.read_pickle(data_path)
            return df
        elif form=="database":
            logging.info("Reading dataset from project database")
            DB_path = os.path.join(DATA_DIR, "Database", "retail.db")
            
            conn = sqlite3.connect(DB_path)
            df = pd.read_sql_query("SELECT * FROM transactions", conn)
            conn.close()
            return df
        elif form=="joblib":
            logging.info("Reading dataset from joblib file")
            file = joblib.load(data_path)
            return file
        else:
            logging.debug("No valid format given.")
            raise CustomException("Give a valid format", sys)
            
    except Exception as e:
        logging.debug("Problem in loading dataset from parquet file.")
        raise CustomException(e, sys)

if __name__ == "__main__":
    data = load_data(form="excel")
    print(data.sample(3))
    print(data.shape)
    
    # Rename columns with gap
    data.columns = [col.replace(" ", "_") for col in data.columns]
    
    # Ensure all columns are string-safe before saving
    for col in data.columns:
        if data[col].dtype == 'object' or data[col].apply(lambda x: isinstance(x, str)).any():
            data[col] = data[col].astype(str)
    
    # === Save to CSV ===
    csv_path = os.path.join(DATA_DIR, "raw_data.csv")
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    data.to_csv(csv_path, index=False, encoding='utf-8')
    logging.info(f"Dataset saved as csv")
    print(f"✅ Saved csv successfully (all columns as string): {csv_path}")
    
    # === Save to Parquet ===
    parquet_path = os.path.join(DATA_DIR, "raw_data.parquet")
    os.makedirs(os.path.dirname(parquet_path), exist_ok=True)
    try:
        data.to_parquet(parquet_path, index=False, engine="pyarrow")
        print(f"✅ Saved Parquet to: {parquet_path}")
    except Exception as e:
        print("⚠️ Error saving Parquet:", e)
        print("Retrying after forcing all columns to strings...")
        data = data.astype(str)
        data.to_parquet(parquet_path, index=False, engine="pyarrow")
        print(f"✅ Saved Parquet successfully (all columns as string): {parquet_path}")

