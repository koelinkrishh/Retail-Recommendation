# Python file to clean our data from excel sheet into our main dataframe
import pandas as pd
import numpy as np
from src.logger import logging
from src.exception import CustomException
import os
import sys

import unicodedata, re

from src.config import *

# Loading functions from data ingestion
from src.Components.data_ingestion import load_data


class DataTransformation:
    def __init__(self):
        self.data = load_data(RAW_PATH_parquet, form="parquet")
        logging.info("Loaded data transformation class")
    
    
    @staticmethod
    def clean_text(self, string):
        string = str(string)
        if pd.isna(string): # handling missing values
            return ""
        
        string = unicodedata.normalize('NFKC', string) # normalize unicode
        # replace non-breaking space with space
        string = string.replace('\u00A0', ' ') 
        # remove zero width and BOM chars
        string = re.sub(r'[\u200B-\u200D\uFEFF]', '', string)
        return string.strip()
    
    # Function to locate invalid transactions

    
    def datatype_column(self):
        logging.info("Changing type of date columns")
        try:
            self.data['Customer_ID'] = self.data['Customer_ID'].astype(np.int64)
            self.data['Country'] = self.data['Country'].astype(str)

            self.data['StockCode'] = self.data['StockCode'].astype(str)
            self.data['Quantity'] = self.data['Quantity'].astype(np.int64)
            self.data['Price'] = self.data['Price'].astype(np.float64)
            self.data['Description'] = self.data['Description'].astype(str)
            self.data['Amount'] = self.data['Price'] * self.data['Quantity']
        except Exception as e:
            logging.debug("Error in datatype conversion")
            raise CustomException(e, sys)
    
    def Time_cleaning(self):
        """Function to clean datetime column an create features. """
        logging.info("Working on Datetime column")
        self.data['InvoiceDate'] = pd.to_datetime(self.data['InvoiceDate'], errors='coerce')
        self.data['Purchase_Date'] = self.data['InvoiceDate'].dt.normalize() # midnight average
        self.data['Purchase_time'] = self.data['InvoiceDate'].dt.strftime('%H:%M:%S') # keep as string but stardardized
        
        if 'Purchase_Time' not in self.data.columns:
            logging.warning("'Purchase_Time' not found — creating it from 'Purchase_Date' if available")
            if 'Purchase_Date' in self.data.columns:
                self.data['Purchase_Time'] = pd.to_datetime(self.data['Purchase_Date'], errors='coerce').dt.time
        
        self.data['Year'] = self.data['InvoiceDate'].dt.year
        self.data['Month'] = self.data['InvoiceDate'].dt.year
        self.data['Day'] = self.data['InvoiceDate'].dt.year
        

    
    def Locate_duplicate_transactions(self, subset_cols=None):
        """
        Manually detect duplicate rows based on subset_cols.

        Parameters:
        - df: pandas DataFrame
        - subset_cols: list of columns to check for duplicates.
                       If None, all columns are used.

        Returns:
        - List of indices of duplicate rows
        """
        if subset_cols is None:
            subset_cols = self.data.columns.tolist()

        seen = {}  # dictionary to store unique row signatures
        duplicate_indices = []

        # Iterate row by row
        for idx, row in self.data.iterrows():
            # Create a tuple of normalized values for subset columns
            row_signature = tuple(row[col] for col in subset_cols)
            # print(row_signature)

            if row_signature in seen:
                # Already seen → mark as duplicate
                duplicate_indices.append(idx)
            else:
                # First occurrence → store it
                seen[row_signature] = idx

        return duplicate_indices
    
    def Locate_invalid_transactions(self):
        logging.info("locating all invalid transactions")
        invalid_pattern = re.compile('|'.join([
            r'^\?+$',                          # only question marks
            r'^\.+$',                          # only dots
            r'^(?:nan|null|missing|unknown)$', # placeholders
            r'^(?:test|sample|invalid|none)$', # other placeholders
            r'^(?:postage|shipping|delivery|charges|discount|adjustment)$', # non-product rows
            r'^(?:gift card|gift certificate)$', # gift cards
            r'^(?:thrown|away|sales|given|manual|dotcom)$', # dirty items
        ]), flags=re.IGNORECASE
        )
        
        pattern = re.compile(
            r'^(?:\?+|\.{1,}|nan|null|missing|unknown|test|sample|invalid|n/a|none|postage|shipping|delivery|charges|discount|adjustment|thrown|away|sales|given)$',
            flags=re.IGNORECASE
        )
        
        # Use the regex pattern to filter rows
        logging.info("Dropping all invalid transactions")
        invalid_indices = self.data[self.data['Description'].str.contains(pattern, na=False)].index.tolist()
        return invalid_indices
    
    def Locate_accidental_transactions(self):
        buying_data = self.data[self.data['Quantity'] > 0].copy()
        returned_data = self.data[self.data['Quantity'] < 0].copy()
        
        buying_data['orig_index_buy'] = buying_data.index
        returned_data['orig_index_ret'] = returned_data.index
        if returned_data.iloc[0]['Quantity'] < 0:
            returned_data['Quantity'] = -returned_data['Quantity']
        
        # Merge on relevant keys
        keys = ['StockCode', 'Quantity', 'Customer_ID', 'Price', 'Country', 'Purchase_Date'] # same purchase data, means accidental buy or bad quality
        accidently_transactions = pd.merge(
            buying_data,
            returned_data,
            on=keys,
            how='inner',
            suffixes=('_buy', '_ret')
        )
        # The original indexes of matched transactions
        matched_indexes = accidently_transactions[['orig_index_buy', 'orig_index_ret']]
        matched_index = matched_indexes.values.tolist()

        # Flatten the list of matched indexes into a 1D array
        instant_return_ind = []
        for L in matched_index:
          for i in L:
            instant_return_ind.append(i)

        logging.info(f"Number of accidental transactions: {len(instant_return_ind)}")
        return instant_return_ind
    
    
    def Description_cleaning(self):
        try:
            # Clear description
            self.data['Description'] = self.data['Description'].str.strip().str.lower().apply(self.clean_text)
            self.data['Description'] = self.data['Description'].str.split().str.join('|')  # remove extra spaces
            self.data['Description'] = self.data['Description'].str.replace(r'\|+', '|', regex=True)  # replace multiple '|' with single '|'
        except:
            CustomException("Error in description cleaning", sys)

    def Removing_invalid_transactions(self):
        try:
            print("Original number of transactions in dataset:", self.data.shape[0])
            
            subset = ['Invoice', 'StockCode', 'Description', 'Quantity', 'Price', 'Customer_ID']
            
            rows_to_drop = set()
            
            duplicate_row = self.Locate_duplicate_transactions(subset_cols=subset)
            # duplicate_row = self.data.duplicated(subset=subset, keep='first').index
            logging.info(f"Found {len(duplicate_row)} duplicate transactions")
            logging.info(f"Removing duplicate transactions")
            rows_to_drop.update(duplicate_row)
            # print(len(rows_to_drop))

            invalid_row = self.Locate_invalid_transactions()
            logging.info(f"Found {len(invalid_row)} invalid transactions")
            logging.info(f"Removing invalid transactions")
            rows_to_drop.update(invalid_row)
            # print(len(rows_to_drop))
            
            accidental_row = self.Locate_accidental_transactions()
            logging.info(f"Found {len(accidental_row)} accidental transactions")
            rows_to_drop.update(accidental_row)
            # print(len(rows_to_drop))
            
            useless_trans = self.data[(self.data['Quantity'] <= 0) | (self.data['Price'] <= 0)].index
            logging.info(f"Found {len(useless_trans)} useless transactions. Removing them")
            rows_to_drop.update(useless_trans)
            # print(len(rows_to_drop))

            invalid_stockcode = self.data[self.data['StockCode'].str.match(r'^[A-Za-z]', na=False)].index
            logging.info(f"Found {len(invalid_stockcode)} invalid stock codes. Removing them")
            rows_to_drop.update(invalid_stockcode)
            # print(len(rows_to_drop))
            
            ## Droping all invalid rows
            self.data = self.data.drop(rows_to_drop).reset_index(drop=True)

            # Removing all other countries
            logging.info(f"Keeping only UK transactions")
            self.data = self.data[self.data['Country'].isin(['United Kingdom'])].reset_index(drop=True)
            
            print("Remaining transactions in dataset:", self.data.shape[0])
        except Exception as e:
            logging.debug("Error in removing invalid transactions")
            raise CustomException(e, sys)
        
        
    def Imputing(self):
        logging.info("Imputing missing column values")
        self.data['Customer_ID'] = self.data['Customer_ID'].fillna(-1)
        
        # Dropping all other missing transactions
        self.data = self.data.dropna()
        
    def Combine_transaction(self):
        # Combine left out items out of baskets
        self.data = self.data.groupby(['Invoice', 'StockCode'], as_index=False).agg({
            'Description': 'first',
            'Quantity': 'sum',
            'InvoiceDate': 'first',
            'Price': 'first',
            'Customer_ID': 'first',
            'Country': 'first',
            'Purchase_Date': 'first',
            'Purchase_Time': 'first',
            'Amount': 'sum',
        })
        
    def Save(self):
        # Convert all object columns to strings
        for col in self.data.select_dtypes(include='object').columns:
            self.data[col] = self.data[col].astype(str)
        
        self.data.to_parquet(Processed_PATH)
        

    def run(self):
        self.Imputing() # need to file na first
        self.datatype_column()
        self.Time_cleaning()
        self.Description_cleaning()
        self.Removing_invalid_transactions()
        self.Combine_transaction()
        self.Save()
    
    
if __name__ == "__main__":
    data_transformation = DataTransformation()
    data_transformation.run()
    
    # Loading new data
    test = pd.read_parquet(Processed_PATH)
    print("Shape of the dataset:", test.shape)
    print("test sample:", test.sample(5))
        



