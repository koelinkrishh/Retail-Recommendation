# Code for feature engineering:

import os
import sys
from src.logger import logging
from src.exception import CustomException

import numpy as np
import pandas as pd
import joblib
import pickle


# File to note all paths
from src.Components.data_ingestion import load_data
from src.Components.data_processing import DataTransformation
from src.config import *

from sklearn.feature_extraction.text import TfidfVectorizer
from mlxtend.preprocessing import TransactionEncoder
from efficient_apriori import apriori
# from mlxtend.frequent_patterns import apriori, association_rules


class FeatureEnginerring():
    def __init__(self):
        self.data = load_data(Processed_PATH, form="parquet")
        logging.info("Loaded feature enginerring class")
        
        self.data = self.data.sort_values('InvoiceDate')
    
    # We will create methods to create features and dataset format we need for our model.
    def Build_Item_dataset(self):
        try:
            logging.info("Building item dataset.")
            item_summary = self.data.groupby('StockCode').agg(
                Description = ("Description", 'first'),
                Current_Price = ("Price", 'last'),
                Num_orders = ("Invoice", 'nunique'),
                Total_quantity = ("Quantity", 'sum'),
                Total_sales = ("Amount",'sum'),
                Num_customers = ("Customer_ID", 'nunique'),
                Last_sale = ('Purchase_Date', 'last')
            ).reset_index().sort_values('Total_quantity', ascending=False)

            item_summary['Frequency'] = item_summary['Num_orders']/item_summary['Num_customers']
            
            logging.info(f"Sucessfully built item dataset with shape: {item_summary.shape}")
            return item_summary
        except Exception as e:
            logging.debug(f"Error in aggregating item dataset: {e}")
            raise CustomException(e, sys)
        
    def Build_customer_dataset(self):
        try:
            logging.info("Building customers dataset.")
            customer = self.data.groupby('Customer_ID')

            def aggregate_customer(group):
                purchase_count = group.groupby("StockCode")['Invoice'].nunique().to_dict()
                purchase_qty = group.groupby("StockCode")['Quantity'].sum().to_dict()
                last_purchase_date = group.groupby("StockCode")['Purchase_Date'].max().to_dict()

                # Need to return series to concatenate into multiple columns
                return pd.Series({
                    'StockCode': sorted(group['StockCode'].unique()),
                    'Purchase_count': purchase_count,
                    'Purchase_quantity': purchase_qty,
                    'Last_purchase_date': last_purchase_date
                })

            customer = customer.apply(aggregate_customer, include_groups=False).reset_index()
            
            logging.info(f"Sucessfully aggregated customer dataset with shape: {customer.shape}")
            return customer 
           
        except Exception as e:
            logging.debug(f"Error in aggregating item dataset: {e}")
            raise CustomException(e, sys)
    
    def Reduce_transactions(self, threshold=0.02):
        percentage = self.data['StockCode'].value_counts()/self.data['Invoice'].nunique()
        percentage = percentage.sort_values(ascending=False)
        rare_items = percentage[percentage < threshold].index.to_list() # item should be present more then 2% of its part in total dataset
        
        self.data = self.data[~self.data['StockCode'].isin(rare_items)]
    
    def Build_baskets_dataset(self):
        try:
            logging.info("Building baskets dataset.")
            baskets = self.data.groupby(["Invoice", "InvoiceDate"]).agg({
                "Customer_ID": 'first', # sale for all rows in the invoice
                "Purchase_Date": 'first',
                "Purchase_Time": 'first',
                
                # aggregate product-level details
                "StockCode": lambda x: list(x),
                "Description": lambda x: list(x),
                "Quantity": lambda x: list(x),
                "Price": lambda x: list(x),
                "Amount": lambda x: list(x),
            })
            ## adding more columns
            baskets['Total_amount'] = baskets['Amount'].apply(sum)
            baskets['Num_products'] = baskets['StockCode'].apply(len)
            
            baskets = baskets.reset_index().rename(columns={'index': 'Invoice'})

            logging.info(f"Sucessfully aggregated baskets with shape: {baskets.shape}")
            print("Baskets columns: ", baskets.columns)
            return baskets
        
        except Exception as e:
            logging.debug(f"Error in aggregating baskets: {e}")
            raise CustomException(e, sys)
    
    ## Feature engineering for description:
    def Build_vocabulary(self):
        try:
            logging.info("Building vocabulary.")
            vectorizer = TfidfVectorizer(ngram_range=(1,2), stop_words='english', max_features=2000)
            vectorizer.fit(self.data['Description'])
            
            print("Vocabulary size: ",len(vectorizer.vocabulary_))
            
            return vectorizer
        except Exception as e:
            logging.error("Problem with vectorizer.")
            raise CustomException(e, sys)
        
    logging.info("Now, we will find association rules for the itemsets.")
            
    def process_baskets(self):
        logging.info("Processing baskets.")
        basket = self.Build_baskets_dataset()
        
        self.transactions = basket['StockCode'].apply(lambda x: list(set(x))).tolist()
        
        # te = TransactionEncoder()
        # te_ary = te.fit_transform(self.transactions, sparse=True)
        
        # basket_bool = pd.DataFrame.sparse.from_spmatrix(te_ary, columns=te.columns_)
        # # print("Baskets shape: ", basket_bool.shape)
        
        logging.info("Baskets processed.")
        return basket
    
    def association_rules(self):
        logging.info("Finding association rules using Apriori algorithm.")
        basket = self.process_baskets()

        # Run Apriori directly on list of transactions
        itemsets, ruleset = apriori(
            self.transactions,
            min_support=0.01,        # 1% support threshold
            min_confidence=0.2,      # 20% confidence
            max_length=3,             # up to 3 items
        )

        rules_df = pd.DataFrame([{
                'antecedent': tuple(rule.lhs),
                'consequent': tuple(rule.rhs),
                'support': rule.support,
                'confidence': rule.confidence,
                'lift': rule.lift
            } for rule in ruleset
        ]).sort_values('lift', ascending=False)
        
        logging.info("Association rules found.")
        print(rules_df.head())
        
        return basket, rules_df
    
    def Run(self):
        try:
            logging.info("Saving all datasets.")
            
            items = self.Build_Item_dataset()
            customer = self.Build_customer_dataset()
            self.Reduce_transactions() # reduce itemset

            try:
                # Dont save vocabulary as it is too large
                vectorizer = joblib.load(Vectorizer_data_PATH)
            except:
                vectorizer = self.Build_vocabulary()
                joblib.dump(vectorizer, Vectorizer_data_PATH)
                logging.info("Saved Vectorizer model: ")
                
            basket, ruleset = self.association_rules()


            items.to_pickle(Item_data_PATH)
            logging.info("Saved items datasets")
            customer.to_pickle(Customer_data_PATH)
            logging.info("Saved Customers purchase history datasets")
            basket.to_pickle(Baskets_data_PATH)
            logging.info("Saved Baskets")


            ruleset.to_pickle(Rules_data_PATH)
            logging.info("Saved Association rules: ")
        except Exception as e:
            logging.error("Error while running Feature Engineering.")
            raise CustomException(e, sys)
        
    
    
if __name__ == "__main__":
    obj = FeatureEnginerring()
    obj.Run()

