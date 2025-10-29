## Python File to Evaluate the model performance
import sys

import pandas as pd
import numpy as np


from src.logger import logging
from src.exception import CustomException

from src.Components.data_ingestion import load_data
from src.config import *
from src.Components.train import Retail_recommendation_model


""" In this file, we will evaluate our model's current states performance """


class Evaluation:
    def __init__(self, MODEL: Retail_recommendation_model):
        if MODEL is None:
            raise ValueError("Model Evaluation requires is not initialized.")

        self.model = MODEL
        self.metrics = {}
    
    
    def evaluate(self, test_data, top_n=100, batch_size=128, budget_column=None, error=1e-8, verbose=True):
        """
        Evaluate the model on unseen baskets using its recommend() method.

        Computes:
            - Precision@N, Recall@N, F1@N
            - HitRate (Hit@N)
            - Rank-aware metrics: MRR, NDCG, MAP

        Returns:
            dict of aggregated metrics.
        """
        if test_data.empty:
            raise ValueError("Test baskets dataframe is empty.")
        test_data = test_data.reset_index(drop=True)
        self.top_n = top_n
        self.error = error
        
        if batch_size is not None:
            self.batch_size = min(batch_size, len(test_data))
        
        # --- Run inference ---
        logging.info(f"Running evaluation with top N = {top_n}")
        try:
            results = self.model.recommend(test_data, top_n=top_n, batch_size=self.batch_size,
                budget_column=budget_column,)    
        except Exception as e:
            raise CustomException(f"Error in model.recommend: {e}", sys)
        
        ## ----- Metrics Computation ----- ##
        hits, total_true = 0, len(test_data)
        precisions, recalls, f1s = [],[],[]
        reciprocal_ranks, ndcgs, average_precisions = [],[],[]
        
        for idx, row in test_data.iterrows():
            true_item = row['Y']
            if true_item is None or idx not in results:
                logging.warning(f"Baskets {idx} is skipped.")
                continue
            
            result = results[idx]
            if 'Rank' in result.columns:
                rank_item = result.sort_values('Rank')['StockCode'].tolist()[:top_n]
            else:
                rank_item = result.sort_values('Probability')['StockCode'].tolist()[:top_n]
            
            if not rank_item:
                logging.debug("Top ranked items not found")
                continue
            
            # Hit based calculation
            hit_i = 1 if (true_item in rank_item) else 0
            hits += hit_i
            
            precision_i = hit_i/len(rank_item)
            recall_i = hit_i
            f1_score = 2 *((precision_i*recall_i)/(precision_i + recall_i + self.error))
            
            precisions.append( precision_i ) # chance of item being in their
            recalls.append( recall_i )
            f1s.append( f1_score )
            
            # Rank based metrics
            if hit_i:
                rank = rank_item.index(true_item) + 1
                reciprocal_ranks.append( 1.0/rank )
                ndcgs.append( 1.0/np.log2(rank+1) )
                average_precisions.append(1.0/rank)
            else:
                reciprocal_ranks.append(0.0)
                ndcgs.append(0.0)
                average_precisions.append(0.0)
        
        # === Aggregating metrics ===
        self.metrics = {
            'Precision@N': np.mean(precisions),
            'Recall@N': np.mean(recalls),
            'F1@N': np.mean(f1s),
            'HitRate': hits/ max(1, total_true),
            'MRR@N': np.mean(reciprocal_ranks),
            'NDCG@N': np.mean(ndcgs),
            'MAP@N': np.mean(average_precisions),
            'Correct_predictions': hits,
            'Total_baskets': total_true
        }
        
        logging.info("Evaluation completed successfully")
        for k,v in self.metrics.items():
            logging.info(f"{k:<20}: {v:.4f}")
            if verbose:
                print(f"{k} -> {v}")
            
        return self.metrics
    

if  __name__ == "__main__":
    np.random.seed(42)
    
    rec = Retail_recommendation_model(include_description=True, d_effect=1.0)
    obj = Evaluation(rec)
    
    train_data = load_data(TRAIN_FILE, form="parquet")
    test_data = load_data(TEST_FILE, form="parquet")

    print("\nStarting training...")
    final_weights, history = rec.train_model(
        train_data,
        lr=0.2,
        n_iter=10,
        top_n=100,
        batch_size=64,
        Learning_rate_decay=0.05,
        reg_lambda=0.05,
        clip_value=5.0,
        early_stopping=True,
        patience=7,
        verbose=True
    )
    evaluator = Evaluation(rec)
    metrics = evaluator.evaluate(test_data, top_n=100)
    print(metrics)
            
            
            
            
        
