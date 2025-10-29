## Python file for training model

import os
import sys
import joblib
import pickle
import copy
import tqdm

import pandas as pd
import numpy as np

from typing import Optional
from collections import defaultdict

from itertools import combinations
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.logger import logging
from src.exception import CustomException

# Loading function from other files
from src.Components.data_ingestion import load_data
from src.Components.data_processing import DataTransformation
from src.config import *

EPS = 1e-9

class Retail_recommendation_model:
    def __init__(self,
        initial_weights=None, d_effect=0.5, current_date=None, Half_life=7, Time_period=100,
        filter_items=False, include_description=True, iteration_bar=False):
        try:
            logging.info("Initializing Retail Recommendation model")
            self.df = load_data(Processed_PATH, 'parquet')
            self.customer_data = load_data(Customer_data_PATH, 'pickle')
            self.item_data = load_data(Item_data_PATH, 'pickle')
            self.rules = load_data(Rules_data_PATH, 'pickle')
            self.baskets = load_data(Baskets_data_PATH, 'pickle')

            self.include_description = include_description
            self.iteration_bar = iteration_bar

            self.vectorizer = None
            if self.include_description:
                try:
                    logging.info("Loading vectorizer...")
                    self.vectorizer = load_data(Vectorizer_data_PATH, form="joblib")
                except:
                    logging.warning("Vectorizer not found, initializing new one.")
                    self.vectorizer = TfidfVectorizer(ngram_range=(1,2), stop_words='english')
                    self.vectorizer.fit(self.item_data['Description'].astype(str))


            self.all_items = np.array(self.item_data['StockCode'].values)
            self.item_to_index = {item: i for i, item in enumerate(self.all_items)}
            self.n_items = len(self.all_items)

            
            if filter_items:
                self._remove_rare_items()

            # Time and weights
            default_weights = {
                'alpha': 1.0,   # history
                'beta': 0.75,    # rules
                'gamma': 1.0,   # recency multiplier weight
                'delta': 0.5,   # price
                'eta': 0.6,     # discount
                'epsilon': 0.4, # description similarity
                'd_effect': 1 # recency decay strength
            }
            self.weights = default_weights if initial_weights is None else dict(initial_weights)
            self.d_effect = d_effect if d_effect is not None else self.weights['d_effect']

            self.initial_weights = initial_weights
            self.current_date = ( pd.Timestamp(self.df['Purchase_Date'].max()) + pd.DateOffset(days=1) if current_date is None else pd.Timestamp(current_date) )
            self.Half_life = Half_life
            self.Time_period = Time_period
            
            """Method to setup our model for prediction"""
            # 1) Cache for features
            self.history_cache = {}
            self.recency_cache = {}
            
            # 2) Pre-compute static arrays
            self.Bias = self._compute_bias()
            self.rules_lookup = self._build_rules_index()
            self.rules_boost = self._prepare_rules_boost()
            
            # 3) Setup for description similarity
            self.similarity_matrix = None
            if self.include_description: # use vectorizer
                vocab = self.vectorizer.transform(self.item_data['Description'])
                self.similarity_matrix = cosine_similarity(vocab)
            
            # 4) Initialize best_params for training
            self.best_params = copy.deepcopy(self.weights)
            self.best_results = 0
            
            # 5) Initialize basket split
            # self.Basket_splitter(self.baskets, save=True, split_type='time')
            
            logging.info("Retail Recommendation model successfully initialized")
        except Exception as e:
            logging.error("Problem with Loading Retail Recommendation model")
            raise CustomException(e, sys)
            
    
    
    # ----- Helper Functions -----
    @staticmethod
    def Basket_splitter(baskets, train_frac=0.7, valid_frac=0.15, test_frac=0.15, save=True, split_type="random", artificial_budget=False):
        try:
            if abs(train_frac + valid_frac + test_frac - 1.0) > 1e-6:
                raise ValueError("train_frac, valid_frac and test_frac must sum to 1.0")
            # Preparing features for baskets -> Carts
            baskets = baskets[ baskets['Num_products']>= 5].copy()
            # we assume all purchases will be made today -> recency bias will not be trained well.
            baskets['X'] = baskets['StockCode'].apply(lambda x: x[:-1])
            baskets['Y'] = baskets['StockCode'].apply(lambda x: x[-1])
            
            if artificial_budget:
                baskets['Total_Cost'] = baskets['Price'].apply(np.sum)
                # randomly scale up.down total cost to simulate budget
                # ±20% variation (uniform)
                variation = np.random.uniform(0.9, 1.25, size=len(baskets))
                baskets['pseudo_budget'] = baskets['Total_Cost'] * variation
            
            logging.info("Splitting baskets into train, test, validation")
            
            if split_type=="random":
                ## Random sampling -> train test split
                train_baskets, test_baskets = train_test_split(baskets, shuffle=True, test_size=(1-train_frac), random_state=42)
                test_baskets, validation_baskets = train_test_split(test_baskets, shuffle=True, test_size=test_frac/valid_frac, random_state=42)
            elif split_type=="time":
                ## Time based splitting
                df_dates = baskets['Purchase_Date']
                cut1 = df_dates.quantile(train_frac)
                cut2 = df_dates.quantile(train_frac+valid_frac)
                train_baskets = baskets[df_dates < cut1]
                validation_baskets = baskets[(df_dates >= cut1) & (df_dates < cut2)]
                test_baskets = baskets[df_dates >= cut2]
            else:
                raise ValueError("split_type must be either 'random' or 'time'")
                
            if save:
                train_baskets.to_parquet(os.path.join(DATA_DIR, "train_baskets.parquet"))
                test_baskets.to_parquet(os.path.join(DATA_DIR, "test_baskets.parquet"))
                validation_baskets.to_parquet(os.path.join(DATA_DIR, "validation_baskets.parquet"))
                logging.info("Saved train, test, validation baskets")
                
            # print(train_baskets.sample(3))
            
            return train_baskets, test_baskets, validation_baskets
        except Exception as e:
            logging.error("Problem with splitting baskets")
            raise CustomException(e, sys)
    
    @staticmethod
    def _normalize_dict(d):
        if not d:
            return {}
        mv = max(d.values())
        if mv == 0:
            return {k: 0.0 for k in d}
        return {k: v / mv for k, v in d.items()}
    
    @staticmethod
    def _normalize_array(arr, method='minmax', eps=1e-8):
        """
        Robust feature normalization utility.
        Supports min-max or z-score normalization.
        Handles edge-cases: empty arrays, 1D arrays, constant columns.

        Parameters
        ----------
        arr : np.ndarray
            Input array of shape (batch_size, n_items) or 1D array
        method : str, optional
            'minmax' or 'zscore'. Default is 'minmax'.
        eps : float, optional
            Small constant to avoid division by zero.
        """
        if arr is None:
            return arr

        arr = np.array(arr, copy=False)

        # If totally empty -> return as is
        if arr.size == 0:
            return arr

        # Replace NaNs/inf
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

        # 1D case (vector)
        if arr.ndim == 1:
            if method == 'minmax':
                mn = np.min(arr)
                mx = np.max(arr)
                if (mx - mn) < eps:
                    return np.zeros_like(arr, dtype=np.float32)
                return ((arr - mn) / (mx - mn)).astype(np.float32)
            elif method == 'zscore':
                mu = np.mean(arr)
                sd = np.std(arr)
                if sd < eps:
                    return np.clip(arr - mu, -5.0, 5.0).astype(np.float32)
                return np.clip((arr - mu) / sd, -5.0, 5.0).astype(np.float32)
            else:
                raise ValueError(f"Unknown normalization method '{method}'")

        # 2D (or higher) — normalize column-wise
        if method == 'minmax':
            # use nanmin/nanmax to be robust
            min_vals = np.nanmin(arr, axis=0, keepdims=True)
            max_vals = np.nanmax(arr, axis=0, keepdims=True)
            denom = np.where((max_vals - min_vals) < eps, 1.0, (max_vals - min_vals))
            norm = (arr - min_vals) / denom
            return np.clip(norm, 0.0, 1.0).astype(np.float32)

        elif method == 'zscore':
            mean_vals = np.nanmean(arr, axis=0, keepdims=True)
            std_vals = np.nanstd(arr, axis=0, keepdims=True)
            denom = np.where(std_vals < eps, 1.0, std_vals)
            norm = (arr - mean_vals) / denom
            return np.clip(norm, -5.0, 5.0).astype(np.float32)

        else:
            raise ValueError(f"Unknown normalization method '{method}'")

    def _remove_rare_items(self, min_orders=300, min_customers=100):
        """
        Filter items to keep only those with sufficient popularity (Frequency > min_orders).
        Also cleans customer_data dictionaries to remove rare/unused items.

        Args:
            min_orders : int
                Minimum number of orders required for an item to be kept.
            min_customers : int
                Minimum number of unique customers required for an item to be kept.
        """
        logging.info(f"Removing rare items with min_orders={min_orders} and min_customers={min_customers}")
        df = self.item_data
        mask = (df['Num_orders'] > min_orders) & (df['Num_customers'] > min_customers)
        self.item_data = df.loc[mask].reset_index(drop=True)
        

    def _build_rules_index(self):
        logging.info("Building rules index")
        lookup = defaultdict(list)
        if self.rules.empty:
            return lookup
        for _, r in self.rules.iterrows():
            ant = tuple(sorted(r['antecedent']))
            lookup[ant].append(r)
        return lookup

    def _prepare_rules_boost(self):
        logging.info("Preparing rules boost")
        rb = {}
        for ant, rows in self.rules_lookup.items():
            boost_vector = defaultdict(float)
            for r in rows:
                boost_val = r['confidence'] * np.log1p(r['lift'])
                r['consequent'] = tuple(sorted(r['consequent']))
                for conseq in r['consequent']:
                    boost_vector[conseq] += boost_val

            rb[ant] = self._normalize_dict(boost_vector)
        return rb
    
    
    # ----- Feature Computations -----
    def _compute_bias(self):
        logging.info("Computing bias")
        freq = self.item_data['Frequency']
        bias = np.log1p(freq).astype(np.float32)
        return bias / bias.max()
    
    def compute_history(self, cust_id):
        if cust_id in self.history_cache:
            return self.history_cache[cust_id]
        
        cust_data = self.customer_data[self.customer_data['Customer_ID'] == cust_id]
        hist = np.zeros(self.n_items, dtype=np.float32)
        if cust_data.empty:
            self.history_cache[cust_id] = hist
            return hist

        count = cust_data.iloc[0]['Purchase_count']
        qty = cust_data.iloc[0]['Purchase_quantity']

        for item, q in qty.items():
            idx = self.item_to_index.get(item)
            quantity = qty.get(item, 0)
            hist[idx] = np.log1p(quantity / (count.get(item, 1) + EPS))

        self.history_cache[cust_id] = self._normalize_array(hist)
        return self.history_cache[cust_id]
    
    def compute_recency(self, cust_id, d_effect = None):
        if d_effect is None: d_effect = float(self.d_effect)
        
        cache_key = (cust_id, float(d_effect))
        if cache_key in self.recency_cache:
            return self.recency_cache[cache_key].copy()
        
        cust_data = self.customer_data[self.customer_data['Customer_ID'] == cust_id]
        rec = np.ones(self.n_items, dtype=np.float32)
        if cust_data.empty:
            self.recency_cache[cache_key] = rec
            return rec

        qty_dict = cust_data.iloc[0]['Purchase_quantity']
        freq = np.array([np.sqrt(np.log1p(qty_dict.get(it, 0))) for it in self.all_items], dtype=np.float32)
        freq /= (freq.max() + EPS)
        
        kmin = np.log(2) / self.Half_life
        kmax = np.log(2) / max(1.0, float(self.Time_period))
        K = kmin + (kmax - kmin) * freq

        last_purchase = cust_data.iloc[0]['Last_purchase_date']
        safe_min_date = pd.Timestamp(self.df['Purchase_Date'].min()) - pd.DateOffset(days=10) # pd.Timestamp("2000-01-01")
        deltas = np.array([(self.current_date - last_purchase.get(it, safe_min_date)).days for it in self.all_items], dtype=np.float32)
        deltas = np.clip(deltas, 0, 365)

        rec = 1.0 + float(d_effect) * np.exp(-K * deltas)
        self.recency_cache[cache_key] = rec
        return self.recency_cache[cache_key]
    
    def compute_rules(self, basket):
        rules_arr = np.zeros(self.n_items, dtype=np.float32)
        current = set(basket)
        for ant, boost_dict in self.rules_boost.items():
            if set(ant).issubset(current):
                for item, val in boost_dict.items():
                    idx = self.item_to_index.get(item)
                    if idx is not None:
                        rules_arr[idx] += val
        rules_arr = self._normalize_array(rules_arr)
        return rules_arr
    
    def compute_discount_array(self, discount_dict):
        arr = np.zeros(self.n_items, dtype=np.float32)
        if not discount_dict: return arr
        for item, val in discount_dict.items():
            idx = self.item_to_index.get(item)
            if idx is not None: 
                arr[idx] = np.log1p(val)
            
        if arr.max()>0: arr = self._normalize_array(arr)
        return arr

    def compute_price_array(self, budget):
        arr = np.zeros(self.n_items, dtype=np.float32)
        if budget is None:
            return arr
        
        prices = self.item_data['Current_Price'].values
        arr = np.log1p(budget/(prices + EPS)).astype(np.float32)
        return self._normalize_array(arr)
    
    def compute_description_boost(self, baskets_df):
        if self.similarity_matrix is None:
            raise ValueError("Description similarity not available. Enable include_description at init.")
        
        desc_map = {}
        # If empty input
        if baskets_df is None or len(baskets_df) == 0:
            return desc_map
        
        for idx, row in baskets_df.iterrows():
            basket_items = row.get('StockCode', None) if isinstance(row, dict) else row.get('StockCode', None) if hasattr(row, 'get') else None
                # fallback for pandas Series / Row object:
            if basket_items is None:
                try:
                    basket_items = row['StockCode']
                except Exception:
                    # try alternate names
                    basket_items = row.get('X', []) if hasattr(row, 'get') else []
            
            # robust conversion to list of strings
            if isinstance(basket_items, np.ndarray):
                basket_items = basket_items.tolist()
            if isinstance(basket_items, (str, int)):
                # single code -> put into list
                basket_items = [basket_items]
            if basket_items is None:
                basket_items = []
            
            # map items to indicies
            basket_idx = [self.item_to_index[it] for it in basket_items if it in self.item_to_index]
            if len(basket_idx) == 0:
                desc_map[idx] = np.zeros(self.n_items, dtype=np.float32)
                continue
            sim_vec = self.similarity_matrix[basket_idx]
            avg_vec = sim_vec.mean(axis=0)
            max_val = avg_vec.max()
            desc_map[idx] = avg_vec/(max_val+EPS) if max_val>0 else np.zeros(self.n_items, dtype=np.float32)
        
        # print("Description similarity computed.")
        # for key, array_desc in desc_map.items():
        #     print("For basket", key, " maximum is:", max(array_desc))
        return desc_map
    
    # ===  Helper: Compute all features in batch form   ===
    def _compute_features_batch(self, baskets_df, d_effect=None, budget_for_batch=None, discount_for_batch=None):
        try:
            logging.info("Computing features")
            if d_effect is None:
                d_effect = float(self.d_effect)

            n = len(baskets_df)
            H = np.zeros((n, self.n_items), dtype=np.float32)
            R = np.zeros((n, self.n_items), dtype=np.float32)
            T = np.ones((n, self.n_items), dtype=np.float32)
            P = np.zeros((n, self.n_items), dtype=np.float32)
            D = np.zeros((n, self.n_items), dtype=np.float32)

            for i, row in enumerate(baskets_df.itertuples(index=False)):               
                logging.info(f"Processing row {i+1}/{n}")
                basket_items = getattr(row, 'StockCode', getattr(row, 'X', []))
                # print("Basket items: ",basket_items)
                if isinstance(basket_items, np.ndarray):
                    basket_items = basket_items.tolist()
                if isinstance(basket_items, (str, int)):
                    basket_items = [basket_items]
                if basket_items is None:
                    basket_items = []
                
                cid = int(getattr(row, 'Customer_ID', -1))

                H[i, :] = self.compute_history(cid)
                R[i, :] = self.compute_rules(basket_items)
                T[i, :] = self.compute_recency(cid, d_effect=d_effect)

                # Budget
                budget_val = None
                if isinstance(budget_for_batch, (list, np.ndarray, pd.Series)) and len(budget_for_batch) > i:
                    budget_val = budget_for_batch[i]
                elif isinstance(budget_for_batch, (int, float)):
                    budget_val = budget_for_batch

                if budget_val is not None:
                    P[i, :] = self.compute_price_array(budget_val)

                # Discount
                if isinstance(discount_for_batch, dict):
                    D[i, :] = self.compute_discount_array(discount_for_batch)

            return {
                'H': self._normalize_array(H),
                'R': self._normalize_array(R),
                'T': T, # do not normalize recency
                'P': self._normalize_array(P),
                'D': self._normalize_array(D)
            }
        except Exception as e:
            logging.error(f"Error computing features: {e}")
            raise CustomException(e, sys)
        
    # ------------------ Recommendation ------------------
    def recommend(self, baskets, Coefficients=None, budget_column=None, discount_dict=None, top_n=100):
        try: 
            logging.info("Recommending next items...")
        
            if Coefficients is not None:
                self.weights = Coefficients

            # ---- Normalize Input ----
            single_input = False
            if isinstance(baskets, dict) or isinstance(baskets, pd.Series):
                baskets = pd.DataFrame([baskets])
                single_input = True
            elif (isinstance(baskets, pd.DataFrame) and baskets.shape[0]==1):
                single_input = True
            else:
                baskets = pd.DataFrame([baskets])
                single_input = True

            if budget_column:
                budget_dict = baskets[budget_column].to_dict()
            else:
                budget_dict = None

            logging.info("Computing features...")
            feats = self._compute_features_batch(baskets, d_effect=self.d_effect, budget_for_batch=budget_dict, discount_for_batch=discount_dict)
            H, R, T, P, D = feats['H'], feats['R'], feats['T'], feats['P'], feats['D']
            
            logging.info("Calculating logit...")
            Bias_mat = self.Bias if self.Bias.ndim == 2 else np.broadcast_to(self.Bias, (len(baskets), self.n_items))
            logit = (Bias_mat + self.weights['alpha']*H + self.weights['beta']*R +
                     self.weights['eta']*D + self.weights['delta']*P + self.weights['gamma']*T)
            
            if self.include_description:
                Desc_map = self.compute_description_boost(baskets)
                Desc_matrix = np.zeros((len(baskets), self.n_items), dtype=np.float32)
                for i, idx in enumerate(baskets.index):
                    Desc_matrix[i,:] = Desc_map.get(idx, np.zeros(self.n_items, dtype=np.float32))
                logit += self.weights['epsilon'] * np.log1p(Desc_matrix)

            # Softmax
            logit_max = logit.max(axis=1, keepdims=True)
            exps = np.exp(logit - logit_max)
            prob_matrix = exps / (exps.sum(axis=1, keepdims=True) + EPS)

            if self.iteration_bar:
                looper = tqdm.tqdm(enumerate(baskets.index), total=baskets.shape[0])
            else:
                looper = enumerate(baskets.index)

            logging.info("Getting top items...")
            results = {}
            for i, idx in looper:
                prob = prob_matrix[i]
                if top_n == -1 or top_n >= self.n_items:
                    top_idx = np.argsort(prob)[::-1]
                else:
                    # partial sort for top_n items
                    part = np.argpartition(prob, -top_n)[-top_n:]
                    top_idx = part[np.argsort(prob[part])[::-1]]
                top_codes = self.all_items[top_idx]
                top_probs = prob[top_idx]

                df_top = self.item_data[self.item_data['StockCode'].isin(top_codes)].copy()
                df_top = df_top.set_index('StockCode').loc[top_codes].reset_index()
                df_top['Probability'] = top_probs
                df_top['Rank'] = np.arange(1, len(df_top)+1)
                results[idx] = df_top

            logging.info("Successfully recommended next items.")
            return results[0] if single_input else results
        except Exception as e:
            logging.error("Problem with recommendation method.")
            raise CustomException(e, sys)

    def train_model(self, Carts, lr=0.05, n_iter=20, error=1e-6, sample_size=None,
            batch_size=32, top_n=100, budget_column=None, verbose=True,
            Learning_rate_decay=0, reg_lambda=0.0, clip_value=None, early_stopping=True, patience=5):
        """
        Args:
            - Carts: DataFrame with baskets data
            - lr: learning rate
            - n_iter: number of gradient descent iterations
            - error: Minimum error threshold for each item
            - sample_size: size of sample for training our self
            
            - batch_size: size of batches for prediction at each iteration
            - top_n: number of items to recommend
            
            - budget_column: Column name for budget value in our baskets dataset
        
        Parameters for regularization:
            - Learning_rate_decay: learning rate decay
            - reg_lambda: L2 regularization strength (0 => no reg)
            - clip_value: gradient clipping limit (None or 0 => no clipping)
            - early_stopping: enable early stopping
            - patience: early stopping patience (epochs)
        """
        try:
            logging.info("Initializing model training: ----")
            # save learning rate decay param as before
            self.lr_decay = Learning_rate_decay

            self.weights = self.weights.copy()
            self.weights['d_effect'] = self.d_effect
            history = []

            # Early stopping setup
            best_val_loss = np.inf
            patience_counter = 0
            self.best_result = 0

            logging.info("Mapping targets to indices")
            def map_targets_to_indices(Y_batch):
                return np.array([self.item_to_index.get(y, -1) for y in Y_batch], dtype=int)

            logging.info("For each baskets:")
            for epoch in range(1, n_iter + 1):
                # Optional sampling for speed
                if sample_size is not None and sample_size < len(Carts):
                    Carts_epoch = Carts.sample(sample_size, random_state=epoch)
                else:
                    Carts_epoch = Carts

                n_batches = int(np.ceil(len(Carts_epoch) / batch_size))

                total_loss = 0.0
                total_count = 0
                correct_total = 0
                grads = {key: 0.0 for key in self.weights.keys()}
                grads['d_effect'] = 0.0

                if self.include_description:
                    all_desc = self.compute_description_boost(Carts_epoch)

                loop = range(n_batches) if not verbose else tqdm.tqdm(range(n_batches), desc=f"Epoch {epoch}/{n_iter}")
                for b in loop:
                    # logging.info(f"Epoch {epoch}/{n_iter} ,Batch {b+1}/{n_batches}")
                    batch = Carts_epoch.iloc[b * batch_size:(b + 1) * batch_size]
                    if isinstance(batch, pd.Series):
                        batch = batch.to_frame().T
                    if batch.empty:
                        continue
                    
                    # extract budget if applied
                    budget_for_batch = None
                    if budget_column and budget_column in batch.columns:
                        budget_for_batch = batch[budget_column].values
                        
                    # ensure Y exists
                    if 'Y' not in batch.columns:
                        raise CustomException("Training batch missing 'Y' column (target item).", sys)
                    
                    X_batch = batch.drop(columns=['Y'], errors='ignore')
                    if X_batch.empty:
                        continue
                
                    feats = self._compute_features_batch(X_batch, d_effect=self.d_effect, budget_for_batch=budget_for_batch,)
                    Y_batch = batch['Y'].values
                    Y_idx = map_targets_to_indices(Y_batch)
                    valid_mask = (Y_idx != -1)

                    # --- Forward ---
                    # logging.info("Computing forward pass")
                    ## Use broadcasting: Bias may be 1d -> expand to shape (batch_size, n_items)
                    Bias_mat = self.Bias if self.Bias.ndim == 2 else np.broadcast_to(self.Bias, (len(X_batch), self.n_items))
                    
                    logit = (Bias_mat
                             + self.weights['alpha'] * feats['H']
                             + self.weights['beta'] * feats['R']
                             + self.weights['gamma'] * feats['T']
                             + self.weights['delta'] * feats['P']
                             + self.weights['eta'] * feats['D']).astype(np.float32)

                    # logging.info("Adding description boost")
                    if self.include_description and self.weights['epsilon'] != 0.0:
                        Desc_mat = np.vstack([all_desc[i] for i in X_batch.index]) # shape [batch_size, n_items]
                        # transformation in your self: epsilon * log(1 + C)
                        Desc_log = np.log1p(Desc_mat + error)
                        logit += self.weights['epsilon'] * Desc_log

                    logit = logit.astype(np.float32)
                    logit -= logit.max(axis=1, keepdims=True)
                    exps = np.exp(logit)
                    probs = exps / (exps.sum(axis=1, keepdims=True) + EPS) # [batch_size, n_items]

                    # logging.info("Computing loss")
                    # --- Loss ---
                    row_ids = np.arange(len(Y_batch))[valid_mask]
                    p_y = probs[row_ids, Y_idx[valid_mask]]

                    if top_n != -1 and top_n < self.n_items:
                        part = np.argpartition(probs, -top_n, axis=1)[:, -top_n:]
                        order = np.argsort(probs[np.arange(probs.shape[0])[:, None], part], axis=1)[:, ::-1]
                        top_idx_matrix = np.take_along_axis(part, order, axis=1)
                        isin_top = np.any(top_idx_matrix == Y_idx[:, None], axis=1)

                        if isin_top.any():
                            idxs_ok = row_ids[isin_top]
                            total_loss += -np.log(probs[idxs_ok, Y_idx[valid_mask][isin_top]] + EPS).sum()

                        penalized_count = (~isin_top).sum()
                        if penalized_count > 0:
                            total_loss += (-np.log(error)) * penalized_count
                    else:
                        total_loss += -np.log(p_y + EPS).sum()

                    total_count += valid_mask.sum()

                    # logging.info("Computing model accuracy")
                    # --- Accuracy ---
                    if top_n == -1 or top_n >= self.n_items:
                        top_idx_matrix_full = np.argsort(probs, axis=1)[:, ::-1]
                    else:
                        part = np.argpartition(probs, -top_n, axis=1)[:, -top_n:]
                        order = np.argsort(probs[np.arange(probs.shape[0])[:, None], part], axis=1)[:, ::-1]
                        top_idx_matrix_full = np.take_along_axis(part, order, axis=1)

                    topn_codes = self.all_items[top_idx_matrix_full]
                    true_codes = Y_batch[valid_mask][:, None]
                    valid_topn_codes = topn_codes[valid_mask]

                    correct_total += int(np.sum(np.any(valid_topn_codes == true_codes, axis=1)))

                    # --- Gradients ---
                    # logging.info("Computing gradients")
                    one_hot = np.zeros_like(probs ,dtype=np.float32)
                    one_hot[np.arange(len(Y_batch))[valid_mask], Y_idx[valid_mask]] = 1.0

                    # derivative of cross-entropy wrt logits
                    valid_count = max(1, int(valid_mask.sum()))
                    diff = (probs - one_hot) / valid_count # Y_batch can have invalid values so use valid_mask

                    grads['alpha'] += np.sum(diff * feats['H'])
                    grads['beta'] += np.sum(diff * feats['R'])
                    grads['gamma'] += np.sum(diff * feats['T'])
                    grads['delta'] += np.sum(diff * feats['P'])
                    grads['eta'] += np.sum(diff * feats['D'])

                    if self.include_description and self.weights['epsilon'] != 0.0:
                        grads['epsilon'] = grads.get('epsilon', 0.0) + np.sum(diff * Desc_log)
                    else:
                        grads['epsilon'] = grads.get('epsilon', 0.0) # ensure key exists

                    # feats['T'] = 1 + d_effect * exp(-K * delta) 
                    # dT/d(d_effect) = exp(-K * delta)
                    # so d(logit)/ dT * dT/d(d_effect) =  d(logit)/d(d_effect) = gamma * exp(-K * delta)
                    # compute exp(-K * delta) safely: feats['T'] - 1 = d_effect * exp(-K * delta)  => exp(-K * delta) = (T - 1)/d_effect  (d_effect used to compute feats)
                    d_eff_val = float(self.weights.get('d_effect', 0.0))
                    if abs(d_eff_val) > 1e-12:
                        E_mat = (feats['T'] - 1.0) / (d_eff_val + EPS)
                    else:
                        # fallback: if d_effect is ~0 , approximate exp(-K * delta) as T-1 because T ≈ 1 + small*E
                        E_mat = (feats['T'] - 1.0)
                    
                    # scale back to avoid underflow due to (1/valid_count)
                    grads['d_effect']  = grads.get('d_effect', 0.0) + (np.sum(diff * (self.weights['gamma']* E_mat)))*valid_count
                    # print(grads['d_effect'])
                
                    
                # logging.info("Implementing regularization")
                # --- L2 Regularization ---
                if reg_lambda > 0:
                    l2_term = sum(v**2 for v in self.weights.values())
                    total_loss += reg_lambda * l2_term # adding l2 term to loss

                    # Update gradient to include L2 term
                    for k in self.weights.keys():
                        if k=="d_effect":
                            grads['d_effect'] += 2.0 * reg_lambda * self.d_effect
                        else:
                            grads[k] += 2.0 * reg_lambda * self.weights[k]

                # --- Gradient Clipping + Normalize gradients ---
                norm = np.sqrt(sum(g**2 for g in grads.values()))
                max_norm = clip_value
                if (clip_value is not None) and norm > max_norm:
                    for k in grads.keys():
                        grads[k] = grads[k] *(max_norm / (norm+1e-8))

                # Normalize + update
                if total_count == 0:
                    if verbose:
                        print("No valid samples; stopping early.")
                    break
                
                effective_lr = lr / (1.0 + self.lr_decay * epoch) # learning rate decay
                for k, g in grads.items():
                    # print(f"Gradients for {k}: {g}")
                    if k == 'd_effect':
                        self.d_effect -= effective_lr * grads[k]
                        self.weights[k] -= effective_lr * grads[k]
                    else:
                        if k not in self.weights: # skip weights update if gradient for feature is missing
                            continue
                        self.weights[k] -= effective_lr * grads[k]

                epoch_loss = total_loss / max(1, total_count)
                epoch_acc = correct_total / max(1, total_count)
                history.append((epoch_loss, epoch_acc, copy.deepcopy(self.weights)))

                if verbose:
                    print(f"Epoch {epoch}/{n_iter} | Loss: {epoch_loss:.6f} ") # | Acc: {epoch_acc:.4f}
                    print(f"self.weights: { {k: round(v,4) for k,v in self.weights.items()} }")
                    print(f"Correct predictions:  {correct_total}/{total_count} "  )

                # --- Early stopping evaluation (validate and possibly stop) ---
                # logging.info("Early stopping evaluation")
                if early_stopping:
                    if epoch_loss + error < best_val_loss: # reset counter
                        patience_counter = 0
                        best_val_loss = epoch_loss
                        best_epoch = epoch
                    else:
                        patience_counter += 1

                    if patience_counter >= patience:
                        if verbose:
                            print(f"Early stopping triggered at epoch {epoch}. Restoring best parameters from epoch {best_epoch}.")
                            pass
                        self.d_effect = self.best_params.get('d_effect', self.d_effect)
                        self.weights = {k: v for k, v in self.best_params.items() if k != 'd_effect'}
                        break
                    
                if correct_total > self.best_result:
                    self.best_result = correct_total
                    self.best_params = copy.deepcopy(self.weights)


            final_weights = copy.deepcopy(self.weights)
            return final_weights, history
        except Exception as e:
            logging.error("Problem during model training")
            raise CustomException(e, sys)
            
    
if __name__ == "__main__":
    # Minimal test
    try:
        # Instantiate model
        model = Retail_recommendation_model(include_description=False, iteration_bar=False)
        print("Model initialized successfully.")

        
        train_baskets = load_data(TRAIN_FILE, form="parquet")
        
        print("Starting model training: ")
        final_weights, history = model.train_model(Carts=train_baskets.sample(1000), lr=0.05, n_iter=5, top_n=100, verbose=True)
        print(f"Final weights: {final_weights}")
        logging.info("Model successfully trained")
        
        
        test_basket = model.baskets.sample(5)
        print("Now recommendation: ")
        rec = model.recommend(test_basket, top_n=10, batch_size=5)
        
        for idx in test_basket.index:
            print(f"Original {idx} basket: \n {test_basket.loc[idx]}")
            print("Recommended items:")
            print(rec[idx])
    except Exception as e:
        logging.error("Problem during Running File.")
        raise CustomException(e, sys)
