## File for importing into Training notebook
import pandas as pd
import numpy as np
from collections import defaultdict
import time
import math
import tqdm

import pickle
import joblib
import copy

from itertools import combinations
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class Retail_Recommendation:
    """
    Extremely optimized version (description boost removed).
    Uses full NumPy vectorization, cached arrays, and minimal loops.
    """

    def __init__(self, items_data, customer_data, rules, vectorizer=None, vectorizer_path=None,
        initial_weights=None, d_effect=1, current_date=None, Half_life = 7, Time_period=100, 
        filter_items=False, include_description=False, iteration_bar = True):
        """
        Args:
            items_data (pd.DataFrame): must contain ['StockCode','Description','Current_Price',
                                                  'Total_quantity','Num_orders'].
            customer_data (pd.DataFrame): rows may contain dicts in columns 'Purchase count',
                                        'Purchase quantity', 'Last purchase date'.
            rules_df (pd.DataFrame): cols ['antecedent','consequent','confidence','lift'].
            vectorizer: optional pre-trained sklearn vectorizer for descriptions (TF-IDF).
            vectorizer_path: Path to pre-trained sklearn vectorizer for descriptions (TF-IDF).
            initial_weights: dict with keys alpha,beta,delta,gamma,epsilon,eta (defaults used if None).
            d_effect: base recency multiplier for items (floats).
            today: pd.Timestamp or pd.Date for "current" date; if None derived from items_df if possible.
            time_period: integer days span; if None computed from items_df.
            filter_items: bool, whether to remove rare items from item data.
            include_description: bool, whether to use description similarity.
            iteration_bar: bool, whether to show a progress bar for scoring.
        """
        self.iteration_bar = iteration_bar
        
        self.item_data = items_data.copy()
        self.customer_data = customer_data
        self.rules = rules

        # Items setup
        if filter_items:
            self._remove_rare_items()
        self.all_items = np.array(self.item_data['StockCode'].values)
        self.item_to_index = {item: i for i, item in enumerate(self.all_items)}
        self.n_items = len(self.all_items)

        # Time and weights
        default_weights = {
            'alpha': 1.0,   # history
            'beta': 1.0,    # rules
            'delta': 0.5,   # price
            'eta': 0.5,     # discount
            'gamma': 0.5,   # recency multiplier weight
            'epsilon': 0.2, # description similarity
            'd_effect': 0.5 # recency decay strength
        }
        self.weights = default_weights if initial_weights is None else dict(initial_weights)
        # # ensure all keys exist
        # for k, v in default_weights.items():
        #     self.weights.setdefault(k, v)

        if d_effect is not None:
            self.d_effect = d_effect
        elif 'd_effect' in self.weights:
            self.d_effect = self.weights['d_effect']
        else:
            self.d_effect = 0.5
        
        # self.current_date = ( pd.Timestamp.now().date() if current_date is None else pd.Timestamp(current_date) )
        self.current_date = ( pd.Timestamp(df['Purchase Date'].max()) + pd.DateOffset(days=1) if current_date is None else pd.Timestamp(current_date) )
        self.Time_period = (
            pd.Timedelta(days=100) # (items_data['Last_sale'].max() - items_data['Last_sale'].min()).days
            if Time_period is None else Time_period
        )
        self.half_life = Half_life

        # Caches -> keted by (cust_id, d_effect)
        self.history_cache = {}
        self.recency_cache = {}

        # Precompute static arrays
        self.Bias = self._compute_bias()
        self.rules_lookup = self._build_rules_index()
        self.rules_boost = self._prepare_rules_boost()
        
        self.include_description = include_description
        
        if self.include_description:
                # ----------------- Description similarity -----------------
            if 'Clean_Description' not in self.item_data.columns:
                self.item_data = self.item_data.copy()
                self.item_data['Clean_Description'] = (self.item_data['Description'].str.lower().str.replace('[^A-Za-z]+',' ', regex=True).str.strip())
            if vectorizer is None:
                if vectorizer_path:
                    import joblib
                    vectorizer = joblib.load(vectorizer_path)
                else:
                    vectorizer = TfidfVectorizer(ngram_range=(1,2), stop_words='english')
                    vectorizer.fit(self.item_data['Clean_Description'])
            self.vectorizer = vectorizer

            self.vocab = self.vectorizer.transform(self.item_data['Clean_Description'])
            self.similarity_matrix = cosine_similarity(self.vocab, dense_output=False).astype(np.float32)
        
        # Initialize best_params for training/early stopping
        self.best_params = copy.deepcopy(self.weights)
        self.best_result = 0
        

    # ------------------ Helper methods ------------------
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
        df = self.item_data
        mask = (df['Num_orders'] > min_orders) & (df['Num_customers'] > min_customers)
        self.item_data = df.loc[mask].reset_index(drop=True)
        self.all_items = np.asarray(self.item_data['StockCode'].values)
        self.item_to_index = {item: i for i, item in enumerate(self.all_items)}
        self.n_items = len(self.all_items)
        
        print(f"Number of important items: {self.item_data.shape[0]}")
    
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

        Parameters
        ----------
        arr : np.ndarray
            Input array of shape (batch_size, n_items)
        method : str, optional
            'minmax' or 'zscore'. Default is 'minmax'.
        eps : float, optional
            Small constant to avoid division by zero.

        Returns
        -------
        np.ndarray
            Normalized array of same shape as input.
        """
        if arr is None or not isinstance(arr, np.ndarray):
            return arr

        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

        if method == 'minmax':
            # Compute per-feature (column-wise) min and max
            min_vals = np.min(arr, axis=0, keepdims=True)
            max_vals = np.max(arr, axis=0, keepdims=True)
            denom = np.where((max_vals - min_vals) < eps, 1.0, (max_vals - min_vals))
            norm = (arr - min_vals) / denom
            return np.clip(norm, 0.0, 1.0)

        elif method == 'zscore':
            # Standard score normalization (zero mean, unit variance)
            mean_vals = np.mean(arr, axis=0, keepdims=True)
            std_vals = np.std(arr, axis=0, keepdims=True)
            denom = np.where(std_vals < eps, 1.0, std_vals)
            norm = (arr - mean_vals) / denom
            # optional: clip extreme z-scores
            return np.clip(norm, -5.0, 5.0)

        else:
            raise ValueError(f"Unknown normalization method '{method}'")

    def _build_rules_index(self):
        lookup = defaultdict(list)
        for _, r in self.rules.iterrows():
            ant = tuple(sorted(r['antecedent']))
            lookup[ant].append(r)
        return lookup

    def _prepare_rules_boost(self):
        rb = {}
        for ant, rows in self.rules_lookup.items():
            boost_vector = defaultdict(float)
            for r in rows:
                boost_val = r['confidence'] * math.log1p(r['lift'])
                for conseq in r['consequent']:
                    boost_vector[conseq] += boost_val
            # Normalize
            rb[ant] = self._normalize_dict(boost_vector)
        return rb

    def _compute_bias(self):
        """
        Compute the bias feature vector for all items.

        The bias feature is the item's frequency, normalized by
        the maximum frequency.

        Returns:
            bias : numpy.ndarray
                A vector of the bias feature for all items.
        """

        freq = self.item_data['Frequency'].values
        bias = np.log1p(freq).astype(np.float32)
        return self._normalize_array(bias)

    # ------------------ Feature computations ------------------
    def compute_history(self, cust_id):
        """
        Compute the history feature vector for a given customer.

        The history feature is the item's purchase quantity normalized by
        the item's purchase count, and then normalized by the maximum value.

        Args:
            cust_id : int
                The customer ID for which to compute the history feature.

        Returns:
            hist : numpy.ndarray
                A vector of the history feature for all items.
        """
        if cust_id in self.history_cache:
            return self.history_cache[cust_id]

        cust_data = self.customer_data[self.customer_data['Customer ID'] == cust_id]
        hist = np.zeros(self.n_items, dtype=np.float32)
        if cust_data.empty:
            self.history_cache[cust_id] = hist
            return hist

        count = cust_data.iloc[0]['Purchase count']
        qty = cust_data.iloc[0]['Purchase quantity']

        for item, q in qty.items():
            idx = self.item_to_index.get(item)
            quantity = qty.get(item, 0)
            hist[idx] = math.log1p(quantity / count[item])

        self.history_cache[cust_id] = self._normalize_array(hist)
        return self.history_cache[cust_id]

    def compute_recency(self, cust_id, d_effect = None):
        """
        Compute the recency feature vector for a given customer.

        The recency feature is the item's purchase quantity normalized by
        the item's purchase quantity, and then normalized by the maximum value.

        Args:
            cust_id : int
                The customer ID for which to compute the recency feature.
            d_effect: float
                The decay factor for the recency feature.

        Returns:
            rec : numpy.ndarray
                A vector of the recency feature for all items.
        """
        if d_effect is None: d_effect = float(self.d_effect)
        
        
        cache_key = (cust_id, float(d_effect))
        if cache_key in self.recency_cache:
            return self.recency_cache[cache_key].copy()
        
        cust_data = self.customer_data[self.customer_data['Customer ID'] == cust_id]
        rec = np.ones(self.n_items, dtype=np.float32)
        if cust_data.empty:
            self.recency_cache[cache_key] = rec
            return rec

        qty_dict = cust_data.iloc[0]['Purchase quantity']
        freq = np.array([np.sqrt(np.log1p(qty_dict.get(it, 0))) for it in self.all_items], dtype=np.float32)
        freq /= (freq.max() + EPS)

        kmin = np.log(2) / self.half_life
        if isinstance(self.Time_period, (int, float)):
            tp_days = max(1.0, float(self.Time_period))
        else:
            tp_days = max(1.0, float(getattr(self.Time_period, "days", 100)))
        kmax = np.log(2) / tp_days
        K = kmin + (kmax - kmin) * freq

        last_purchase = cust_data.iloc[0]['Last purchase date']
        safe_min_date = pd.Timestamp(df['Purchase Date'].min()) - pd.DateOffset(days=10) # pd.Timestamp("2000-01-01")
        deltas = np.array(
            [(self.current_date - last_purchase.get(it, safe_min_date)).days for it in self.all_items],
            dtype=np.float32
        )
        
        max_delta = 365
        deltas = np.clip(deltas, 0, max_delta)

        rec = 1.0 + float(d_effect) * np.exp(-K * deltas)
        self.recency_cache[cache_key] = self._normalize_array(rec)
        return self.recency_cache[cache_key]

    def compute_rules_array(self, basket):
        """
        Compute the rules feature array for a given basket.

        Args:
            basket : set
                The set of items in the basket.

        Returns:
            rules_arr : numpy.ndarray
                A vector of the rules feature for all items.
        """
        rules_arr = np.zeros(self.n_items, dtype=np.float32)
        if not basket: return rules_arr
        
        current = set(basket)
        for ant, boost_dict in self.rules_boost.items():
            if set(ant).issubset(current):
                for item, val in boost_dict.items():
                    idx = self.item_to_index.get(item)
                    if idx is not None:
                        rules_arr[idx] += val
        if rules_arr.max() > 0:
            rules_arr = self._normalize_array(rules_arr)
        return rules_arr

    def compute_discount_array(self, discount_dict):
        """
        Compute the discount feature array for a given discount dictionary.

        Args:
            discount_dict : dict
                A dictionary of item to discount value.

        Returns:
            arr : numpy.ndarray
                A vector of the discount feature for all items.
        """
        arr = np.zeros(self.n_items, dtype=np.float32)
        if not discount_dict: return arr
        
        if discount_dict:
            for item, val in discount_dict.items():
                idx = self.item_to_index.get(item)
                if idx is not None:
                    arr[idx] = np.log1p(val)
        if arr.max() > 0:
            arr = self._normalize_array(arr)
        return arr

    def compute_price_array(self, budget):
        """
        Compute the price feature array for a given budget.

        Args:
            budget : float
                The budget for the recommendation.

        Returns:
            arr : numpy.ndarray
                A vector of the price feature for all items.
        """
        arr = np.zeros(self.n_items, dtype=np.float32)
        if budget is None:
            return arr
        prices = self.item_data['Current_Price'].values
        arr = np.log1p(budget / (prices + EPS)).astype(np.float32)
        
        return self._normalize_array(arr)
    
    def compute_description_boost(self, baskets_df):
        """
        Compute the description similarity feature array for a given baskets dataframe.

        Args:
            baskets_df : pd.DataFrame
                A dataframe containing all baskets being processed.

        Returns:
            desc_map : dict
                A dictionary mapping each basket index to its corresponding description similarity feature array.
        """
        if self.similarity_matrix is None:
            raise ValueError("Description similarity not available. Enable include_description at init.")
        
        desc_map = {}
        for idx, row in baskets_df.iterrows():
            basket_items = set(row['StockCode']) if isinstance(row['StockCode'], (list, set)) else set()
            if not basket_items:
                desc_map[idx] = np.zeros(self.n_items, dtype=np.float32)
                continue
            
            # map items to indicies
            basket_idx = [self.item_to_index[item] for item in basket_items if item in self.item_to_index]
            sim_vec = self.similarity_matrix[basket_idx].toarray()  # shape: len(basket) x n_items
            avg_vec = sim_vec.mean(axis=0)
            max_val = avg_vec.max()
            desc_map[idx] = avg_vec / (max_val + EPS) if max_val>0 else np.zeros(self.n_items, dtype=np.float32)
        return desc_map
    
    
    
    # ===  Helper: Compute all features in batch form   ===
    def _compute_features_batch(self, baskets_df, d_effect=None, budget_for_batch=None, discount_for_batch=None):
        """
        Vectorized feature assembler that returns dict with arrays shape (batch_size, n_items)
        Keys: 'H','R','T','P','D'
        Safe with empty baskets. Uses compute_* methods above.
        """
        if d_effect is None:
            d_effect = float(self.d_effect)

        n = len(baskets_df)
        H = np.zeros((n, self.n_items), dtype=np.float32)
        R = np.zeros((n, self.n_items), dtype=np.float32)
        T = np.ones((n, self.n_items), dtype=np.float32)
        P = np.zeros((n, self.n_items), dtype=np.float32)
        D = np.zeros((n, self.n_items), dtype=np.float32)

        # iterate rows (we keep one loop over batch only)
        for i, row in enumerate(baskets_df.itertuples(index=False)):
            # robust extraction of items (you use 'StockCode' or 'X' in different places)
            basket_items = []
            if hasattr(row, 'StockCode') and isinstance(row.StockCode, (list, set)):
                basket_items = row.StockCode
            elif hasattr(row, 'X') and isinstance(row.X, (list, set)):
                basket_items = row.X

            if not basket_items:
                # leave H=0,R=0,P=0,D=0 and T=1 (no recency boost)
                T[i, :] = 1.0
                continue

            cid = getattr(row, 'Customer ID', -1)
            H[i, :] = self.compute_history(cid)
            R[i, :] = self.compute_rules_array(basket_items)
            T[i, :] = self.compute_recency(cid, d_effect=d_effect)

            # --- Budget handling ---
            budget_val = None
            if isinstance(budget_for_batch, (int, float)):
                budget_val = budget_for_batch
            elif isinstance(budget_for_batch, (list, np.ndarray, pd.Series)) and len(budget_for_batch) > i:
                budget_val = budget_for_batch[i]
            elif 'pseudo_budget' in baskets_df.columns:
                budget_val = baskets_df.iloc[i]['pseudo_budget']

            if budget_val is not None and not np.isnan(budget_val):
                P[i, :] = self.compute_price_array(budget_val)

            # --- Discount handling ---
            if isinstance(discount_for_batch, dict):
                D[i, :] = self.compute_discount_array(discount_for_batch)

        return {'H': self._normalize_array(H), 'R': self._normalize_array(R), 'T': self._normalize_array(T), 'P': self._normalize_array(P), 'D': self._normalize_array(D)}

    # ------------------ Recommendation ------------------
    def recommend(self, baskets, Coefficients=None, budget_column=None, discount_dict=None, batch_size=None, top_n=5, verbose=True,
            reg_lambda=0.0, clip_value=None, early_stopping=False):
        """
        Produce top-N recommended items for each basket in `baskets` DataFrame.
        Returns a dict: basket_index -> DataFrame with columns [StockCode, Description, Current_Price, Probability, Rank]
        
        Args:
        - baskets: DataFrame with columns at least ['Customer ID', 'StockCode'] OR ['Customer ID', 'X'] (X is list of codes)
        - Coefficients: dict of self coefficients
        - budget_dict: either a single numeric budget or dict mapping customer/basket to budget (if dict, this function pulls budget per basket)
        - discount_dict: dict of discounts for the batch (applies to all baskets processed)
        - top_n: number of items to recommend
        
        """
        if Coefficients is not None:
            self.weights = Coefficients
        
        # Handle single basket case
        single_input = False
        if isinstance(baskets, dict):
            baskets = pd.DataFrame([baskets])
            single_input = True
        elif isinstance(baskets, pd.Series):
            baskets = pd.DataFrame([baskets])
            single_input = True
        elif isinstance(baskets, pd.DataFrame) and baskets.shape[0] == 1:
            single_input = True
        
        
        if batch_size is not None and (not single_input) and getattr(self, '_allow_sampling', True):
            baskets = baskets.sample(batch_size)
        
        # === Vectorized feature computation ===
        if budget_column is not None:
            budget_dict = baskets[budget_column].to_dict()
        else:
            budget_dict = None
        
        
        feats = self._compute_features_batch(baskets, d_effect=self.d_effect,
            budget_for_batch=budget_dict, discount_for_batch=discount_dict)
        
        H, R, T, P, D = feats['H'], feats['R'], feats['T'], feats['P'], feats['D']
        
        " If top_n == -1, then return all items "
        # Base logit computation
        logit = (self.Bias + self.weights['alpha']*H + self.weights['beta']*R + self.weights['eta']*D + self.weights['delta']*P + self.weights['gamma']*T)
        
        # precompute shared arrays
        if self.include_description:
            Desc_map = self.compute_description_boost(baskets)
            Desc_matrix = np.zeros((len(baskets), self.n_items), dtype=np.float32)
            for i, idx in enumerate(baskets.index):
                Desc_matrix[i,:] = Desc_map.get(idx, np.zeros(self.n_items, dtype=np.float32))
            logit += self.weights['epsilon'] * np.log1p(Desc_matrix)


        # Softmax probability
        logit_max = logit.max(axis=1, keepdims=True)
        exps = np.exp(logit - logit_max)
        prob_matrix = exps / (exps.sum() + EPS)
        
        if self.iteration_bar:
            looper = tqdm.tqdm(enumerate(baskets.index), total=baskets.shape[0])
        else:
            looper = enumerate(baskets.index)
            
        # prepare recommendations
        results = {}
        for i,idx in looper:
            prob = prob_matrix[i]

            # Top-N items
            if top_n == -1:
                top_idx = np.argsort(prob)[::-1]
            else:
                top_idx = np.argpartition(prob, -top_n)[-top_n:]
                top_idx = top_idx[np.argsort(prob[top_idx])[::-1]]
            
            top_codes = self.all_items[top_idx]
            top_probs = prob[top_idx]

            df_top = self.item_data[self.item_data['StockCode'].isin(top_codes)].copy()
            df_top = df_top.set_index('StockCode').loc[top_codes].reset_index()  
            df_top['Probability'] = top_probs
            df_top['Rank'] = np.arange(1, len(df_top) + 1)
            
            results[idx] = df_top

        if single_input:
            return results[0]
        return results
    
    # ----- Analytical Model training -----
    def train_model(self, Carts, lr=0.05, n_iter=20, error=1e-6, sample_size=None,
            batch_size=32, top_n=50, budget_column=None, verbose=True,
            Learning_rate_decay=0, reg_lambda=0.0, clip_value=None, early_stopping=False, patience=5):
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
        # save learning rate decay param as before
        self.lr_decay = Learning_rate_decay
        
        self.weights = self.weights.copy()
        self.weights['d_effect'] = self.d_effect
        history = []
        
        # Early stopping setup
        best_val_loss = np.inf
        patience_counter = 0
        
        self.best_result = 0

        def map_targets_to_indices(Y_batch):
            return np.array([self.item_to_index.get(y, -1) for y in Y_batch], dtype=int)

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
            # weights does not have d_effect
            grads['d_effect'] = 0.0 # make sure to include it
            
            if self.include_description:
                all_desc = self.compute_description_boost(Carts_epoch)

            loop = range(n_batches) if not verbose else tqdm.tqdm(range(n_batches), desc=f"Epoch {epoch}/{n_iter}")
            for b in loop:
                batch = Carts_epoch.iloc[b * batch_size:(b + 1) * batch_size]
                if batch.empty:
                    continue
                
                # extract budget if applied
                budget_for_batch = None
                if budget_column and budget_column in batch.columns:
                    budget_for_batch = batch[budget_column].values

                X_batch = batch.drop(columns=['Y'])
                feats = self._compute_features_batch(X_batch)
                Y_batch = batch['Y'].values
                Y_idx = map_targets_to_indices(Y_batch)
                valid_mask = (Y_idx != -1)

                # --- Forward ---
                logit = (self.Bias
                         + self.weights['alpha'] * feats['H']
                         + self.weights['beta'] * feats['R']
                         + self.weights['gamma'] * feats['T']
                         + self.weights['delta'] * feats['P']
                         + self.weights['eta'] * feats['D']).astype(np.float32)

                if self.include_description and self.weights['epsilon'] != 0.0:
                    Desc_mat = np.vstack([all_desc[i] for i in X_batch.index]) # shape [batch_size, n_items]
                    # transformation in your self: epsilon * log(1 + C)
                    Desc_log = np.log1p(Desc_mat + error)
                    logit += self.weights['epsilon'] * Desc_log

                logit = logit.astype(np.float32)
                logit -= logit.max(axis=1, keepdims=True)
                exps = np.exp(logit)
                probs = exps / (exps.sum(axis=1, keepdims=True) + EPS) # [batch_size, n_items]

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
                grads['d_effect']  = grads.get('d_effect', 0.0) + np.sum(diff * (self.weights['gamma']* E_mat))

            
            # --- L2 Regularization ---
            if reg_lambda > 0:
                l2_term = sum(v**2 for v in self.weights.values())
                total_loss += reg_lambda * l2_term # adding l2 term to loss

                # Update gradient to include L2 term
                for k in self.weights.keys():
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
                if k == 'd_effect':
                    self.d_effect -= effective_lr * grads[k]
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
                    self.d_effect = self.best_params.get('d_effect', self.d_effect)
                    self.weights = {k: v for k, v in self.best_params.items() if k != 'd_effect'}
                    break
            
            if correct_total > self.best_result:
                self.best_result = correct_total
                self.best_params = copy.deepcopy(self.weights)
            

        final_weights = copy.deepcopy(self.weights)
        return final_weights, history
    
    
    def Evaluate_model(self, test_baskets, top_n=100, batch_size=128, budget_column='pseudo_budget', error=1e-8):
        """ Evaluate our recommednation self on unseen baskets
        Computes metrics: 
            - Precision, Recall, F1,
            - Hits@N (fraction where true item appears in top-N recommendations)

        Parameters
        ----------
        self : Retail_Recommendation
        test_baskets : pd.DataFrame
            Test baskets dataframe
        top_n : int
            Number of top items to consider from recommendations
        batch_size : int
            Number of baskets to process in one batch
        budget_column : str
            Column name for budget
        error: float
            minimum error associate with each quantity for failsafe

        Returns a dict of aggregated scores.
        """
        if test_baskets.empty: 
            raise ValueError("Test baskets dataframe is empty.")
        test_baskets = test_baskets.reset_index(drop=True)
        
        # Handle small batch issue safely
        if batch_size is not None:
            batch_size = min(batch_size, len(test_baskets))

        # Run self inference
        results = self.recommend(test_baskets, top_n=top_n, batch_size=batch_size, budget_column=budget_column)

        hits = 0
        total_true = len(test_baskets)
        precisions, recalls, f1s = [], [], []
        reciprocal_ranks, ndcgs, average_precisions = [], [], []

        for idx, row in test_baskets.iterrows():
            true_item = row['Y']
            if idx not in results: # No prediction for this basket
                continue
            
            Result = results[idx] # predicted items
            if 'Rank' in Result.columns:
                ranks_item = Result.sort_values(by='Rank')['StockCode'].tolist()[:top_n]
            else:
                ranks_item = Result.sort_values(by='Probability', ascending=False)['StockCode'].tolist()[:top_n]
            
            if not ranks_item: continue

            # --- Hit@N ---
            hits_i = 1 if (true_item in ranks_item) else 0
            hits += hits_i
            
            precision_i = hits_i / len(ranks_item)
            recall_i = hits_i
            f1_i = (2 * precision_i * recall_i) / (precision_i + recall_i + error)
            
            precisions.append(precision_i)
            recalls.append(recall_i)
            f1s.append(f1_i)

            # --- Rank-sensitive metrics ---
            if hits_i:
                rank = ranks_item.index(true_item) + 1

                # MRR: Reciprocal rank of true item
                reciprocal_ranks.append(1.0 / rank)

                # NDCG: discounted gain normalized by ideal DCG (1.0)
                dcg = 1.0 / math.log2(rank + 1)
                ndcgs.append(dcg)

                # Average Precision: since 1 true item, AP = precision at that rank
                ap = 1.0 / rank
                average_precisions.append(ap)
            else:
                reciprocal_ranks.append(0.0)
                ndcgs.append(0.0)
                average_precisions.append(0.0)

        metrics = {
            'Precision@N': np.mean(precisions) if precisions else 0.0,
            'Recall@N': np.mean(recalls) if recalls else 0.0,
            'F1@N': np.mean(f1s) if f1s else 0.0,
            'HitRate': hits / max(1, total_true),
            'MRR@N': np.mean(reciprocal_ranks) if reciprocal_ranks else 0.0,
            'NDCG@N': np.mean(ndcgs) if ndcgs else 0.0,
            'MAP@N': np.mean(average_precisions) if average_precisions else 0.0,
            "Correct recommendations": hits,
            'Fraction correct as ratio': f"{hits}/{total_true}",
        }

        self.eval_metrics = metrics
        return metrics
    
    # ================================
# DATA LOADING PLACEHOLDER (for notebook usage only)
# ================================
# In production (like Final_model.py), data will be loaded externally
# and passed as arguments when instantiating Retail_Recommendation.
# So we comment out dataset loading to avoid FileNotFound errors during import.

# Example manual loading (only runs if you execute Model.py directly)
if __name__ == "__main__":
    print("Running Model.py directly — loading datasets for standalone testing...")
    df = pd.read_parquet('../Data/data_with_features.parquet')
    customer = pd.read_pickle('../Data/customer_history.pkl')
    items = pd.read_pickle('../Data/item_summary.pkl')
    baskets = pd.read_pickle('../Data/baskets.pkl')
    rules = pd.read_pickle('../Data/rules.pkl')
    vectorizer = joblib.load('../Models/vectorizer.joblib')
else:
    df = None
    customer = None
    items = None
    baskets = None
    rules = None
    vectorizer = None

# ================================
# CONSTANTS AND GLOBALS
# ================================
initial_weights = {'alpha': 1.0, 'beta': 1.0, 'delta': 1.0, 'eta': 1.0, 
                   'gamma': 1.0, 'epsilon': 1.0, 'd_effect': 0.5}
d_effect = 0.5  # decay factor
Today = pd.Timestamp.now()  # placeholder (will be overridden by external caller)
Time_period = 200  # placeholder (computed later from data)

EPS = 1e-9  # small constant to avoid log(0) / division by zero
DEFAULT_HALF_LIFE_DAYS = 7
MAGIC_KMIN_PERIOD = 30

    



