# Retail-Recommendation
#### 🧠 Overview: 
This project implements a Retail Product Recommendation Model designed to predict what items a customer is most likely to buy next, based on historical purchase data and item-level metadata.

It’s a hybrid recommender system combining:
- Collaborative features (purchase history, recency)
- Association rule-based reasoning
- Content-based item similarity (TF-IDF on item descriptions)
- Price & discount awareness
- Temporal decay modeling

The model is interpretable, modular, and trainable through a custom gradient optimization routine.

### ✨ Key Features

- 🧩 Hybrid Recommendation Engine — Combines rules, content, and behavioral data.
- 🕒 Temporal Decay & Recency Bias — Uses exponential half-life modeling to weight recent purchases higher.
- 🛍 Customer-Level Personalization — History, recency, and purchase frequency per customer.
- 💰 Price & Discount Aware — Adjusts predictions dynamically using pricing signals.
- 🔤 TF-IDF Description Embeddings — Captures semantic similarity between product descriptions.

⚙️ Trainable via Gradient Descent — Fully trainable weighting system across multiple features.

> 🧾 Supports Multi-File Inputs — Modular data ingestion for preprocessed parquet/pickle datasets.


### Project Structure
``` plaintext
Retail-Recommendation/
│
├── src/
│   ├── Components/                   (contains submodules)
│       ├── data_ingestion.py             ✅ load raw dataset from multiple sources
│       ├── data_processing.py            ✅ clean, sanitize, and preprocess raw data
│       ├── feature_eng.py                ✅ create structured datasets and model features
│       ├── train.py                      ⚪ (Recommendation model class with training implementation)
│       ├── evaluate.py                   ⚪ (evaluation + metrics — next step)
│       ├── optimization.py               ⚪ (hyperparameter optimization — next step)
│       ├── 
│   ├── logger.py                     ✅ central logging setup
│   ├── exception.py                  ✅ custom exception class
│   ├── config.py                     ✅ (stores paths, constants, hyperparameters)
│   └── utils.py                      ⚪ (helper utilities like plotting, validation)
│
├── data/
│   ├── raw/                          (Excel, CSV, Pickle, Parquet)
│   ├── processed/                    (cleaned + combined parquet)
│   ├── intermediate/                 (feature datasets)
│   └── models/                       (trained model weights)
│
├── notebooks/                        (EDA, experimentation)
├── requirements.txt
├── README.md
└── app/                              (will hold FastAPI or Streamlit app later)
```

### 🔄 Data Pipeline

Input Datasets:
1. Processed Transactions (Processed_PATH)
- Cleaned transactional data with columns like Customer_ID, StockCode, Quantity, Purchase_Date, etc.
2. Customer Data (Customer_data_PATH)
- Per-customer purchase summary dictionaries (quantities, counts, recency).
3. Item Data (Item_data_PATH)
- Product-level attributes: frequency, price, description, discount, etc.
4. Association Rules (Rules_data_PATH)
- Apriori/FP-Growth–derived rules (antecedent → consequent).
5. Baskets (Baskets_data_PATH)
- Pre-grouped customer baskets for model training.


###  Training workflow: 
File: `train.py`
1. Load train/test data
2. Initialize model
``` python
rec = Retail_recommendation_model(include_description=True, d_effect=1.0)
```
3. Train model
``` python
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
```
4. Evaluate performance:
``` python
evaluator = Evaluation(rec)
metrics = evaluator.evaluate(test_data, top_n=100)
print(metrics)
```

### 📊 Evaluation Pipeline
File: `src/Components/evaluate.py`
Implements class Evaluation, which:
- Calls model.recommend() on unseen baskets
- Computes both hit-based and rank-aware metrics

##### Metrics computed:
| Metric      | Description                                              |
| ----------- | -------------------------------------------------------- |
| Precision@N | Fraction of top-N items that were correct                |
| Recall@N    | True items retrieved                                     |
| F1@N        | Harmonic mean of Precision and Recall                    |
| HitRate     | Fraction of baskets with at least one correct prediction |
| MRR         | Mean Reciprocal Rank                                     |
| NDCG        | Rank discount-based metric                               |
| MAP         | Mean Average Precision                                   |

### 🧪 Final Model Training
File: `final_training.py`

This script:
- Loads optimal weights and parameters
- Reinitializes model with best settings
- Retrains for more iterations (25 by default)
- Evaluates final performance
- Saves new weights and plots training history

##### Example workflow:
``` python
weights_path = os.path.join(Exp_dir, "best_weights.pkl")
params_path = os.path.join(Exp_dir, "best_params.pkl")

best_weights = load_data(weights_path, form='pickle')
best_params = load_data(params_path, form='pickle')

model = Retail_recommendation_model(
    initial_weights=best_weights,
    d_effect=best_weights['d_effect'],
    Half_life=7,
    Time_period=100,
    include_description=True,
)

weights, history = model.train_model(
    Carts=train,
    lr=best_params['lr'],
    n_iter=25,
    batch_size=256,
    reg_lambda=best_params['reg_lambda'],
    clip_value=best_params['clip_value'],
    early_stopping=True,
    patience=5,
)
```

### 🧩 Usage Instructions

1. Install dependencies
``` 
pip install -r requirements.txt 
```
2. Run initial training
```
python train_model.py 
```
3. Run final model training
```
python final_training.py 
```
4. Evaluate and plot results automatically.

## 🎯 Generating Recommendations
``` python
# Single customer basket
basket = {
    "Customer_ID": 17850,
    "StockCode": ["85099B", "20725", "22727"]
}

recommendations = model.recommend(basket, top_n=10)
print(recommendations[['StockCode', 'Description', 'Probability', 'Rank']])
```
Output includes:
- Ranked items
- Purchase probability scores
- Product metadata




