# Retial-Recommendation
Complete Recommendation system project


### Project Structure

Retail-Recommendation/
│
├── src/
│   ├── logger.py                     ✅ central logging setup
│   ├── exception.py                  ✅ custom exception class
│   ├── data_ingestion.py             ✅ load raw dataset from multiple sources
│   ├── data_processing.py            ✅ clean, sanitize, and preprocess raw data
│   ├── feature_engineering.py        ✅ create structured datasets and model features
│   ├── model.py                      🟡 (your Retail_Recommendation class)
│   ├── train.py                      ⚪ (training orchestration — next step)
│   ├── evaluate.py                   ⚪ (evaluation + metrics — next step)
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


Final Model weights:
{'alpha': np.float32(2.1774216), 'beta': np.float32(3.9744666), 'gamma': np.float32(1.5444027), 'delta': np.float32(-0.4433973), 'eta': np.float32(0.53207666), 'epsilon': np.float32(0.35471776), 'd_effect': np.float32(1.9278386)}
Validation metrics:
{'Precision@N': np.float64(0.007244147157190636), 'Recall@N': np.float64(0.7244147157190636), 'F1@N': np.float64(0.014344845713794867), 'HitRate': 0.7244147157190636, 'MRR@N': np.float64(0.09932481255115745), 'NDCG@N': np.float64(0.21643825650416038), 'MAP@N': np.float64(0.09932481255115745), 'Correct_predictions': 2166, 'Total_baskets': 2990}
