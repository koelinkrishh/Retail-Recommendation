### Final model training to get best model weights

import os
import sys

import gc
import copy
import math
import pickle
import joblib

import numpy as np
import pandas as pd

from src.logger import logging
from src.exception import CustomException

from src.Components.data_ingestion import load_data
from src.Components.train import Retail_recommendation_model
from src.Components.evaluate import Evaluation
from src.config import *

try:
    logging.info("""Starting Final model training -> with optimal weights as initial weights and
                Optimal hyperparameters.""")

    logging.info("Loading best weights and params")
    weights_path = os.path.join(Exp_dir, "best_weights.pkl")
    best_weights = load_data(weights_path, form='pickle')
    params_path = os.path.join(Exp_dir, "best_params.pkl")
    best_params = load_data(params_path, form='pickle')

    print(best_weights, best_params)

    train = load_data(TRAIN_FILE, form='parquet')
    val_data = load_data(VALID_FILE, form='parquet')

    # Enter initial weights
    model = Retail_recommendation_model(
        initial_weights = best_weights,
        d_effect=best_weights['d_effect'],
        Half_life= 7,
        Time_period=100,
        filter_items=False,
        include_description=True,
        iteration_bar=False,
    )

    logging.info("Starting final model training")

    weights, history = model.train_model(
        Carts=train,
        lr=best_params['lr'],
        n_iter=25,
        batch_size=256,
        top_n=100,
        Learning_rate_decay=best_params['lr_decay'],
        reg_lambda=best_params['reg_lambda'],
        clip_value=best_params['clip_value'],
        early_stopping=True,
        verbose=True,
        patience=5,
    )
    logging.info("Final model training completed, now evaluating its performance on validation data")
    # Evaluate -> use seperate evaluation component
    evaluator = Evaluation(model)
    metrics = evaluator.evaluate(test_data=val_data, batch_size=128, verbose=False)

    print("Final Model weights: ")
    print(weights)
    print("Validation metrics:")
    print(metrics)


    logging.info("Final model evaluate and saved successfully")
    ## Saving final model weights
    # model_path = os.path.join(Exp_dir, "final_model.joblib")
    # joblib.dump(model, model_path)
    """Saved model is way too large. Just make new model with same weights. """
    
    
    final_weights_path = os.path.join(Exp_dir, "final_weights.pkl")
    with open(final_weights_path, "wb") as f:
        pickle.dump(weights, f)

    """ This is the model which will be used for recommendation. """

    logging.info("Plotting history of final model training")
    import matplotlib.pyplot as plt

    plt.figure(figsize=(8, 5))
    plt.plot([h["loss"] for h in history], label="Loss", linewidth=2)
    plt.plot([h["acc"] for h in history], label="Accuracy", linewidth=2)
    plt.title("Final Model Training History")
    plt.xlabel("Epoch")
    plt.ylabel("Metric Value")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.show()

    logging.info("📈 Training curves plotted successfully.")

except Exception as e:
    logging.error("Error during final model training pipeline.")
    raise CustomException(e, sys)


    