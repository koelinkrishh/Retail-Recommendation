# Final file to work out a final pipeline to train entire model from base.

import os
import sys

import gc
import copy
import math
import pickle
from typing import Optional, Dict, Any

import optuna
import joblib
import traceback

import numpy as np
import pandas as pd

from src.logger import logging
from src.exception import CustomException
from src.Components.data_ingestion import load_data
from src.config import *

# Import your training model class (Train.py should expose it)
from src.Components.train import Retail_recommendation_model  # adjust import path if different
from src.Components.evaluate import Evaluation

EPS = 1e-9


class OptunaTrainer:
    """
    Manage Optuna hyperparameter search for Retail recommendation model.

    Requires:
      - model_template: an instantiated Retail_recommendation_model (with data loaded).
        The template will be used only to read constructor args (item/customer/rules data).
      - train_df, val_df: optionally provided dataframes (if None, will be loaded from config paths)
    """
    def __init__(self, model_template, resample=False):
        try:
            self.model_template = model_template
            self.Weights = getattr(model_template, "Weights", None)
            
            if resample:
                logging.info("Resampling baskets from scratch")
                baskets = load_data(Baskets_data_PATH, form='pickle')
                self.train_data, self.test_data, self.val_data = Retail_recommendation_model().Basket_splitter(baskets, save=False, split_type='time')
            else:
                logging.info("Loading splitted baskets data")
                self.train_data = load_data(TRAIN_FILE, form='parquet')
                self.val_data = load_data(VALID_FILE, form='parquet')
                self.test_data = load_data(TEST_FILE, form='parquet')
        except Exception as e:
            logging.error("Error Initializing Optuna Trainer")
            raise CustomException(e, sys)
    
    
    # Utility: build a fresh model with specific initial weights
    @staticmethod
    def make_model_from_template(template_model, Weights):
        gc.collect()
        # We instantiate a fresh model using the same high-level args (data loaded inside constructor)
        # Template properties may include include_description, d_effect, Half_life etc.
        try:
            new_model = Retail_recommendation_model(
                initial_weights = Weights,
                d_effect=getattr(template_model, "d_effect", 0.5),
                current_date=getattr(template_model, "current_date", None),
                Half_life=getattr(template_model, "Half_life", 7),
                Time_period=getattr(template_model, "Time_period", 100),
                filter_items=False,
                include_description=getattr(template_model, "include_description", False),
                iteration_bar=False,
            )
            return new_model
        except Exception as e:
            logging.error("Failed to instantiate model from template")
            raise CustomException(e, sys)
        
    """ Optuna Experimental studies: """
    ## 1_ Defining objective function
    def objective(self, trial, model_template, Weights, train, val, 
        top_n=100, n_iter=10, batch_size=128, patience=5):
        """
        Objective function for Optuna.
        Trains a temporary model and evaluates it on validation data.
        Returns scalar score to maximize (weighted HitRate + NDCG).
        """
        gc.collect()
        
        lr = trial.suggest_float("lr", 0.01, 0.3, log=True)
        reg_lambda = trial.suggest_float("reg_lambda", 0.0, 0.5)
        lr_decay = trial.suggest_float("lr_decay", 0.0, 0.4)
        clip_value = trial.suggest_float("clip_value", 0.1, 5.0)
        
        temp_model = None
        try:
            temp_model = self.make_model_from_template(model_template, Weights)
            # Train for a few iterations to get a comparative score
            weights, history = temp_model.train_model(
                Carts=train.sample(len(train)//2),
                lr=lr,
                n_iter=n_iter,
                batch_size=batch_size,
                top_n=top_n,
                Learning_rate_decay=lr_decay,
                reg_lambda=reg_lambda,
                clip_value=clip_value,
                early_stopping=False,
                verbose=False,
            )
            # Evaluate -> use seperate evaluation component
            evaluator = Evaluation(temp_model)
            metrics = evaluator.evaluate(test_data=val, batch_size=batch_size, top_n=top_n, verbose=False)
            score = (0.7 * metrics["HitRate"] + 0.3 * metrics["NDCG@N"]) if metrics["NDCG@N"] else metrics["HitRate"]
            
            # Store trial info
            trial.set_user_attr("metrics", metrics)
            trial.set_user_attr("weights", weights)
            trial.set_user_attr("score", score)
            
            return score
        except Exception as e:
            logging.error(f"Trial {trial.number} failed with exception:")
            traceback.print_exc()
            trial.set_user_attr("error", str(e))
            return 0.0
        finally:
            if temp_model:
                del temp_model
            gc.collect()
            
    ## 2_ Run the optuna study
    def run_optuna_search(self, n_trials=10, n_iter=10, top_n=100, batch_size=128, patience=5):
        """
        Run Optuna optimization loop using the given train/validation split.
        """
        if self.train_data is None or self.val_data is None:
            raise CustomException("Train or Validation data not available.", sys)
        
        study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler())
        logging.info(f"Running Optuna study with {n_trials} trails")
        study.optimize(
            lambda trial: self.objective(trial, self.model_template, self.Weights, self.train_data, self.val_data,
                top_n=top_n, n_iter=n_iter, batch_size=batch_size, patience=patience),
            n_trials=n_trials, show_progress_bar=True
        )
        
        self.study = study
        logging.info("Optuna study completed")
        
        return study
    
    ## 3_ Utility functions
    def save_study(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self.study, f)
        logging.info(f"Optuna study saved to {path}")
    
    def load_study(self, path):
        with open(path, "rb") as f:
            self.study = pickle.load(f)
        
        logging.info(f"Optuna study loaded from {path}")
        return self.study
    
    def get_best_results(self, study=None):
        study = study or getattr(self, "study", None)
        if study is None:
            raise CustomException("No study found to extract best results from.", sys)
        
        best_trial = study.best_trial
        best_params = best_trial.params
        best_weights = best_trial.user_attrs.get("weights", {})
        best_metrics = best_trial.user_attrs.get("metrics", {})
        
        return best_weights, best_params, best_metrics

    def build_best_model(self, best_weights):
        best_model = self.make_model_from_template(self.model_template, best_weights)
        logging.info("Built model with best Optuna weights.")
        return best_model
    
    ## 4_ Running the entire function:
    def run_pipeline(self, n_trials=50, n_iter=10, top_n=100, batch_size=128, patience=5,):
        """
        Runs the complete Optuna optimization pipeline end-to-end.

        Steps:
            1. Run Optuna search
            2. Extract best weights, params, and metrics
            3. Build model with best weights
            4. Return all three in a structured format
        """
        logging.info("🚀 Starting full Optuna optimization pipeline...")

        # 1️⃣ Run Optuna optimization
        study = self.run_optuna_search(
            n_trials=n_trials,
            n_iter=n_iter,
            top_n=top_n,
            batch_size=batch_size,
        )

        # 2️⃣ Extract best results
        best_weights, best_params, best_metrics = self.get_best_results(study)

        # 3️⃣ Build best model
        best_model = self.build_best_model(best_weights)

        # 4️⃣ Save the study (optional)
        study_path = os.path.join(Exp_dir, "optuna_study.pkl")
        self.save_study(study_path)

        logging.info(f"✅ Optuna pipeline finished. Study saved at {study_path}")
        logging.info(f"Best Params: {best_params}")
        logging.info(f"Best Metrics: {best_metrics}")

        return best_model, best_weights, best_params, best_metrics
    

if __name__ == "__main__":
    logging.info("Running OptunaTrainer → Run_pipeline() ...")
    try:
        # Create base model template
        base_model = Retail_recommendation_model(include_description=True)

        # Initialize trainer with datasets
        trainer = OptunaTrainer(model_template=base_model)

        # Run the full Optuna pipeline
        best_model, best_weights, best_params, best_metrics = trainer.run_pipeline(
            n_trials=5,      # Try 10 trials for real tuning
            n_iter=5,        # Each trial runs for 15 iterations
            top_n=100,
            batch_size=128,
        )

        # ✅ Save the best model
        model_path = os.path.join(Exp_dir, "Final_best_model.pkl")
        with open(model_path, "wb") as f:
            pickle.dump(best_model, f)
            
        logging.info(f"🏁 Best model saved at {model_path}")

        print("\n=== OPTUNA RESULTS SUMMARY ===")
        print("Best Weights:", best_weights)
        print("Best Params:", best_params)
        print("Best Metrics:", best_metrics)
        print(f"Saved Model: {model_path}")

        ## Saving best metrics
        metrics_path = os.path.join(Exp_dir, "best_metrics.pkl")
        with open(metrics_path, "wb") as f:
            pickle.dump(best_metrics, f)
        ## Saving best params
        params_path = os.path.join(Exp_dir, "best_params.pkl")
        with open(params_path, "wb") as f:
            pickle.dump(best_params, f)


        logging.info("OptunaTrainer pipeline completed.")

    except Exception as e:
        logging.error("OptunaTrainer pipeline failed.")
        raise CustomException(e, sys)


    