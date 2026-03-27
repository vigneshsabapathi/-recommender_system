"""Experiment 7: Cold-Start Analysis - performance by user activity level."""
import sys, os, time
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.experiment_utils import *

def run_experiment():
    mlflow = init_mlflow("cold-start-analysis")

    test_df = load_test_data()
    train_df = load_train_data()
    models = load_models()

    # Count ratings per user in training set
    user_train_counts = train_df.groupby("userId").size()

    # Define strata
    strata = {
        "light_1_20": user_train_counts[(user_train_counts >= 1) & (user_train_counts <= 20)].index,
        "medium_21_100": user_train_counts[(user_train_counts > 20) & (user_train_counts <= 100)].index,
        "heavy_100plus": user_train_counts[user_train_counts > 100].index,
    }

    print("User strata:")
    for name, users in strata.items():
        print(f"  {name}: {len(users)} users")

    for model_name in ["collaborative", "content_based", "als", "hybrid"]:
        model = models.get(model_name)
        if model is None:
            continue

        for stratum_name, stratum_users in strata.items():
            run_name = f"{model_name}_{stratum_name}"
            print(f"\n[{run_name}]")

            # Filter test_df to only stratum users
            stratum_test = test_df[test_df["userId"].isin(stratum_users)]
            if len(stratum_test) == 0:
                print("  No test data for this stratum, skipping")
                continue

            with mlflow.start_run(run_name=run_name):
                mlflow.set_tags({
                    "experiment_type": "cold_start_analysis",
                    "model": model_name,
                    "stratum": stratum_name,
                })
                mlflow.log_params({
                    "model_type": model_name,
                    "stratum": stratum_name,
                    "stratum_total_users": len(stratum_users),
                    "stratum_test_rows": len(stratum_test),
                    "eval_k": 10,
                    "max_test_users": 200,
                })

                start = time.time()
                metrics = evaluate_model_on_users(
                    model, stratum_test, train_df,
                    k=10, max_users=200, seed=42
                )
                metrics["eval_time_s"] = time.time() - start
                mlflow.log_metrics(metrics)

                print(f"  Users evaluated: {metrics.get('users_evaluated', 0)}")
                print(f"  NDCG@10: {metrics.get('ndcg_at_10', 'N/A'):.4f}" if 'ndcg_at_10' in metrics else "  NDCG@10: N/A")
                print(f"  RMSE: {metrics.get('rmse', 'N/A')}")

    print(f"\nhttps://dagshub.com/vigneshsabapathi/recommender_system.mlflow")

if __name__ == "__main__":
    run_experiment()
