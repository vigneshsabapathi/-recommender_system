"""Experiment 2: CF Top-K Similar Items Sweep."""
import sys, os, time, json
import numpy as np
from scipy.sparse import load_npz
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.experiment_utils import *
from pathlib import Path

def run_experiment():
    mlflow = init_mlflow("cf-topk-sweep")

    test_df = load_test_data()
    train_df = load_train_data()

    ROOT = Path(__file__).resolve().parent.parent
    processed = ROOT / "data" / "processed"

    # Load sparse matrix and id maps
    user_item_matrix = load_npz(processed / "user_item_matrix.npz")
    with open(processed / "movie_id_to_idx.json") as f:
        movie_id_to_idx = {int(k): v for k, v in json.load(f).items()}
    with open(processed / "user_id_to_idx.json") as f:
        user_id_to_idx = {int(k): v for k, v in json.load(f).items()}

    k_values = [10, 20, 50, 100, 200]

    from src.models.collaborative import ItemItemCF

    for top_k in k_values:
        run_name = f"cf_topk_{top_k}"
        print(f"\n{'='*50}")
        print(f"Training CF with top_k_similar={top_k}")

        with mlflow.start_run(run_name=run_name):
            mlflow.set_tags({"experiment_type": "cf_topk_sweep"})
            mlflow.log_params({
                "top_k_similar": top_k,
                "similarity_metric": "cosine",
                "eval_k": 10,
                "max_test_users": 300,
            })

            # Train - top_k is a constructor parameter
            cf = ItemItemCF(top_k=top_k)
            start = time.time()
            cf.fit(user_item_matrix, movie_id_to_idx, user_id_to_idx)
            train_time = time.time() - start

            # Log similarity matrix stats
            if hasattr(cf, 'similarity') and cf.similarity is not None:
                mlflow.log_metric("similarity_nnz", cf.similarity.nnz)

            mlflow.log_metric("training_time_s", train_time)
            print(f"  Training time: {train_time:.1f}s")

            # Evaluate
            metrics = evaluate_model_on_users(
                cf, test_df, train_df,
                k=10, max_users=300, seed=42
            )
            mlflow.log_metrics(metrics)

            print(f"  NDCG@10: {metrics.get('ndcg_at_10', 'N/A')}")
            print(f"  P@10: {metrics.get('precision_at_10', 'N/A')}")
            print(f"  Coverage: {metrics.get('catalog_coverage', 'N/A')}")
            print(f"  RMSE: {metrics.get('rmse', 'N/A')}")

    print(f"\nhttps://dagshub.com/vigneshsabapathi/recommender_system.mlflow")

if __name__ == "__main__":
    run_experiment()
