"""Experiment 4: Hybrid Weight Optimization - sweep cf/content/als weights."""
import sys, os, time, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.experiment_utils import *

def run_experiment():
    mlflow = init_mlflow("hybrid-weight-optimization")

    print("Loading data and models...")
    test_df = load_test_data()
    train_df = load_train_data()
    models = load_models()

    # Need the individual sub-models
    cf_model = models.get("collaborative")
    content_model = models.get("content_based")
    als_model = models.get("als")

    from src.models.hybrid import WeightedHybridRecommender

    # Sweep weights in 0.1 increments where all >= 0 and sum = 1.0
    weight_configs = []
    for cf_w in np.arange(0.0, 0.6, 0.1):
        for content_w in np.arange(0.0, 0.6, 0.1):
            als_w = round(1.0 - cf_w - content_w, 2)
            if als_w >= 0 and als_w <= 1.0:
                weight_configs.append({
                    "cf_weight": round(cf_w, 2),
                    "content_weight": round(content_w, 2),
                    "als_weight": round(als_w, 2),
                })

    print(f"Running {len(weight_configs)} weight configurations...")

    best_ndcg = 0
    best_config = None
    results = []

    for i, weights in enumerate(weight_configs):
        run_name = f"cf{weights['cf_weight']}_cnt{weights['content_weight']}_als{weights['als_weight']}"
        print(f"\n[{i+1}/{len(weight_configs)}] {run_name}")

        hybrid = WeightedHybridRecommender(
            cf_model=cf_model,
            content_model=content_model,
            als_model=als_model,
            weights=weights,
        )

        with mlflow.start_run(run_name=run_name):
            mlflow.set_tags({"experiment_type": "hybrid_weight_optimization"})
            mlflow.log_params(weights)
            mlflow.log_param("eval_k", 10)

            start = time.time()
            metrics = evaluate_model_on_users(
                hybrid, test_df, train_df,
                k=10, max_users=300, seed=42
            )
            metrics["eval_time_s"] = time.time() - start
            mlflow.log_metrics(metrics)

            ndcg = metrics.get("ndcg_at_10", 0)
            print(f"  NDCG@10: {ndcg:.4f}, RMSE: {metrics.get('rmse', 'N/A')}, P@10: {metrics.get('precision_at_10', 'N/A')}")

            if ndcg > best_ndcg:
                best_ndcg = ndcg
                best_config = weights

            results.append({**weights, **metrics})

    # Print summary
    print(f"\n{'='*60}")
    print(f"BEST CONFIG: {best_config}")
    print(f"BEST NDCG@10: {best_ndcg:.4f}")
    print(f"\nTop 5 configurations by NDCG@10:")
    sorted_results = sorted(results, key=lambda x: x.get("ndcg_at_10", 0), reverse=True)
    for r in sorted_results[:5]:
        print(f"  cf={r['cf_weight']}, cnt={r['content_weight']}, als={r['als_weight']} -> NDCG={r.get('ndcg_at_10',0):.4f}")

    # Save results
    with open("models/hybrid_weight_results.json", "w") as f:
        json.dump({"best_config": best_config, "best_ndcg": best_ndcg, "all_results": sorted_results[:10]}, f, indent=2)

    print(f"\nhttps://dagshub.com/vigneshsabapathi/recommender_system.mlflow")

if __name__ == "__main__":
    run_experiment()
