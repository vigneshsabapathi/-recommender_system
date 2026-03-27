"""Experiment 5: Head-to-head model comparison at multiple K values."""
import time
import sys, os

# Force unbuffered output so progress is visible in background runs
os.environ["PYTHONUNBUFFERED"] = "1"

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.experiment_utils import *


def log(msg=""):
    print(msg, flush=True)


def run_experiment():
    mlflow = init_mlflow("model-comparison")

    log("Loading data...")
    test_df = load_test_data()
    train_df = load_train_data()

    log("Loading models...")
    models = load_models()

    # Add popularity baseline
    models["popularity"] = PopularityBaseline(train_df)

    k_values = [5, 10, 20]

    # Collect all results for the summary table
    all_results = []

    for model_name, model in models.items():
        for k in k_values:
            run_name = f"{model_name}_k{k}"
            log(f"\n{'='*60}")
            log(f"Evaluating: {run_name}")
            log(f"{'='*60}")

            with mlflow.start_run(run_name=run_name):
                mlflow.set_tags({
                    "experiment_type": "model_comparison",
                    "model": model_name,
                    "eval_k": k,
                })
                mlflow.log_params({
                    "model_type": model_name,
                    "eval_k": k,
                    "max_test_users": 500,
                    "relevance_threshold": 4.0,
                    "seed": 42,
                })

                start = time.time()
                metrics = evaluate_model_on_users(
                    model, test_df, train_df,
                    k=k, max_users=500, relevance_threshold=4.0, seed=42
                )
                eval_time = time.time() - start
                metrics["eval_time_s"] = eval_time

                mlflow.log_metrics(metrics)

                log(f"  RMSE: {metrics.get('rmse', 'N/A')}")
                log(f"  NDCG@{k}: {metrics.get(f'ndcg_at_{k}', 'N/A')}")
                log(f"  P@{k}: {metrics.get(f'precision_at_{k}', 'N/A')}")
                log(f"  Coverage: {metrics.get('catalog_coverage', 'N/A')}")
                log(f"  Time: {eval_time:.1f}s")

                all_results.append({
                    "model": model_name,
                    "k": k,
                    "rmse": metrics.get("rmse"),
                    f"ndcg": metrics.get(f"ndcg_at_{k}"),
                    f"precision": metrics.get(f"precision_at_{k}"),
                    f"recall": metrics.get(f"recall_at_{k}"),
                    f"map": metrics.get(f"map_at_{k}"),
                    "coverage": metrics.get("catalog_coverage"),
                    "users_eval": metrics.get("users_evaluated"),
                    "time_s": eval_time,
                })

    # Print summary table
    log("\n")
    log("=" * 110)
    log("SUMMARY TABLE: Model Comparison Results")
    log("=" * 110)

    header = (
        f"{'Model':<16} {'K':>3}  {'RMSE':>8}  {'NDCG@K':>8}  {'P@K':>8}  "
        f"{'R@K':>8}  {'MAP@K':>8}  {'Coverage':>8}  {'Users':>5}  {'Time(s)':>7}"
    )
    log(header)
    log("-" * 110)

    for r in all_results:
        def fmt(v, digits=4):
            return f"{v:.{digits}f}" if v is not None else "N/A"

        line = (
            f"{r['model']:<16} {r['k']:>3}  {fmt(r['rmse']):>8}  "
            f"{fmt(r['ndcg']):>8}  {fmt(r['precision']):>8}  "
            f"{fmt(r['recall']):>8}  {fmt(r['map']):>8}  "
            f"{fmt(r['coverage']):>8}  {str(r.get('users_eval', 'N/A')):>5}  "
            f"{r['time_s']:>7.1f}"
        )
        log(line)

    log("=" * 110)

    # Best model per K for key metrics
    log("\nBest models per K value:")
    for k in k_values:
        k_results = [r for r in all_results if r["k"] == k]
        if not k_results:
            continue
        # Best NDCG
        valid_ndcg = [r for r in k_results if r["ndcg"] is not None]
        if valid_ndcg:
            best_ndcg = max(valid_ndcg, key=lambda r: r["ndcg"])
            log(f"  K={k:>2}  Best NDCG: {best_ndcg['model']:<16} ({best_ndcg['ndcg']:.4f})")
        # Best RMSE (lower is better)
        valid_rmse = [r for r in k_results if r["rmse"] is not None]
        if valid_rmse:
            best_rmse = min(valid_rmse, key=lambda r: r["rmse"])
            log(f"  K={k:>2}  Best RMSE: {best_rmse['model']:<16} ({best_rmse['rmse']:.4f})")

    log("\n\nDone! View results at:")
    log("https://dagshub.com/vigneshsabapathi/recommender_system.mlflow")


if __name__ == "__main__":
    run_experiment()
