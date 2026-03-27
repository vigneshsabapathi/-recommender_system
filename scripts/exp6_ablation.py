"""Experiment 6: Ablation Study -- remove one component at a time from the hybrid."""
import time
import sys, os

# Force unbuffered output so progress is visible in background runs
os.environ["PYTHONUNBUFFERED"] = "1"

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.experiment_utils import *


def log(msg=""):
    print(msg, flush=True)


# ---------------------------------------------------------------------------
# Ablation configurations
# ---------------------------------------------------------------------------
ABLATION_CONFIGS = [
    {
        "name": "full_hybrid",
        "removed": "none",
        "use_cf": True,
        "use_content": True,
        "use_als": True,
        "weights": {"cf_weight": 0.2, "content_weight": 0.2, "als_weight": 0.6},
    },
    {
        "name": "no_cf",
        "removed": "cf",
        "use_cf": False,
        "use_content": True,
        "use_als": True,
        "weights": {"cf_weight": 0.0, "content_weight": 0.25, "als_weight": 0.75},
    },
    {
        "name": "no_content",
        "removed": "content",
        "use_cf": True,
        "use_content": False,
        "use_als": True,
        "weights": {"cf_weight": 0.25, "content_weight": 0.0, "als_weight": 0.75},
    },
    {
        "name": "no_als",
        "removed": "als",
        "use_cf": True,
        "use_content": True,
        "use_als": False,
        "weights": {"cf_weight": 0.5, "content_weight": 0.5, "als_weight": 0.0},
    },
    {
        "name": "only_cf",
        "removed": "content+als",
        "use_cf": True,
        "use_content": False,
        "use_als": False,
        "weights": {"cf_weight": 1.0, "content_weight": 0.0, "als_weight": 0.0},
    },
    {
        "name": "only_content",
        "removed": "cf+als",
        "use_cf": False,
        "use_content": True,
        "use_als": False,
        "weights": {"cf_weight": 0.0, "content_weight": 1.0, "als_weight": 0.0},
    },
    {
        "name": "only_als",
        "removed": "cf+content",
        "use_cf": False,
        "use_content": False,
        "use_als": True,
        "weights": {"cf_weight": 0.0, "content_weight": 0.0, "als_weight": 1.0},
    },
]


def run_experiment():
    from src.models.hybrid import WeightedHybridRecommender

    mlflow = init_mlflow("ablation-study")

    log("Loading data...")
    test_df = load_test_data()
    train_df = load_train_data()

    log("Loading models...")
    models = load_models()
    cf_model = models.get("collaborative")
    content_model = models.get("content_based")
    als_model = models.get("als")

    log(f"  CF loaded:      {cf_model is not None}")
    log(f"  Content loaded: {content_model is not None}")
    log(f"  ALS loaded:     {als_model is not None}")

    K = 10
    MAX_USERS = 300
    all_results = []
    baseline_ndcg = None

    for cfg in ABLATION_CONFIGS:
        run_name = cfg["name"]
        log(f"\n{'='*60}")
        log(f"Ablation: {run_name}  (removed: {cfg['removed']})")
        log(f"  Weights: {cfg['weights']}")
        log(f"{'='*60}")

        # Build the hybrid with the appropriate subset of models
        hybrid = WeightedHybridRecommender(
            cf_model=cf_model if cfg["use_cf"] else None,
            content_model=content_model if cfg["use_content"] else None,
            als_model=als_model if cfg["use_als"] else None,
            weights=cfg["weights"],
        )
        hybrid.fit()

        with mlflow.start_run(run_name=run_name):
            mlflow.set_tags({
                "experiment_type": "ablation_study",
                "ablation_target": cfg["removed"],
                "config_name": cfg["name"],
            })
            mlflow.log_params({
                "ablation_target": cfg["removed"],
                "cf_weight": cfg["weights"]["cf_weight"],
                "content_weight": cfg["weights"]["content_weight"],
                "als_weight": cfg["weights"]["als_weight"],
                "use_cf": cfg["use_cf"],
                "use_content": cfg["use_content"],
                "use_als": cfg["use_als"],
                "eval_k": K,
                "max_test_users": MAX_USERS,
                "relevance_threshold": 4.0,
                "seed": 42,
            })

            start = time.time()
            metrics = evaluate_model_on_users(
                hybrid, test_df, train_df,
                k=K, max_users=MAX_USERS, relevance_threshold=4.0, seed=42,
            )
            eval_time = time.time() - start
            metrics["eval_time_s"] = eval_time

            ndcg_key = f"ndcg_at_{K}"
            current_ndcg = metrics.get(ndcg_key, 0.0)

            # Record baseline NDCG from full_hybrid
            if cfg["name"] == "full_hybrid":
                baseline_ndcg = current_ndcg

            # Compute delta vs full hybrid
            if baseline_ndcg is not None:
                delta = baseline_ndcg - current_ndcg
            else:
                delta = 0.0
            metrics["delta_ndcg_vs_full"] = delta

            mlflow.log_metrics(metrics)

            prec_key = f"precision_at_{K}"
            log(f"  RMSE:             {metrics.get('rmse', 'N/A')}")
            log(f"  NDCG@{K}:          {metrics.get(ndcg_key, 'N/A')}")
            log(f"  P@{K}:             {metrics.get(prec_key, 'N/A')}")
            log(f"  Coverage:         {metrics.get('catalog_coverage', 'N/A')}")
            log(f"  Delta NDCG:       {delta:+.4f}")
            log(f"  Time:             {eval_time:.1f}s")

            all_results.append({
                "name": cfg["name"],
                "removed": cfg["removed"],
                "rmse": metrics.get("rmse"),
                "ndcg": metrics.get(ndcg_key),
                "precision": metrics.get(prec_key),
                "coverage": metrics.get("catalog_coverage"),
                "delta_ndcg": delta,
                "users_eval": metrics.get("users_evaluated"),
                "time_s": eval_time,
            })

    # ------------------------------------------------------------------
    # Summary Table
    # ------------------------------------------------------------------
    log("\n")
    log("=" * 110)
    log("ABLATION STUDY RESULTS")
    log("=" * 110)

    header = (
        f"{'Config':<16} {'Removed':<14} {'RMSE':>8}  {'NDCG@10':>8}  "
        f"{'P@10':>8}  {'Coverage':>8}  {'dNDCG':>8}  {'Users':>5}  {'Time(s)':>7}"
    )
    log(header)
    log("-" * 110)

    for r in all_results:
        def fmt(v, digits=4):
            return f"{v:.{digits}f}" if v is not None else "N/A"

        delta_str = f"{r['delta_ndcg']:+.4f}" if r["delta_ndcg"] is not None else "N/A"
        line = (
            f"{r['name']:<16} {r['removed']:<14} {fmt(r['rmse']):>8}  "
            f"{fmt(r['ndcg']):>8}  {fmt(r['precision']):>8}  "
            f"{fmt(r['coverage']):>8}  {delta_str:>8}  "
            f"{str(r.get('users_eval', 'N/A')):>5}  {r['time_s']:>7.1f}"
        )
        log(line)

    log("=" * 110)

    # Impact summary
    log("\nComponent Impact (drop in NDCG@10 when removed from full hybrid):")
    log("-" * 50)
    removal_configs = [r for r in all_results if r["removed"] in ("cf", "content", "als")]
    removal_configs.sort(key=lambda r: r["delta_ndcg"] if r["delta_ndcg"] is not None else 0, reverse=True)
    for r in removal_configs:
        delta_str = f"{r['delta_ndcg']:+.4f}" if r["delta_ndcg"] is not None else "N/A"
        log(f"  Remove {r['removed']:<10} -> NDCG drops by {delta_str}")

    if removal_configs:
        most_critical = removal_configs[0]
        log(f"\n  Most critical component: {most_critical['removed']} "
            f"(removing it causes the largest NDCG drop of {most_critical['delta_ndcg']:+.4f})")

    log("\n\nDone! View results at:")
    log("https://dagshub.com/vigneshsabapathi/recommender_system.mlflow")


if __name__ == "__main__":
    run_experiment()
