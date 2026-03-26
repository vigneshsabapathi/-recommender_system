"""Weighted hybrid recommender combining collaborative, content-based, and ALS.

Aggregates normalised scores from multiple sub-models using configurable
weights.  Unknown users or items are handled gracefully -- if a sub-model
cannot produce scores, the remaining models' weights are re-normalised.

Typical usage::

    from src.models.hybrid import WeightedHybridRecommender
    hybrid = WeightedHybridRecommender(
        cf_model=cf, content_model=cb, als_model=als,
        weights={"cf_weight": 0.4, "content_weight": 0.3, "als_weight": 0.3},
    )
    recs = hybrid.recommend(user_id=42, n=10)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from src.models.base import BaseRecommender
from src.utils.logger import get_logger

logger = get_logger(__name__)

_GLOBAL_MEAN_FALLBACK = 3.5


class WeightedHybridRecommender(BaseRecommender):
    """Weighted ensemble of collaborative, content-based, and ALS models.

    Parameters
    ----------
    cf_model : BaseRecommender, optional
        Item-item collaborative filtering model.
    content_model : BaseRecommender, optional
        Content-based recommender model.
    als_model : BaseRecommender, optional
        ALS matrix-factorisation model.
    weights : dict, optional
        Must contain keys ``cf_weight``, ``content_weight``, ``als_weight``
        with float values that sum to 1.0.
    """

    def __init__(
        self,
        cf_model: BaseRecommender | None = None,
        content_model: BaseRecommender | None = None,
        als_model: BaseRecommender | None = None,
        weights: dict[str, float] | None = None,
    ):
        self.cf_model = cf_model
        self.content_model = content_model
        self.als_model = als_model

        default_weights = {"cf_weight": 0.25, "content_weight": 0.35, "als_weight": 0.4}
        self.weights = weights or default_weights

        self.global_mean: float = _GLOBAL_MEAN_FALLBACK

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def fit(self, train_data=None, **kwargs: Any) -> None:
        """No-op for the hybrid -- sub-models must be fitted independently.

        The hybrid model is an aggregator; it does not train any parameters
        of its own.  Call ``fit()`` on each sub-model before constructing
        the hybrid.

        Parameters
        ----------
        train_data
            Ignored.
        """
        # Compute a blended global mean from available sub-models
        means: list[float] = []
        for model in self._available_models():
            if hasattr(model, "global_mean"):
                means.append(model.global_mean)
        if means:
            self.global_mean = float(np.mean(means))

        logger.info(
            "WeightedHybridRecommender initialised with weights: %s (global_mean=%.3f)",
            self.weights, self.global_mean,
        )

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    def recommend(
        self,
        user_id: int,
        n: int = 20,
        exclude_seen: bool = True,
    ) -> list[tuple[int, float]]:
        """Generate recommendations by combining normalised sub-model scores."""
        # Collect raw score dicts from each sub-model
        candidate_pool_size = n * 5  # request extra to allow for overlap filtering
        model_scores: list[tuple[str, dict[int, float], float]] = []

        if self.cf_model is not None:
            cf_recs = self.cf_model.recommend(user_id, n=candidate_pool_size, exclude_seen=exclude_seen)
            if cf_recs:
                scores = {mid: score for mid, score in cf_recs}
                model_scores.append(("cf", scores, self.weights.get("cf_weight", 0.0)))

        if self.content_model is not None:
            cb_recs = self.content_model.recommend(user_id, n=candidate_pool_size, exclude_seen=exclude_seen)
            if cb_recs:
                scores = {mid: score for mid, score in cb_recs}
                model_scores.append(("content", scores, self.weights.get("content_weight", 0.0)))

        if self.als_model is not None:
            als_recs = self.als_model.recommend(user_id, n=candidate_pool_size, exclude_seen=exclude_seen)
            if als_recs:
                scores = {mid: score for mid, score in als_recs}
                model_scores.append(("als", scores, self.weights.get("als_weight", 0.0)))

        if not model_scores:
            # All models failed -- attempt popular fallback from any sub-model
            for model in self._available_models():
                fallback = model.recommend(user_id, n=n, exclude_seen=exclude_seen)
                if fallback:
                    return fallback
            return []

        # Re-normalise weights to account for missing models
        total_weight = sum(w for _, _, w in model_scores)
        if total_weight <= 0:
            total_weight = 1.0

        # Collect all candidate movie IDs
        all_movie_ids: set[int] = set()
        for _, scores, _ in model_scores:
            all_movie_ids.update(scores.keys())

        # Min-max normalise each model's scores to [0, 1]
        normalised: list[tuple[dict[int, float], float]] = []
        for name, scores, weight in model_scores:
            vals = np.array(list(scores.values()), dtype=np.float64)
            s_min = vals.min()
            s_max = vals.max()
            spread = s_max - s_min

            if spread > 0:
                norm_scores = {mid: (s - s_min) / spread for mid, s in scores.items()}
            else:
                # All scores identical -- normalise to 0.5
                norm_scores = {mid: 0.5 for mid in scores}

            normalised.append((norm_scores, weight / total_weight))

        # Combine weighted scores
        combined: dict[int, float] = {}
        for mid in all_movie_ids:
            total = 0.0
            for norm_scores, weight in normalised:
                total += norm_scores.get(mid, 0.0) * weight
            combined[mid] = total

        # Sort and return top-n
        sorted_items = sorted(combined.items(), key=lambda x: x[1], reverse=True)
        return sorted_items[:n]

    def similar_items(self, movie_id: int, n: int = 20) -> list[tuple[int, float]]:
        """Aggregate similar-items results from sub-models.

        Uses the same min-max normalisation and weighting strategy as
        :meth:`recommend`.
        """
        model_scores: list[tuple[str, dict[int, float], float]] = []

        if self.cf_model is not None:
            sims = self.cf_model.similar_items(movie_id, n=n * 3)
            if sims:
                model_scores.append(("cf", dict(sims), self.weights.get("cf_weight", 0.0)))

        if self.content_model is not None:
            sims = self.content_model.similar_items(movie_id, n=n * 3)
            if sims:
                model_scores.append(("content", dict(sims), self.weights.get("content_weight", 0.0)))

        if self.als_model is not None:
            sims = self.als_model.similar_items(movie_id, n=n * 3)
            if sims:
                model_scores.append(("als", dict(sims), self.weights.get("als_weight", 0.0)))

        if not model_scores:
            return []

        total_weight = sum(w for _, _, w in model_scores)
        if total_weight <= 0:
            total_weight = 1.0

        all_ids: set[int] = set()
        for _, scores, _ in model_scores:
            all_ids.update(scores.keys())

        # Normalise and combine
        combined: dict[int, float] = {mid: 0.0 for mid in all_ids}
        for _, scores, weight in model_scores:
            vals = np.array(list(scores.values()), dtype=np.float64)
            s_min, s_max = vals.min(), vals.max()
            spread = s_max - s_min
            for mid, s in scores.items():
                norm_s = (s - s_min) / spread if spread > 0 else 0.5
                combined[mid] += norm_s * (weight / total_weight)

        sorted_items = sorted(combined.items(), key=lambda x: x[1], reverse=True)
        return sorted_items[:n]

    def predict_rating(self, user_id: int, movie_id: int) -> float:
        """Weighted average of sub-model rating predictions."""
        predictions: list[tuple[float, float]] = []

        if self.cf_model is not None:
            pred = self.cf_model.predict_rating(user_id, movie_id)
            predictions.append((pred, self.weights.get("cf_weight", 0.0)))

        if self.content_model is not None:
            pred = self.content_model.predict_rating(user_id, movie_id)
            predictions.append((pred, self.weights.get("content_weight", 0.0)))

        if self.als_model is not None:
            pred = self.als_model.predict_rating(user_id, movie_id)
            predictions.append((pred, self.weights.get("als_weight", 0.0)))

        if not predictions:
            return self.global_mean

        total_weight = sum(w for _, w in predictions)
        if total_weight <= 0:
            return self.global_mean

        weighted_sum = sum(pred * w for pred, w in predictions)
        result = weighted_sum / total_weight
        return float(np.clip(result, 0.5, 5.0))

    # ------------------------------------------------------------------
    # Explainability
    # ------------------------------------------------------------------
    def explain(self, user_id: int, movie_id: int) -> dict:
        """Show per-model contribution breakdown for a recommendation."""
        result: dict[str, Any] = {
            "user_id": user_id,
            "movie_id": movie_id,
            "method": "weighted_hybrid",
            "weights": self.weights,
            "model_contributions": {},
        }

        if self.cf_model is not None:
            cf_explanation = self.cf_model.explain(user_id, movie_id)
            result["model_contributions"]["collaborative"] = {
                "weight": self.weights.get("cf_weight", 0.0),
                "predicted_score": cf_explanation.get("score", self.global_mean),
                "explanation": cf_explanation,
            }

        if self.content_model is not None:
            cb_explanation = self.content_model.explain(user_id, movie_id)
            result["model_contributions"]["content_based"] = {
                "weight": self.weights.get("content_weight", 0.0),
                "predicted_score": cb_explanation.get("score", self.global_mean),
                "explanation": cb_explanation,
            }

        if self.als_model is not None:
            als_explanation = self.als_model.explain(user_id, movie_id)
            result["model_contributions"]["als"] = {
                "weight": self.weights.get("als_weight", 0.0),
                "predicted_score": als_explanation.get("score", self.global_mean),
                "explanation": als_explanation,
            }

        result["score"] = self.predict_rating(user_id, movie_id)
        return result

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def save(self, path: Path) -> None:
        """Save the hybrid model (weights + paths to sub-model directories).

        Each sub-model is saved to its own subdirectory within *path*.
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        meta: dict[str, Any] = {
            "weights": self.weights,
            "global_mean": self.global_mean,
            "sub_models": {},
        }

        if self.cf_model is not None:
            cf_dir = path / "collaborative"
            self.cf_model.save(cf_dir)
            meta["sub_models"]["collaborative"] = str(cf_dir)

        if self.content_model is not None:
            cb_dir = path / "content_based"
            self.content_model.save(cb_dir)
            meta["sub_models"]["content_based"] = str(cb_dir)

        if self.als_model is not None:
            als_dir = path / "als"
            self.als_model.save(als_dir)
            meta["sub_models"]["als"] = str(als_dir)

        with open(path / "meta.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        logger.info("WeightedHybridRecommender saved to %s", path)

    @classmethod
    def load(cls, path: Path) -> "WeightedHybridRecommender":
        """Load a previously saved hybrid model and its sub-models."""
        path = Path(path)

        with open(path / "meta.json", "r", encoding="utf-8") as f:
            meta = json.load(f)

        cf_model = None
        content_model = None
        als_model = None

        sub_models = meta.get("sub_models", {})

        if "collaborative" in sub_models:
            from src.models.collaborative import ItemItemCF
            cf_model = ItemItemCF.load(Path(sub_models["collaborative"]))

        if "content_based" in sub_models:
            from src.models.content_based import ContentBasedRecommender
            content_model = ContentBasedRecommender.load(Path(sub_models["content_based"]))

        if "als" in sub_models:
            from src.models.als_model import SparkALSRecommender
            als_model = SparkALSRecommender.load(Path(sub_models["als"]))

        model = cls(
            cf_model=cf_model,
            content_model=content_model,
            als_model=als_model,
            weights=meta.get("weights"),
        )
        model.global_mean = meta.get("global_mean", _GLOBAL_MEAN_FALLBACK)

        logger.info("WeightedHybridRecommender loaded from %s", path)
        return model

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _available_models(self) -> list[BaseRecommender]:
        """Return the list of non-None sub-models."""
        models: list[BaseRecommender] = []
        if self.cf_model is not None:
            models.append(self.cf_model)
        if self.content_model is not None:
            models.append(self.content_model)
        if self.als_model is not None:
            models.append(self.als_model)
        return models
