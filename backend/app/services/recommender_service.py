"""Recommender service -- loads ML models and exposes a unified inference API.

Handles four algorithm back-ends:
  - collaborative  (ItemItemCF)
  - content_based  (ContentBasedRecommender)
  - als            (SparkALSRecommender)
  - hybrid         (WeightedHybridRecommender)

Unknown users receive a popularity-based fallback.  Missing models are
skipped gracefully at startup so the API can still serve with partial
availability.
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path
from typing import Any

from backend.app.config import get_settings

logger = logging.getLogger(__name__)

# Algorithm name -> (model directory name, class import path)
_MODEL_REGISTRY: dict[str, tuple[str, str, str]] = {
    "collaborative": ("collaborative", "src.models.collaborative", "ItemItemCF"),
    "content_based": ("content_based", "src.models.content_based", "ContentBasedRecommender"),
    "als": ("als", "src.models.als_model", "SparkALSRecommender"),
}

# Valid algorithm names exposed to the API
VALID_ALGORITHMS = {"collaborative", "content_based", "als", "hybrid", "blended_similar"}


def _ensure_src_on_path() -> None:
    """Ensure the project root is on ``sys.path`` so ``src.models`` imports resolve."""
    settings = get_settings()
    project_root = str(settings.PROJECT_ROOT)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)


class RecommenderService:
    """Facade over the four recommender models.

    Models are loaded lazily at init time from the configured ``MODEL_DIR``.
    Any model whose artefacts are missing is silently skipped.
    """

    def __init__(self) -> None:
        self.models: dict[str, Any] = {}
        self._loaded_algorithms: list[str] = []

    # ------------------------------------------------------------------
    # Bootstrap
    # ------------------------------------------------------------------
    def load_models(self) -> list[str]:
        """Attempt to load each registered model from disk.

        Returns the list of algorithms that loaded successfully.
        """
        _ensure_src_on_path()
        settings = get_settings()
        model_dir = settings.MODEL_DIR

        for algo_name, (dir_name, module_path, class_name) in _MODEL_REGISTRY.items():
            model_path = model_dir / dir_name
            if not model_path.exists() or not any(model_path.iterdir()):
                logger.warning("Model directory empty or missing: %s -- skipping %s", model_path, algo_name)
                continue
            try:
                t0 = time.perf_counter()
                import importlib
                module = importlib.import_module(module_path)
                cls = getattr(module, class_name)
                model = cls.load(model_path)
                self.models[algo_name] = model
                elapsed = (time.perf_counter() - t0) * 1000
                logger.info("Loaded %s model in %.0f ms", algo_name, elapsed)
            except Exception:
                logger.exception("Failed to load %s model from %s", algo_name, model_path)

        # Build hybrid if at least two sub-models loaded
        self._build_hybrid(model_dir)

        self._loaded_algorithms = list(self.models.keys())
        logger.info("Models available: %s", self._loaded_algorithms)
        return self._loaded_algorithms

    def _build_hybrid(self, model_dir: Path) -> None:
        """Construct the hybrid model from loaded sub-models.

        If a pre-saved hybrid exists on disk, load it.  Otherwise, build
        one in-memory from whichever sub-models are available.
        """
        hybrid_dir = model_dir / "hybrid"
        meta_path = hybrid_dir / "meta.json"

        # Try loading a saved hybrid first
        if meta_path.exists():
            try:
                _ensure_src_on_path()
                from src.models.hybrid import WeightedHybridRecommender
                model = WeightedHybridRecommender.load(hybrid_dir)
                self.models["hybrid"] = model
                logger.info("Loaded saved hybrid model from %s", hybrid_dir)
                return
            except Exception:
                logger.warning("Failed to load saved hybrid model -- building from sub-models")

        # Build from loaded sub-models
        cf = self.models.get("collaborative")
        cb = self.models.get("content_based")
        als = self.models.get("als")

        available_count = sum(1 for m in (cf, cb, als) if m is not None)
        if available_count < 1:
            logger.info("Not enough sub-models to build hybrid (need >= 1, have %d)", available_count)
            return

        try:
            _ensure_src_on_path()
            from src.models.hybrid import WeightedHybridRecommender
            hybrid = WeightedHybridRecommender(
                cf_model=cf,
                content_model=cb,
                als_model=als,
            )
            hybrid.fit()
            self.models["hybrid"] = hybrid
            logger.info("Built hybrid model from %d sub-models", available_count)
        except Exception:
            logger.exception("Failed to build hybrid model")

    # ------------------------------------------------------------------
    # Public query API
    # ------------------------------------------------------------------
    @property
    def loaded_algorithms(self) -> list[str]:
        return list(self._loaded_algorithms)

    def get_recommendations(
        self,
        user_id: int,
        algorithm: str = "hybrid",
        n: int = 20,
        explain: bool = False,
    ) -> list[dict[str, Any]]:
        """Generate top-*n* recommendations for a user.

        Returns a list of dicts with keys ``movie_id``, ``score``,
        ``predicted_rating``, and optionally ``explanation``.
        """
        model = self._resolve_model(algorithm)
        if model is None:
            # Total fallback: return popular items from any available model
            return self._popularity_fallback(n)

        try:
            raw_recs = model.recommend(user_id, n=n, exclude_seen=True)
        except Exception:
            logger.exception("recommend() failed for user=%d algo=%s", user_id, algorithm)
            return self._popularity_fallback(n)

        results: list[dict[str, Any]] = []
        for movie_id, score in raw_recs:
            entry: dict[str, Any] = {
                "movie_id": int(movie_id),
                "score": float(score),
                "predicted_rating": None,
                "explanation": [],
            }

            # Predicted rating
            try:
                entry["predicted_rating"] = float(model.predict_rating(user_id, movie_id))
            except Exception:
                pass

            # Explanation
            if explain:
                try:
                    expl = model.explain(user_id, movie_id)
                    entry["explanation"] = self._format_explanation(expl, algorithm)
                except Exception:
                    pass

            results.append(entry)

        return results

    def get_similar(
        self,
        movie_id: int,
        algorithm: str = "blended_similar",
        n: int = 20,
    ) -> list[dict[str, Any]]:
        """Find similar movies to a given movie.

        Returns a list of dicts with ``movie_id`` and ``score``.

        When *algorithm* is ``"blended_similar"`` (the default), scores from
        the content-based model (weight 0.7) and collaborative model (weight
        0.3) are merged after min-max normalisation, producing results that
        respect content similarity while still accounting for user co-viewing
        patterns.
        """
        if algorithm == "blended_similar":
            return self._blended_similar(movie_id, n)

        model = self._resolve_model(algorithm)
        if model is None:
            return []

        try:
            raw_sims = model.similar_items(movie_id, n=n)
        except Exception:
            logger.exception("similar_items() failed for movie=%d algo=%s", movie_id, algorithm)
            return []

        return [
            {"movie_id": int(mid), "score": float(sc), "predicted_rating": None, "explanation": []}
            for mid, sc in raw_sims
        ]

    def _blended_similar(
        self,
        movie_id: int,
        n: int,
        content_weight: float = 0.7,
        cf_weight: float = 0.3,
    ) -> list[dict[str, Any]]:
        """Blend content-based and collaborative similar-items scores.

        Fetches ``n * 2`` candidates from each model, min-max normalises each
        set of scores to [0, 1], then combines them as::

            final_score = content_weight * content_score + cf_weight * cf_score

        Returns the top *n* items sorted by ``final_score``.
        """
        import numpy as np

        fetch_n = n * 2

        # -- Content-based scores (primary signal) -------------------------
        content_scores: dict[int, float] = {}
        cb_model = self.models.get("content_based")
        if cb_model is not None:
            try:
                raw = cb_model.similar_items(movie_id, n=fetch_n)
                content_scores = {int(mid): float(sc) for mid, sc in raw}
            except Exception:
                logger.exception("content_based.similar_items() failed for movie=%d", movie_id)

        # -- Collaborative scores (secondary signal) -----------------------
        cf_scores: dict[int, float] = {}
        cf_model = self.models.get("collaborative")
        if cf_model is not None:
            try:
                raw = cf_model.similar_items(movie_id, n=fetch_n)
                cf_scores = {int(mid): float(sc) for mid, sc in raw}
            except Exception:
                logger.exception("collaborative.similar_items() failed for movie=%d", movie_id)

        # Fallback: if only one model is available, use it directly
        if not content_scores and not cf_scores:
            return []
        if not content_scores:
            sorted_cf = sorted(cf_scores.items(), key=lambda x: x[1], reverse=True)[:n]
            return [{"movie_id": mid, "score": sc, "predicted_rating": None, "explanation": []} for mid, sc in sorted_cf]
        if not cf_scores:
            sorted_cb = sorted(content_scores.items(), key=lambda x: x[1], reverse=True)[:n]
            return [{"movie_id": mid, "score": sc, "predicted_rating": None, "explanation": []} for mid, sc in sorted_cb]

        # -- Min-max normalise each score set to [0, 1] --------------------
        def _normalise(scores: dict[int, float]) -> dict[int, float]:
            vals = np.array(list(scores.values()), dtype=np.float64)
            s_min, s_max = vals.min(), vals.max()
            spread = s_max - s_min
            if spread > 0:
                return {mid: (s - s_min) / spread for mid, s in scores.items()}
            return {mid: 0.5 for mid in scores}

        norm_content = _normalise(content_scores)
        norm_cf = _normalise(cf_scores)

        # -- Merge scores --------------------------------------------------
        all_ids = set(norm_content.keys()) | set(norm_cf.keys())
        combined: dict[int, float] = {}
        for mid in all_ids:
            cs = norm_content.get(mid, 0.0)
            cfs = norm_cf.get(mid, 0.0)
            combined[mid] = content_weight * cs + cf_weight * cfs

        sorted_items = sorted(combined.items(), key=lambda x: x[1], reverse=True)[:n]
        return [
            {"movie_id": mid, "score": round(sc, 6), "predicted_rating": None, "explanation": []}
            for mid, sc in sorted_items
        ]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _resolve_model(self, algorithm: str) -> Any | None:
        """Look up a model by algorithm name, with fallback chain."""
        if algorithm in self.models:
            return self.models[algorithm]

        # Fallback: try hybrid -> collaborative -> content_based -> als
        for fallback in ("hybrid", "collaborative", "content_based", "als"):
            if fallback in self.models:
                logger.info("Algorithm '%s' not available; falling back to '%s'", algorithm, fallback)
                return self.models[fallback]

        return None

    def _popularity_fallback(self, n: int) -> list[dict[str, Any]]:
        """Return popular-item stubs when no model can serve the user."""
        for model in self.models.values():
            try:
                recs = model.recommend(user_id=-1, n=n, exclude_seen=False)
                return [
                    {"movie_id": int(mid), "score": float(sc), "predicted_rating": None, "explanation": []}
                    for mid, sc in recs
                ]
            except Exception:
                continue
        return []

    @staticmethod
    def _format_explanation(raw: dict, algorithm: str) -> list[dict[str, Any]]:
        """Normalise model-specific explanation dicts into ExplanationItem-like dicts."""
        items: list[dict[str, Any]] = []
        method = raw.get("method", algorithm)

        # Collaborative: contributing items
        for contrib in raw.get("contributing_items", [])[:5]:
            items.append({
                "algorithm": method,
                "score": contrib.get("weighted_contribution", 0.0),
                "reason": (
                    f"You rated movie {contrib.get('movie_id')} "
                    f"({contrib.get('user_rating', '?')}/5) -- "
                    f"similarity {contrib.get('similarity', 0):.3f}"
                ),
            })

        # Content-based: similar items by content
        for sim in raw.get("similar_items_by_content", [])[:5]:
            items.append({
                "algorithm": method,
                "score": sim.get("similarity", 0.0),
                "reason": f"Content-similar movie {sim.get('movie_id')} (similarity {sim.get('similarity', 0):.3f})",
            })

        # ALS: latent factors
        for lf in raw.get("latent_factor_contributions", [])[:5]:
            items.append({
                "algorithm": method,
                "score": lf.get("contribution", 0.0),
                "reason": f"Latent dimension {lf.get('dimension')} contribution {lf.get('contribution', 0):.4f}",
            })

        # Hybrid: per-model contributions
        for model_name, contrib in raw.get("model_contributions", {}).items():
            items.append({
                "algorithm": f"hybrid/{model_name}",
                "score": contrib.get("predicted_score", 0.0),
                "reason": f"{model_name} predicted {contrib.get('predicted_score', 0):.2f} (weight {contrib.get('weight', 0):.2f})",
            })

        # Generic fallback if nothing above matched
        if not items and "score" in raw:
            items.append({
                "algorithm": method,
                "score": raw["score"],
                "reason": raw.get("note", ""),
            })

        return items
