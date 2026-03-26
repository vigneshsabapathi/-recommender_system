"""Abstract base class for all recommender models.

Every recommender in this project inherits from :class:`BaseRecommender`,
which enforces a consistent interface for training, inference, persistence,
and explainability.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path


class BaseRecommender(ABC):
    """Abstract recommender interface.

    Subclasses must implement all abstract methods.  The ``save`` / ``load``
    contract allows models to be serialised to a directory and restored
    without knowing the concrete class at load time (each subclass bundles
    its own serialisation logic).
    """

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    @abstractmethod
    def fit(self, train_data, **kwargs) -> None:
        """Train (or fit) the model on the provided data.

        Parameters
        ----------
        train_data
            Primary training artefact -- the exact type depends on the
            subclass (e.g. a sparse matrix, a pandas DataFrame, etc.).
        **kwargs
            Additional data or configuration required by the concrete
            model (e.g. id-mapping dicts, params, etc.).
        """

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    @abstractmethod
    def recommend(
        self,
        user_id: int,
        n: int = 20,
        exclude_seen: bool = True,
    ) -> list[tuple[int, float]]:
        """Generate top-*n* recommendations for a user.

        Parameters
        ----------
        user_id : int
            Original user ID (not the internal matrix index).
        n : int
            Number of recommendations to return.
        exclude_seen : bool
            If ``True``, items the user has already rated are excluded.

        Returns
        -------
        list[tuple[int, float]]
            List of ``(movie_id, score)`` pairs sorted by descending score.
            ``movie_id`` is the original ID, ``score`` is model-specific.
        """

    @abstractmethod
    def similar_items(
        self,
        movie_id: int,
        n: int = 20,
    ) -> list[tuple[int, float]]:
        """Find the *n* most similar items to a given movie.

        Parameters
        ----------
        movie_id : int
            Original movie ID.
        n : int
            Number of similar items to return.

        Returns
        -------
        list[tuple[int, float]]
            List of ``(movie_id, similarity)`` sorted descending.
        """

    @abstractmethod
    def predict_rating(self, user_id: int, movie_id: int) -> float:
        """Predict a single user-movie rating.

        Used primarily by the evaluation harness for RMSE computation.

        Parameters
        ----------
        user_id : int
            Original user ID.
        movie_id : int
            Original movie ID.

        Returns
        -------
        float
            Predicted rating (typically on the same scale as the training
            data, e.g. 0.5 -- 5.0 for MovieLens).
        """

    # ------------------------------------------------------------------
    # Explainability
    # ------------------------------------------------------------------
    @abstractmethod
    def explain(self, user_id: int, movie_id: int) -> dict:
        """Explain *why* a movie is recommended to a user.

        The structure of the returned dict is model-specific but should
        always contain a ``"score"`` key with the predicted relevance.

        Parameters
        ----------
        user_id : int
            Original user ID.
        movie_id : int
            Original movie ID.

        Returns
        -------
        dict
            Explanation payload (model-specific keys).
        """

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    @abstractmethod
    def save(self, path: Path) -> None:
        """Serialise the fitted model to *path* (a directory).

        The directory is created if it does not exist.

        Parameters
        ----------
        path : Path
            Target directory for model artefacts.
        """

    @classmethod
    @abstractmethod
    def load(cls, path: Path) -> "BaseRecommender":
        """Deserialise a model previously saved with :meth:`save`.

        Parameters
        ----------
        path : Path
            Directory containing saved artefacts.

        Returns
        -------
        BaseRecommender
            A fully initialised model ready for inference.
        """
