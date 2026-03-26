import axios from "axios";

/**
 * Axios client pointing at the FastAPI backend.
 * In development Next.js rewrites /api/* to the backend,
 * so we use a relative URL base.
 */
const api = axios.create({
  baseURL: "/api/v1",
  timeout: 15000,
  headers: { "Content-Type": "application/json" },
});

// ─── Movies ──────────────────────────────────────────────────────────
export async function fetchMovies({
  genre = null,
  page = 1,
  perPage = 20,
  sortBy = "num_ratings",
} = {}) {
  const params = { page, per_page: perPage, sort_by: sortBy };
  if (genre) params.genre = genre;
  const { data } = await api.get("/movies", { params });
  return data; // { items: MovieCard[], meta: PaginationMeta }
}

export async function fetchMovieDetail(movieId) {
  const { data } = await api.get(`/movies/${movieId}`);
  return data; // MovieDetail
}

export async function searchMovies(query, limit = 20) {
  if (!query || query.trim().length === 0) return [];
  const { data } = await api.get("/movies/search", {
    params: { q: query, limit },
  });
  return data; // MovieCard[]
}

// ─── Recommendations ─────────────────────────────────────────────────
export async function fetchRecommendations({
  userId,
  algorithm = "hybrid",
  n = 20,
  explain = true,
} = {}) {
  const { data } = await api.get(`/recommendations/${userId}`, {
    params: { algorithm, n, explain },
  });
  return data; // RecommendationResponse
}

// ─── Similar Movies ──────────────────────────────────────────────────
export async function fetchSimilarMovies({
  movieId,
  algorithm = "collaborative",
  n = 20,
} = {}) {
  const { data } = await api.get(`/similar/${movieId}`, {
    params: { algorithm, n },
  });
  return data; // SimilarMoviesResponse
}

// ─── Health ──────────────────────────────────────────────────────────
export async function fetchHealth() {
  const { data } = await api.get("/health");
  return data;
}

export default api;
