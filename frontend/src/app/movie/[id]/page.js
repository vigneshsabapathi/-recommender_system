"use client";

import { use } from "react";
import { useMovieDetail, useSimilarMovies } from "@/hooks/useMovies";
import MovieDetail from "@/components/movie/MovieDetail";
import SimilarMovies from "@/components/movie/SimilarMovies";
import useAlgorithmStore from "@/stores/algorithmStore";

export default function MoviePage({ params }) {
  const { id } = use(params);
  const movieId = parseInt(id, 10);
  const algorithm = useAlgorithmStore((s) => s.algorithm);

  const {
    data: movie,
    isLoading: movieLoading,
    error: movieError,
    refetch: refetchMovie,
  } = useMovieDetail(movieId);

  const {
    data: similarData,
    isLoading: similarLoading,
    error: similarError,
    refetch: refetchSimilar,
  } = useSimilarMovies(movieId, algorithm);

  return (
    <div className="bg-netflix-bg min-h-screen">
      <MovieDetail
        movie={movie}
        isLoading={movieLoading}
        error={movieError}
        onRetry={refetchMovie}
      />

      <SimilarMovies
        movies={similarData?.similar}
        isLoading={similarLoading}
        error={similarError}
        onRetry={refetchSimilar}
      />
    </div>
  );
}
