"use client";

import { useQuery } from "@tanstack/react-query";
import { fetchMovies, fetchMovieDetail, fetchSimilarMovies } from "@/lib/api";

export function useMovieList({ genre, page, perPage, sortBy } = {}) {
  return useQuery({
    queryKey: ["movies", genre, page, perPage, sortBy],
    queryFn: () => fetchMovies({ genre, page, perPage, sortBy }),
    staleTime: 5 * 60 * 1000,
    retry: 2,
  });
}

export function useMovieDetail(movieId) {
  return useQuery({
    queryKey: ["movie", movieId],
    queryFn: () => fetchMovieDetail(movieId),
    staleTime: 10 * 60 * 1000,
    enabled: !!movieId,
    retry: 2,
  });
}

export function useSimilarMovies(movieId, algorithm = "collaborative") {
  return useQuery({
    queryKey: ["similar", movieId, algorithm],
    queryFn: () =>
      fetchSimilarMovies({ movieId, algorithm, n: 20 }),
    staleTime: 5 * 60 * 1000,
    enabled: !!movieId,
    retry: 2,
  });
}
