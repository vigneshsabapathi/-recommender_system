"use client";

import { motion } from "framer-motion";
import MovieCard from "@/components/home/MovieCard";
import { MovieCardSkeleton } from "@/components/ui/Skeleton";
import ErrorState from "@/components/ui/ErrorState";

export default function SimilarMovies({
  movies,
  isLoading,
  error,
  onRetry,
}) {
  if (!isLoading && (!movies || movies.length === 0) && !error) return null;

  return (
    <motion.section
      id="similar"
      className="px-4 md:px-12 mt-12 mb-8"
      initial={{ opacity: 0, y: 20 }}
      whileInView={{ opacity: 1, y: 0 }}
      viewport={{ once: true }}
      transition={{ duration: 0.5 }}
    >
      <h2 className="text-xl md:text-2xl font-semibold text-white mb-6">
        More Like This
      </h2>

      {error ? (
        <ErrorState
          message="Couldn't load similar movies."
          onRetry={onRetry}
        />
      ) : (
        <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5 xl:grid-cols-6 gap-3 md:gap-4">
          {isLoading
            ? Array.from({ length: 12 }).map((_, i) => (
                <MovieCardSkeleton key={i} />
              ))
            : movies?.map((movie, i) => (
                <MovieCard
                  key={movie?.movie?.id || movie?.id || i}
                  movie={movie}
                  index={i}
                />
              ))}
        </div>
      )}
    </motion.section>
  );
}
