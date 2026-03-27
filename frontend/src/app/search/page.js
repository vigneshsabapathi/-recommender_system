"use client";

import { useSearchParams } from "next/navigation";
import { Suspense } from "react";
import { motion } from "framer-motion";
import { FiSearch } from "react-icons/fi";
import { useSearch } from "@/hooks/useSearch";
import MovieCard from "@/components/home/MovieCard";
import { MovieCardSkeleton } from "@/components/ui/Skeleton";
import ErrorState from "@/components/ui/ErrorState";
import { Input } from "@/components/ui/input";

function SearchContent() {
  const searchParams = useSearchParams();
  const initialQuery = searchParams.get("q") || "";
  const { query, setQuery, results, isLoading, error } = useSearch(initialQuery);

  return (
    <div className="pt-24 pb-12 px-4 md:px-12 bg-netflix-bg min-h-screen">
      {/* Search input */}
      <div className="max-w-2xl mx-auto mb-10">
        <div className="relative">
          <FiSearch
            size={20}
            className="absolute left-4 top-1/2 -translate-y-1/2 text-netflix-text-secondary z-10"
          />
          <Input
            type="text"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Search movies, genres..."
            autoFocus
            className="w-full bg-netflix-card border-netflix-border text-lg pl-12 pr-4 py-4 h-auto rounded-lg"
          />
        </div>
      </div>

      {/* Results */}
      {error ? (
        <ErrorState message="Search failed. Please try again." />
      ) : isLoading ? (
        <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5 xl:grid-cols-6 gap-3 md:gap-4">
          {Array.from({ length: 12 }).map((_, i) => (
            <MovieCardSkeleton key={i} />
          ))}
        </div>
      ) : results.length > 0 ? (
        <>
          <p className="text-netflix-text-secondary text-sm mb-6">
            {results.length} result{results.length !== 1 ? "s" : ""} for &ldquo;
            {query}&rdquo;
          </p>
          <motion.div
            className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5 xl:grid-cols-6 gap-3 md:gap-4"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ duration: 0.3 }}
          >
            {results.map((movie, i) => (
              <MovieCard key={movie.id} movie={movie} index={i} />
            ))}
          </motion.div>
        </>
      ) : query.trim().length > 0 ? (
        <div className="text-center py-20">
          <p className="text-netflix-text-secondary text-lg mb-2">
            No results found for &ldquo;{query}&rdquo;
          </p>
          <p className="text-netflix-text-secondary/60 text-sm">
            Try different keywords or check the spelling.
          </p>
        </div>
      ) : (
        <div className="text-center py-20">
          <FiSearch size={48} className="text-netflix-text-secondary/30 mx-auto mb-4" />
          <p className="text-netflix-text-secondary text-lg">
            Search for your favorite movies
          </p>
        </div>
      )}
    </div>
  );
}

export default function SearchPage() {
  return (
    <Suspense
      fallback={
        <div className="pt-24 pb-12 px-4 md:px-12 bg-netflix-bg min-h-screen">
          <div className="max-w-2xl mx-auto mb-10">
            <div className="skeleton-shimmer h-14 rounded-lg" />
          </div>
        </div>
      }
    >
      <SearchContent />
    </Suspense>
  );
}
