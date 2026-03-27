"use client";

import { useState } from "react";
import { motion } from "framer-motion";
import MovieCard from "@/components/home/MovieCard";
import { MovieCardSkeleton } from "@/components/ui/Skeleton";
import ErrorState from "@/components/ui/ErrorState";
import { useMovieList } from "@/hooks/useMovies";
import { Button } from "@/components/ui/button";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";

const GENRES = [
  "All",
  "Action",
  "Adventure",
  "Animation",
  "Children",
  "Comedy",
  "Crime",
  "Documentary",
  "Drama",
  "Fantasy",
  "Film-Noir",
  "Horror",
  "Musical",
  "Mystery",
  "Romance",
  "Sci-Fi",
  "Thriller",
  "War",
  "Western",
];

const SORT_OPTIONS = [
  { value: "num_ratings", label: "Most Popular" },
  { value: "avg_rating", label: "Highest Rated" },
  { value: "year", label: "Newest First" },
  { value: "title", label: "A-Z" },
];

export default function BrowsePage() {
  const [selectedGenre, setSelectedGenre] = useState("All");
  const [sortBy, setSortBy] = useState("num_ratings");
  const [page, setPage] = useState(1);

  const genre = selectedGenre === "All" ? null : selectedGenre;

  const { data, isLoading, error, refetch } = useMovieList({
    genre,
    page,
    perPage: 30,
    sortBy,
  });

  const movies = data?.items || [];
  const meta = data?.meta;
  const totalPages = meta?.total_pages || 1;

  return (
    <div className="min-h-screen bg-netflix-bg">
      {/* Spacer for fixed navbar */}
      <div className="h-16" />

      {/* Sticky genre bar - sits below navbar */}
      <div className="sticky top-16 z-40 bg-netflix-bg/95 backdrop-blur-md border-b border-white/5 px-4 md:px-12 py-3">
        <div className="flex items-center gap-3 overflow-x-auto scrollbar-hide">
          {GENRES.map((g) => (
            <button
              key={g}
              onClick={() => {
                setSelectedGenre(g);
                setPage(1);
              }}
              className={`flex-shrink-0 px-4 py-1.5 rounded-lg text-sm font-medium transition-all duration-200 ${
                selectedGenre === g
                  ? "bg-white text-black"
                  : "bg-[#272727] text-white/90 hover:bg-[#3a3a3a]"
              }`}
            >
              {g}
            </button>
          ))}
        </div>
      </div>

      {/* Content */}
      <div className="px-4 md:px-12 pt-6 pb-12">
        {/* Header row with title + sort */}
        <div className="flex items-end justify-between mb-6">
          <div>
            <h1 className="text-xl md:text-2xl font-semibold text-white">
              {selectedGenre === "All" ? "Browse Movies" : selectedGenre}
            </h1>
            <p className="text-sm text-white/40 mt-0.5">
              {meta?.total_items?.toLocaleString() || ""} movies
            </p>
          </div>

          <Select
            value={sortBy}
            onValueChange={(value) => {
              setSortBy(value);
              setPage(1);
            }}
          >
            <SelectTrigger className="bg-[#272727] border-0 text-white/80 text-sm h-9 w-[150px] rounded-lg hover:bg-[#3a3a3a] focus:ring-0">
              <SelectValue />
            </SelectTrigger>
            <SelectContent className="bg-[#212121] border-[#333]">
              {SORT_OPTIONS.map((opt) => (
                <SelectItem
                  key={opt.value}
                  value={opt.value}
                  className="text-white/80 hover:bg-white/10 focus:bg-white/10 focus:text-white text-sm"
                >
                  {opt.label}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>

        {/* Movie Grid */}
        {error ? (
          <ErrorState message="Couldn't load movies." onRetry={refetch} />
        ) : isLoading ? (
          <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5 xl:grid-cols-6 gap-3 md:gap-4">
            {Array.from({ length: 18 }).map((_, i) => (
              <MovieCardSkeleton key={i} />
            ))}
          </div>
        ) : movies.length > 0 ? (
          <>
            <motion.div
              className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5 xl:grid-cols-6 gap-3 md:gap-4"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ duration: 0.3 }}
              key={`${selectedGenre}-${sortBy}-${page}`}
            >
              {movies.map((movie, i) => (
                <MovieCard key={movie.id} movie={movie} index={i} />
              ))}
            </motion.div>

            {/* Pagination */}
            {totalPages > 1 && (
              <div className="flex items-center justify-center gap-2 mt-10">
                <Button
                  variant="ghost"
                  onClick={() => setPage((p) => Math.max(1, p - 1))}
                  disabled={page <= 1}
                  className="text-white/60 hover:text-white hover:bg-white/10 disabled:opacity-30"
                >
                  Previous
                </Button>

                <div className="flex items-center gap-1">
                  {generatePageNumbers(page, totalPages).map((p, i) =>
                    p === "..." ? (
                      <span
                        key={`dots-${i}`}
                        className="px-2 text-white/30"
                      >
                        ...
                      </span>
                    ) : (
                      <button
                        key={p}
                        onClick={() => setPage(p)}
                        className={`w-9 h-9 rounded-lg text-sm font-medium transition-colors ${
                          page === p
                            ? "bg-white text-black"
                            : "text-white/60 hover:bg-white/10 hover:text-white"
                        }`}
                      >
                        {p}
                      </button>
                    )
                  )}
                </div>

                <Button
                  variant="ghost"
                  onClick={() => setPage((p) => Math.min(totalPages, p + 1))}
                  disabled={page >= totalPages}
                  className="text-white/60 hover:text-white hover:bg-white/10 disabled:opacity-30"
                >
                  Next
                </Button>
              </div>
            )}
          </>
        ) : (
          <div className="text-center py-20">
            <p className="text-white/40 text-lg">
              No movies found in this category.
            </p>
          </div>
        )}
      </div>
    </div>
  );
}

function generatePageNumbers(current, total) {
  if (total <= 7) return Array.from({ length: total }, (_, i) => i + 1);

  const pages = [];
  pages.push(1);

  if (current > 3) pages.push("...");

  const start = Math.max(2, current - 1);
  const end = Math.min(total - 1, current + 1);

  for (let i = start; i <= end; i++) pages.push(i);

  if (current < total - 2) pages.push("...");

  pages.push(total);
  return pages;
}
