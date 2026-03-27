"use client";

import { useState } from "react";
import { motion } from "framer-motion";
import MovieCard from "@/components/home/MovieCard";
import { MovieCardSkeleton } from "@/components/ui/Skeleton";
import ErrorState from "@/components/ui/ErrorState";
import { useMovieList } from "@/hooks/useMovies";
import { Badge } from "@/components/ui/badge";
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
    <div className="pt-20 pb-12 px-4 md:px-12 bg-netflix-bg min-h-screen">
      {/* Header */}
      <div className="mb-8">
        <h1 className="text-3xl md:text-4xl font-bold text-white mb-2">
          Browse Movies
        </h1>
        <p className="text-netflix-text-secondary text-sm">
          {meta?.total_items?.toLocaleString() || "Thousands of"} movies to
          discover
        </p>
      </div>

      {/* Filters */}
      <div className="flex flex-col sm:flex-row gap-4 mb-8">
        {/* Genre pills */}
        <div className="flex flex-wrap gap-2 flex-1">
          {GENRES.map((g) => (
            <Badge
              key={g}
              variant={selectedGenre === g ? "default" : "outline"}
              onClick={() => {
                setSelectedGenre(g);
                setPage(1);
              }}
              className={`cursor-pointer px-3 py-1.5 h-auto text-sm transition-all duration-200 ${
                selectedGenre === g
                  ? "bg-netflix-red border-netflix-red text-white hover:bg-netflix-red-hover"
                  : "bg-transparent border-netflix-border/40 text-netflix-text-secondary hover:border-white/40 hover:text-white"
              }`}
            >
              {g}
            </Badge>
          ))}
        </div>

        {/* Sort dropdown */}
        <div className="flex-shrink-0">
          <Select
            value={sortBy}
            onValueChange={(value) => {
              setSortBy(value);
              setPage(1);
            }}
          >
            <SelectTrigger className="bg-netflix-card border-netflix-border/40 text-white hover:border-white/40 min-w-[160px]">
              <SelectValue />
            </SelectTrigger>
            <SelectContent className="bg-netflix-dark border-netflix-border/40">
              {SORT_OPTIONS.map((opt) => (
                <SelectItem
                  key={opt.value}
                  value={opt.value}
                  className="text-white hover:bg-white/10 focus:bg-white/10 focus:text-white"
                >
                  {opt.label}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>
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
                variant="outline"
                onClick={() => setPage((p) => Math.max(1, p - 1))}
                disabled={page <= 1}
                className="bg-netflix-card border-netflix-border/40 text-white hover:bg-netflix-hover"
              >
                Previous
              </Button>

              <div className="flex items-center gap-1">
                {generatePageNumbers(page, totalPages).map((p, i) =>
                  p === "..." ? (
                    <span
                      key={`dots-${i}`}
                      className="px-2 text-netflix-text-secondary"
                    >
                      ...
                    </span>
                  ) : (
                    <Button
                      key={p}
                      variant={page === p ? "default" : "outline"}
                      size="icon"
                      onClick={() => setPage(p)}
                      className={
                        page === p
                          ? "w-9 h-9 bg-netflix-red text-white hover:bg-netflix-red-hover"
                          : "w-9 h-9 bg-netflix-card text-netflix-text-secondary border-netflix-border/40 hover:bg-netflix-hover hover:text-white"
                      }
                    >
                      {p}
                    </Button>
                  )
                )}
              </div>

              <Button
                variant="outline"
                onClick={() => setPage((p) => Math.min(totalPages, p + 1))}
                disabled={page >= totalPages}
                className="bg-netflix-card border-netflix-border/40 text-white hover:bg-netflix-hover"
              >
                Next
              </Button>
            </div>
          )}
        </>
      ) : (
        <div className="text-center py-20">
          <p className="text-netflix-text-secondary text-lg">
            No movies found in this category.
          </p>
        </div>
      )}
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
