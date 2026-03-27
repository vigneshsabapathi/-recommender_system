"use client";

import { useState, useEffect, useRef, Suspense } from "react";
import { useSearchParams } from "next/navigation";
import { motion } from "framer-motion";
import { FiSearch, FiX } from "react-icons/fi";
import MovieCard from "@/components/home/MovieCard";
import { MovieCardSkeleton } from "@/components/ui/Skeleton";
import ErrorState from "@/components/ui/ErrorState";
import { useMovieList } from "@/hooks/useMovies";
import { useSearch } from "@/hooks/useSearch";
import { Button } from "@/components/ui/button";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";

const GENRES = [
  "All", "Action", "Adventure", "Animation", "Children", "Comedy",
  "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror",
  "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western",
];

const SORT_OPTIONS = [
  { value: "num_ratings", label: "Most Popular" },
  { value: "avg_rating", label: "Highest Rated" },
  { value: "year", label: "Newest First" },
  { value: "title", label: "A-Z" },
];

function BrowseContent() {
  const searchParams = useSearchParams();
  const shouldFocusSearch = searchParams.get("search") === "true";

  const [selectedGenre, setSelectedGenre] = useState("All");
  const [sortBy, setSortBy] = useState("num_ratings");
  const [page, setPage] = useState(1);
  const [searchMode, setSearchMode] = useState(shouldFocusSearch);
  const searchInputRef = useRef(null);

  const { query, setQuery, results: searchResults, isLoading: searchLoading } = useSearch("", 30);

  // Auto-focus search input when navigated with ?search=true
  useEffect(() => {
    if (shouldFocusSearch && searchInputRef.current) {
      searchInputRef.current.focus();
      setSearchMode(true);
    }
  }, [shouldFocusSearch]);

  const genre = selectedGenre === "All" ? null : selectedGenre;
  const { data, isLoading: browseLoading, error, refetch } = useMovieList({
    genre,
    page,
    perPage: 30,
    sortBy,
  });

  const isSearching = searchMode && query.trim().length >= 2;
  const movies = isSearching ? (searchResults || []) : (data?.items || []);
  const isLoading = isSearching ? searchLoading : browseLoading;
  const meta = data?.meta;
  const totalPages = meta?.total_pages || 1;

  const handleSearchFocus = () => setSearchMode(true);
  const handleClearSearch = () => {
    setQuery("");
    setSearchMode(false);
    searchInputRef.current?.blur();
  };

  return (
    <div className="min-h-screen bg-netflix-bg">
      {/* Spacer for fixed navbar + breathing room */}
      <div className="h-28" />

      {/* Search - centered */}
      <motion.div
        className="flex justify-center px-4 mb-10"
        initial={{ opacity: 0, y: -10 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.4, ease: "easeOut" }}
      >
        <div className="w-full max-w-md flex items-center bg-white/[0.08] rounded-full border border-white/[0.06] focus-within:bg-white/[0.12] focus-within:border-white/[0.15] transition-all duration-300">
          <span className="pl-5 text-white/30 flex-shrink-0">
            <FiSearch size={16} />
          </span>
          <input
            ref={searchInputRef}
            type="text"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            onFocus={handleSearchFocus}
            placeholder="Search movies, genres, keywords..."
            className="flex-1 bg-transparent text-white py-3 pl-3 pr-4 text-sm placeholder:text-white/25 border-0 outline-none focus:ring-0"
          />
          {query && (
            <button
              onClick={handleClearSearch}
              className="pr-4 text-white/30 hover:text-white transition-colors flex-shrink-0"
            >
              <FiX size={16} />
            </button>
          )}
        </div>
      </motion.div>

      {/* Genre tags - single row, centered, scrollable */}
      <motion.div
        className="px-4 mb-12"
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ duration: 0.4, delay: 0.1 }}
      >
        <div className="flex justify-center">
          <div className="flex flex-nowrap gap-2.5 overflow-x-auto scrollbar-hide py-1 px-1">
          {GENRES.map((g) => (
            <button
              key={g}
              onClick={() => {
                setSelectedGenre(g);
                setPage(1);
                if (searchMode) handleClearSearch();
              }}
              className={`flex-shrink-0 whitespace-nowrap px-4 py-1.5 rounded-full text-[13px] font-medium transition-all duration-200 ${
                !isSearching && selectedGenre === g
                  ? "bg-netflix-red text-white shadow-lg shadow-netflix-red/30"
                  : "bg-white/[0.06] text-white/60 border border-white/[0.08] hover:bg-white/[0.12] hover:text-white hover:border-white/15"
              }`}
            >
              {g}
            </button>
          ))}
          </div>
        </div>
      </motion.div>

      {/* Content */}
      <div className="px-4 md:px-12 pb-12">
        {/* Header row */}
        <div className="flex items-center justify-between mb-6">
          <div>
            <h1 className="text-lg md:text-xl font-medium text-white/90">
              {isSearching
                ? `Results for "${query.trim()}"`
                : selectedGenre === "All"
                  ? "Browse Movies"
                  : selectedGenre}
            </h1>
            <p className="text-xs text-white/30 mt-0.5">
              {isSearching
                ? `${movies.length} movies found`
                : `${meta?.total_items?.toLocaleString() || ""} movies`}
            </p>
          </div>

          {!isSearching && (
            <Select
              value={sortBy}
              onValueChange={(value) => {
                setSortBy(value);
                setPage(1);
              }}
            >
              <SelectTrigger className="bg-white/[0.07] border border-white/[0.08] text-white/80 text-sm h-9 w-[160px] rounded-full hover:bg-white/[0.12] focus:ring-0">
                <SelectValue placeholder="Sort by">
                  {SORT_OPTIONS.find((o) => o.value === sortBy)?.label}
                </SelectValue>
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
          )}
        </div>

        {/* Movie Grid */}
        {error && !isSearching ? (
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
              key={isSearching ? `search-${query}` : `${selectedGenre}-${sortBy}-${page}`}
            >
              {movies.map((movie, i) => (
                <MovieCard key={movie.id} movie={movie} index={i} />
              ))}
            </motion.div>

            {/* Pagination - only for browse mode */}
            {!isSearching && totalPages > 1 && (
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
                      <span key={`dots-${i}`} className="px-2 text-white/30">...</span>
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
              {isSearching
                ? `No movies found for "${query.trim()}"`
                : "No movies found in this category."}
            </p>
          </div>
        )}
      </div>
    </div>
  );
}

export default function BrowsePage() {
  return (
    <Suspense fallback={<div className="min-h-screen bg-netflix-bg" />}>
      <BrowseContent />
    </Suspense>
  );
}

function generatePageNumbers(current, total) {
  if (total <= 7) return Array.from({ length: total }, (_, i) => i + 1);
  const pages = [1];
  if (current > 3) pages.push("...");
  for (let i = Math.max(2, current - 1); i <= Math.min(total - 1, current + 1); i++) pages.push(i);
  if (current < total - 2) pages.push("...");
  pages.push(total);
  return pages;
}
