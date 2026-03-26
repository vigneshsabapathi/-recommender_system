"use client";

import Image from "next/image";
import Link from "next/link";
import { motion } from "framer-motion";
import { FiExternalLink, FiArrowLeft } from "react-icons/fi";
import GenreBadge from "@/components/ui/GenreBadge";
import StarRating from "@/components/ui/StarRating";
import { MovieDetailSkeleton } from "@/components/ui/Skeleton";
import ErrorState from "@/components/ui/ErrorState";

const BACKDROP_GRADIENTS = [
  "linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%)",
  "linear-gradient(135deg, #2d1b2e 0%, #1a1a2e 50%, #0d1b2a 100%)",
  "linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%)",
];

export default function MovieDetail({ movie, isLoading, error, onRetry }) {
  if (isLoading) return <MovieDetailSkeleton />;
  if (error) {
    return (
      <div className="min-h-screen bg-netflix-bg pt-20">
        <ErrorState message="Couldn't load movie details." onRetry={onRetry} />
      </div>
    );
  }
  if (!movie) return null;

  const backdropUrl = movie.backdrop_url || movie.poster_url;
  const gradient = BACKDROP_GRADIENTS[movie.id % BACKDROP_GRADIENTS.length];
  const imdbUrl = movie.imdb_id
    ? `https://www.imdb.com/title/${movie.imdb_id}/`
    : null;

  return (
    <div className="min-h-screen bg-netflix-bg">
      {/* Backdrop */}
      <div className="relative w-full h-[50vh] md:h-[65vh]">
        {backdropUrl ? (
          <Image
            src={backdropUrl}
            alt={movie.title}
            fill
            className="object-cover"
            priority
            sizes="100vw"
          />
        ) : (
          <div className="absolute inset-0" style={{ background: gradient }} />
        )}
        <div className="absolute inset-0 hero-gradient z-[1]" />
        <div className="absolute inset-0 hero-gradient-bottom z-[1]" />

        {/* Back button */}
        <Link
          href="/home"
          className="absolute top-20 left-4 md:left-12 z-10 flex items-center gap-2
                     text-white/70 hover:text-white text-sm transition-colors
                     bg-black/30 backdrop-blur-sm px-3 py-1.5 rounded-full"
        >
          <FiArrowLeft size={16} />
          Back
        </Link>
      </div>

      {/* Content */}
      <motion.div
        className="relative z-10 px-4 md:px-12 -mt-40 md:-mt-52 max-w-5xl"
        initial={{ opacity: 0, y: 30 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.6 }}
      >
        <div className="flex flex-col md:flex-row gap-8">
          {/* Poster */}
          {movie.poster_url && (
            <div className="hidden md:block flex-shrink-0 w-[200px] lg:w-[240px]">
              <div className="relative aspect-[2/3] rounded-lg overflow-hidden shadow-2xl">
                <Image
                  src={movie.poster_url}
                  alt={movie.title}
                  fill
                  className="object-cover"
                  sizes="240px"
                />
              </div>
            </div>
          )}

          {/* Info */}
          <div className="flex-1">
            <h1 className="text-3xl md:text-4xl lg:text-5xl font-bold text-white mb-3 leading-tight">
              {movie.title}
            </h1>

            {/* Meta row */}
            <div className="flex flex-wrap items-center gap-3 mb-4">
              {movie.year && (
                <span className="text-green-400 font-semibold">
                  {movie.year}
                </span>
              )}
              {movie.avg_rating && (
                <StarRating rating={movie.avg_rating} size={16} />
              )}
              {movie.num_ratings && (
                <span className="text-netflix-text-secondary text-sm">
                  {movie.num_ratings.toLocaleString()} ratings
                </span>
              )}
            </div>

            {/* Genres */}
            {movie.genres?.length > 0 && (
              <div className="flex flex-wrap gap-2 mb-5">
                {movie.genres.map((g) => (
                  <GenreBadge key={g} genre={g} />
                ))}
              </div>
            )}

            {/* Overview */}
            {movie.overview && (
              <p className="text-netflix-text-secondary leading-relaxed text-sm md:text-base mb-6 max-w-2xl">
                {movie.overview}
              </p>
            )}

            {/* Tags */}
            {movie.tags?.length > 0 && (
              <div className="mb-6">
                <h3 className="text-white text-sm font-medium mb-2">Tags</h3>
                <div className="flex flex-wrap gap-1.5">
                  {movie.tags.slice(0, 15).map((tag) => (
                    <span
                      key={tag}
                      className="text-xs bg-netflix-hover text-netflix-text-secondary
                                 px-2 py-1 rounded"
                    >
                      {tag}
                    </span>
                  ))}
                </div>
              </div>
            )}

            {/* Links */}
            <div className="flex flex-wrap gap-3">
              {imdbUrl && (
                <a
                  href={imdbUrl}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="flex items-center gap-2 bg-yellow-500/20 hover:bg-yellow-500/30
                             text-yellow-400 font-medium px-4 py-2 rounded-md text-sm
                             transition-colors"
                >
                  <FiExternalLink size={14} />
                  View on IMDb
                </a>
              )}
              {movie.tmdb_id && (
                <a
                  href={`https://www.themoviedb.org/movie/${movie.tmdb_id}`}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="flex items-center gap-2 bg-blue-500/20 hover:bg-blue-500/30
                             text-blue-400 font-medium px-4 py-2 rounded-md text-sm
                             transition-colors"
                >
                  <FiExternalLink size={14} />
                  TMDb
                </a>
              )}
            </div>
          </div>
        </div>
      </motion.div>
    </div>
  );
}
