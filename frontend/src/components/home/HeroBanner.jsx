"use client";

import { useState, useEffect } from "react";
import Image from "next/image";
import Link from "next/link";
import { motion, AnimatePresence } from "framer-motion";
import { FiInfo, FiPlay } from "react-icons/fi";
import GenreBadge from "@/components/ui/GenreBadge";
import StarRating from "@/components/ui/StarRating";
import { HeroBannerSkeleton } from "@/components/ui/Skeleton";

// Rich gradients for movies without backdrop images
const HERO_GRADIENTS = [
  "linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%)",
  "linear-gradient(135deg, #2d1b2e 0%, #1a1a2e 50%, #0d1b2a 100%)",
  "linear-gradient(135deg, #1b2838 0%, #171a21 50%, #1b2838 100%)",
  "linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%)",
  "linear-gradient(135deg, #141e30 0%, #243b55 50%, #141e30 100%)",
];

export default function HeroBanner({ movies, isLoading }) {
  const [currentIndex, setCurrentIndex] = useState(0);

  // Auto-rotate featured movie every 8s
  const featured = movies?.slice(0, 5) || [];
  const movie = featured[currentIndex];

  useEffect(() => {
    if (featured.length <= 1) return;
    const interval = setInterval(() => {
      setCurrentIndex((prev) => (prev + 1) % featured.length);
    }, 8000);
    return () => clearInterval(interval);
  }, [featured.length]);

  if (isLoading || !movie) return <HeroBannerSkeleton />;

  // For recommendations, movie data is nested under .movie
  const movieData = movie.movie || movie;
  const backdropUrl = movieData.backdrop_url || movieData.poster_url;
  const gradient = HERO_GRADIENTS[currentIndex % HERO_GRADIENTS.length];
  const truncatedOverview = movieData.overview
    ? movieData.overview.length > 200
      ? movieData.overview.slice(0, 200) + "..."
      : movieData.overview
    : "Discover your next favorite movie with our AI-powered recommendation engine.";

  return (
    <div className="relative w-full h-[70vh] md:h-[80vh] overflow-hidden">
      {/* Background */}
      <AnimatePresence mode="wait">
        <motion.div
          key={movieData.id}
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          transition={{ duration: 1 }}
          className="absolute inset-0"
        >
          {backdropUrl ? (
            <Image
              src={backdropUrl}
              alt={movieData.title}
              fill
              className="object-cover"
              priority
              sizes="100vw"
            />
          ) : (
            <div
              className="absolute inset-0"
              style={{ background: gradient }}
            />
          )}
        </motion.div>
      </AnimatePresence>

      {/* Left gradient overlay */}
      <div className="absolute inset-0 hero-gradient z-[1]" />
      {/* Bottom gradient overlay */}
      <div className="absolute inset-0 hero-gradient-bottom z-[1]" />

      {/* Content */}
      <AnimatePresence mode="wait">
        <motion.div
          key={movieData.id}
          initial={{ opacity: 0, y: 30 }}
          animate={{ opacity: 1, y: 0 }}
          exit={{ opacity: 0, y: -20 }}
          transition={{ duration: 0.6, delay: 0.2 }}
          className="absolute bottom-16 md:bottom-24 left-4 md:left-12 z-10 max-w-xl lg:max-w-2xl"
        >
          {/* Title */}
          <h1 className="text-3xl md:text-5xl lg:text-6xl font-bold text-white mb-3 leading-tight drop-shadow-lg">
            {movieData.title}
          </h1>

          {/* Year + Rating */}
          <div className="flex items-center gap-3 mb-3">
            {movieData.year && (
              <span className="text-green-400 font-semibold text-sm">
                {movieData.year}
              </span>
            )}
            {movieData.avg_rating && (
              <StarRating rating={movieData.avg_rating} size={14} />
            )}
            {movie.predicted_rating && (
              <span className="text-xs bg-netflix-red/80 px-2 py-0.5 rounded text-white font-medium">
                Predicted: {movie.predicted_rating.toFixed(1)}
              </span>
            )}
          </div>

          {/* Genre Badges */}
          {movieData.genres?.length > 0 && (
            <div className="flex flex-wrap gap-2 mb-4">
              {movieData.genres.slice(0, 4).map((g) => (
                <GenreBadge key={g} genre={g} />
              ))}
            </div>
          )}

          {/* Description */}
          <p className="text-sm md:text-base text-netflix-text-secondary leading-relaxed mb-5 line-clamp-3">
            {truncatedOverview}
          </p>

          {/* Buttons */}
          <div className="flex items-center gap-3">
            <Link
              href={`/movie/${movieData.id}`}
              className="flex items-center gap-2 bg-white/90 hover:bg-white text-black
                         font-semibold px-6 py-2.5 rounded-md text-sm transition-all
                         hover:scale-105 active:scale-95"
            >
              <FiInfo size={18} />
              More Info
            </Link>
            <Link
              href={`/movie/${movieData.id}#similar`}
              className="flex items-center gap-2 bg-white/20 hover:bg-white/30 text-white
                         font-semibold px-6 py-2.5 rounded-md text-sm transition-all
                         backdrop-blur-sm hover:scale-105 active:scale-95"
            >
              <FiPlay size={18} />
              Similar Movies
            </Link>
          </div>
        </motion.div>
      </AnimatePresence>

      {/* Indicator dots */}
      {featured.length > 1 && (
        <div className="absolute bottom-6 right-4 md:right-12 z-10 flex gap-1.5">
          {featured.map((_, i) => (
            <button
              key={i}
              onClick={() => setCurrentIndex(i)}
              className={`w-3 h-1 rounded-full transition-all duration-300 ${
                i === currentIndex
                  ? "bg-white w-6"
                  : "bg-white/40 hover:bg-white/60"
              }`}
              aria-label={`Show featured movie ${i + 1}`}
            />
          ))}
        </div>
      )}
    </div>
  );
}
