"use client";

import { useState, useEffect, useRef, useCallback } from "react";
import Image from "next/image";
import Link from "next/link";
import { motion, AnimatePresence } from "framer-motion";
import { FiInfo, FiPlay } from "react-icons/fi";
import GenreBadge from "@/components/ui/GenreBadge";
import StarRating from "@/components/ui/StarRating";
import { buttonVariants } from "@/components/ui/button";
import { cn } from "@/lib/utils";
import { HeroBannerSkeleton } from "@/components/ui/Skeleton";

// Rich gradients for movies without backdrop images
const HERO_GRADIENTS = [
  "linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%)",
  "linear-gradient(135deg, #2d1b2e 0%, #1a1a2e 50%, #0d1b2a 100%)",
  "linear-gradient(135deg, #1b2838 0%, #171a21 50%, #1b2838 100%)",
  "linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%)",
  "linear-gradient(135deg, #141e30 0%, #243b55 50%, #141e30 100%)",
];

const containerVariants = {
  hidden: { opacity: 0 },
  visible: {
    opacity: 1,
    transition: { staggerChildren: 0.08, delayChildren: 0.2 },
  },
};

const itemVariants = {
  hidden: { opacity: 0, y: 30 },
  visible: {
    opacity: 1,
    y: 0,
    transition: { duration: 0.5, ease: [0.25, 0.1, 0.25, 1] },
  },
};

export default function HeroBanner({ movies, isLoading }) {
  const [currentIndex, setCurrentIndex] = useState(0);
  const isPausedRef = useRef(false);
  const [isPaused, setIsPaused] = useState(false);

  const featured = movies?.slice(0, 5) || [];
  const movie = featured[currentIndex];

  const handleMouseEnter = useCallback(() => {
    isPausedRef.current = true;
    setIsPaused(true);
  }, []);

  const handleMouseLeave = useCallback(() => {
    isPausedRef.current = false;
    setIsPaused(false);
  }, []);

  // Auto-rotate featured movie every 8s, respecting pause
  useEffect(() => {
    if (featured.length <= 1) return;
    const interval = setInterval(() => {
      if (!isPausedRef.current) {
        setCurrentIndex((prev) => (prev + 1) % featured.length);
      }
    }, 8000);
    return () => clearInterval(interval);
  }, [featured.length]);

  if (isLoading || !movie) return <HeroBannerSkeleton />;

  // For recommendations, movie data is nested under .movie
  const movieData = movie.movie || movie;
  const backdropUrl = movieData.backdrop_url || movieData.poster_url;
  const gradient = HERO_GRADIENTS[currentIndex % HERO_GRADIENTS.length];

  const overview =
    movieData.overview ||
    "Discover your next favorite movie with our AI-powered recommendation engine.";

  return (
    <div
      className="relative w-full h-[70vh] md:h-[80vh] overflow-hidden"
      onMouseEnter={handleMouseEnter}
      onMouseLeave={handleMouseLeave}
    >
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
      {/* Vignette overlay */}
      <div className="absolute inset-0 hero-vignette z-[1]" />

      {/* Content */}
      <AnimatePresence mode="wait">
        <motion.div
          key={movieData.id}
          variants={containerVariants}
          initial="hidden"
          animate="visible"
          exit={{ opacity: 0, y: -20 }}
          className="absolute bottom-16 md:bottom-24 left-4 md:left-12 z-10 max-w-xl lg:max-w-2xl"
        >
          {/* Subtitle */}
          <motion.div variants={itemVariants}>
            <span className="text-xs md:text-sm font-semibold uppercase tracking-widest text-netflix-red mb-2 block">
              Recommended for You
            </span>
          </motion.div>

          {/* Title */}
          <motion.div variants={itemVariants}>
            <h1
              className="text-4xl md:text-6xl lg:text-7xl font-bold text-white mb-3 leading-tight tracking-tight"
              style={{ textShadow: "0 2px 40px rgba(0,0,0,0.8)" }}
            >
              {movieData.title}
            </h1>
          </motion.div>

          {/* Year + Rating */}
          <motion.div variants={itemVariants} className="flex items-center gap-3 mb-3">
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
          </motion.div>

          {/* Genre Badges */}
          {movieData.genres?.length > 0 && (
            <motion.div variants={itemVariants} className="flex flex-wrap gap-2 mb-4">
              {movieData.genres.slice(0, 4).map((g) => (
                <GenreBadge key={g} genre={g} />
              ))}
            </motion.div>
          )}

          {/* Description - CSS line-clamp only, no JS truncation */}
          <motion.div variants={itemVariants}>
            <p className="text-sm md:text-base text-netflix-text-secondary leading-relaxed mb-5 line-clamp-3">
              {overview}
            </p>
          </motion.div>

          {/* Buttons */}
          <motion.div variants={itemVariants} className="flex items-center gap-3">
            <Link
              href={`/movie/${movieData.id}`}
              className={cn(
                buttonVariants({ variant: "secondary", size: "lg" }),
                "gap-2 px-6 py-2.5 text-sm font-semibold hover:scale-105 active:scale-95 transition-transform"
              )}
            >
              <FiInfo size={18} />
              More Info
            </Link>
            <Link
              href={`/movie/${movieData.id}#similar`}
              className={cn(
                buttonVariants({ variant: "ghost", size: "lg" }),
                "gap-2 px-6 py-2.5 text-sm font-semibold border border-white/30 text-white backdrop-blur-sm hover:bg-white/10 hover:scale-105 active:scale-95 transition-transform"
              )}
            >
              <FiPlay size={18} />
              Similar Movies
            </Link>
          </motion.div>
        </motion.div>
      </AnimatePresence>

      {/* Progress bar indicators */}
      {featured.length > 1 && (
        <div className="absolute bottom-6 right-4 md:right-12 z-10 flex gap-1.5">
          {featured.map((_, i) => (
            <button
              key={i}
              onClick={() => setCurrentIndex(i)}
              className="h-1 rounded-full overflow-hidden bg-white/20 cursor-pointer"
              style={{ width: 40 }}
              aria-label={`Show featured movie ${i + 1}`}
            >
              {i === currentIndex ? (
                <div
                  className="h-full bg-netflix-red rounded-full"
                  style={{
                    animation: "progress-fill 8s linear forwards",
                    animationPlayState: isPaused ? "paused" : "running",
                  }}
                />
              ) : i < currentIndex ? (
                <div className="h-full bg-white/50 rounded-full w-full" />
              ) : null}
            </button>
          ))}
        </div>
      )}
    </div>
  );
}
