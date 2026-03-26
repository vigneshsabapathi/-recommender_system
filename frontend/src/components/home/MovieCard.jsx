"use client";

import { useState } from "react";
import Image from "next/image";
import Link from "next/link";
import { motion } from "framer-motion";
import { FiInfo, FiStar } from "react-icons/fi";
import GenreBadge from "@/components/ui/GenreBadge";

// Gradient placeholders when poster is missing
const PLACEHOLDER_GRADIENTS = [
  "linear-gradient(135deg, #667eea 0%, #764ba2 100%)",
  "linear-gradient(135deg, #f093fb 0%, #f5576c 100%)",
  "linear-gradient(135deg, #4facfe 0%, #00f2fe 100%)",
  "linear-gradient(135deg, #43e97b 0%, #38f9d7 100%)",
  "linear-gradient(135deg, #fa709a 0%, #fee140 100%)",
  "linear-gradient(135deg, #a18cd1 0%, #fbc2eb 100%)",
  "linear-gradient(135deg, #fccb90 0%, #d57eeb 100%)",
  "linear-gradient(135deg, #e0c3fc 0%, #8ec5fc 100%)",
];

export default function MovieCard({ movie, index = 0, showRank = false }) {
  const [imageLoaded, setImageLoaded] = useState(false);
  const [imageError, setImageError] = useState(false);

  // Handle nested movie data from recommendation response
  const movieData = movie?.movie || movie;
  if (!movieData) return null;

  const posterUrl = movieData.poster_url;
  const hasPoster = posterUrl && !imageError;
  const gradient = PLACEHOLDER_GRADIENTS[index % PLACEHOLDER_GRADIENTS.length];

  return (
    <Link
      href={`/movie/${movieData.id}`}
      className="group relative flex-shrink-0 w-[150px] sm:w-[180px] md:w-[200px] lg:w-[220px]"
    >
      <motion.div
        whileHover={{ scale: 1.08, zIndex: 20 }}
        transition={{ duration: 0.3, ease: "easeOut" }}
        className="relative"
      >
        {/* Poster Container */}
        <div className="relative aspect-[2/3] rounded-md overflow-hidden bg-netflix-card shadow-lg">
          {hasPoster ? (
            <>
              {!imageLoaded && (
                <div className="absolute inset-0 skeleton-shimmer" />
              )}
              <Image
                src={posterUrl}
                alt={movieData.title}
                fill
                className={`object-cover transition-opacity duration-500 ${
                  imageLoaded ? "opacity-100" : "opacity-0"
                }`}
                sizes="(max-width: 640px) 150px, (max-width: 768px) 180px, (max-width: 1024px) 200px, 220px"
                loading="lazy"
                onLoad={() => setImageLoaded(true)}
                onError={() => setImageError(true)}
              />
            </>
          ) : (
            <div
              className="absolute inset-0 flex flex-col items-center justify-center p-3"
              style={{ background: gradient }}
            >
              <span className="text-white/90 text-center text-sm font-semibold leading-snug">
                {movieData.title}
              </span>
              {movieData.year && (
                <span className="text-white/60 text-xs mt-1">
                  ({movieData.year})
                </span>
              )}
            </div>
          )}

          {/* Rank number */}
          {showRank && (
            <div className="absolute top-0 left-0 bg-netflix-red text-white text-xs font-bold
                            w-6 h-6 flex items-center justify-center rounded-br-md">
              {index + 1}
            </div>
          )}

          {/* Hover overlay */}
          <div
            className="absolute inset-0 card-gradient opacity-0 group-hover:opacity-100
                        transition-opacity duration-300 flex flex-col justify-end p-3"
          >
            <h4 className="text-white text-sm font-semibold leading-tight mb-1 line-clamp-2">
              {movieData.title}
            </h4>
            <div className="flex items-center gap-2 mb-1.5">
              {movieData.year && (
                <span className="text-green-400 text-xs font-medium">
                  {movieData.year}
                </span>
              )}
              {movieData.avg_rating && (
                <span className="flex items-center gap-0.5 text-yellow-400 text-xs">
                  <FiStar size={10} />
                  {movieData.avg_rating.toFixed(1)}
                </span>
              )}
            </div>
            {movieData.genres?.length > 0 && (
              <div className="flex flex-wrap gap-1 mb-2">
                {movieData.genres.slice(0, 2).map((g) => (
                  <GenreBadge key={g} genre={g} small />
                ))}
              </div>
            )}
            <div
              className="flex items-center gap-1 text-white text-xs font-medium
                          bg-white/20 backdrop-blur-sm rounded px-2 py-1 w-fit
                          hover:bg-white/30 transition-colors"
            >
              <FiInfo size={12} />
              More Info
            </div>
          </div>
        </div>

        {/* Title below card (visible on non-hover for mobile) */}
        <div className="mt-1.5 group-hover:opacity-0 transition-opacity duration-200">
          <p className="text-netflix-text-secondary text-xs truncate">
            {movieData.title}
          </p>
        </div>
      </motion.div>
    </Link>
  );
}
