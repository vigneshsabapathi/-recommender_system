"use client";

import { useState } from "react";
import Image from "next/image";
import Link from "next/link";
import { motion } from "framer-motion";
import { FiStar } from "react-icons/fi";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { cn } from "@/lib/utils";

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

export default function MovieCard({
  movie,
  index = 0,
  showRank = false,
  isHovered = false,
  onHoverStart,
  onHoverEnd,
  isFirst = false,
  isLast = false,
  hoveredIndex,
}) {
  const [imageLoaded, setImageLoaded] = useState(false);
  const [imageError, setImageError] = useState(false);

  // Handle nested movie data from recommendation response
  const movieData = movie?.movie || movie;
  if (!movieData) return null;

  const posterUrl = movieData.poster_url;
  const hasPoster = posterUrl && !imageError;
  const gradient = PLACEHOLDER_GRADIENTS[index % PLACEHOLDER_GRADIENTS.length];

  // Compute neighbor shift when another card is hovered
  let neighborShiftX = 0;
  if (hoveredIndex !== null && hoveredIndex !== undefined && !isHovered) {
    if (index < hoveredIndex) {
      neighborShiftX = -25;
    } else if (index > hoveredIndex) {
      neighborShiftX = 25;
    }
  }

  // Determine transform-origin based on position
  const transformOrigin = isFirst ? "left" : isLast ? "right" : "center";

  // Match percentage (simulated from avg_rating if available)
  const matchScore = movieData.avg_rating
    ? Math.min(99, Math.round(movieData.avg_rating * 20))
    : null;

  return (
    <div
      className="relative flex-shrink-0 w-[150px] sm:w-[180px] md:w-[200px] lg:w-[220px]"
      onMouseEnter={onHoverStart}
      onMouseLeave={onHoverEnd}
      style={{ zIndex: isHovered ? 30 : 1 }}
    >
      <motion.div
        animate={{
          scale: isHovered ? 1.35 : 1,
          x: neighborShiftX,
        }}
        transition={{ duration: 0.3, ease: "easeOut" }}
        style={{ transformOrigin }}
      >
        <Link href={`/movie/${movieData.id}`} className="block">
          {/* Poster Container */}
          <div className="relative aspect-[2/3] rounded-t-md overflow-hidden bg-netflix-card shadow-lg">
            {hasPoster ? (
              <>
                {!imageLoaded && (
                  <div className="absolute inset-0 skeleton-shimmer" />
                )}
                <Image
                  src={posterUrl}
                  alt={movieData.title}
                  fill
                  className={cn(
                    "object-cover transition-opacity duration-500",
                    imageLoaded ? "opacity-100" : "opacity-0"
                  )}
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
              <div className="absolute top-0 left-0 bg-netflix-red text-white text-xs font-bold w-6 h-6 flex items-center justify-center rounded-br-md">
                {index + 1}
              </div>
            )}
          </div>
        </Link>

        {/* Title below poster (visible when not hovered) */}
        <motion.div
          animate={{ opacity: isHovered ? 0 : 1, height: isHovered ? 0 : "auto" }}
          transition={{ duration: 0.2 }}
          className="mt-1.5 overflow-hidden"
        >
          <p className="text-netflix-text-secondary text-xs truncate">
            {movieData.title}
          </p>
        </motion.div>

        {/* Hovered detail panel */}
        <motion.div
          animate={{
            opacity: isHovered ? 1 : 0,
            height: isHovered ? "auto" : 0,
          }}
          transition={{ duration: 0.3, ease: "easeOut" }}
          className="overflow-hidden bg-netflix-card rounded-b-md shadow-xl"
        >
          <div className="p-3 space-y-2">
            {/* Title */}
            <h4 className="text-white text-sm font-bold truncate">
              {movieData.title}
            </h4>

            {/* Year | Rating | Match */}
            <div className="flex items-center gap-2 text-xs flex-wrap">
              {movieData.year && (
                <span className="text-netflix-text-secondary">
                  {movieData.year}
                </span>
              )}
              {movieData.avg_rating && (
                <span className="flex items-center gap-0.5 text-yellow-400">
                  <FiStar size={10} />
                  {movieData.avg_rating.toFixed(1)}
                </span>
              )}
              {matchScore && (
                <span className="text-green-400 font-semibold">
                  {matchScore}% Match
                </span>
              )}
            </div>

            {/* Genre badges (max 3) */}
            {movieData.genres?.length > 0 && (
              <div className="flex flex-wrap gap-1">
                {movieData.genres.slice(0, 3).map((g) => (
                  <Badge
                    key={g}
                    variant="secondary"
                    className="text-[10px] px-1.5 py-0 h-4"
                  >
                    {g}
                  </Badge>
                ))}
              </div>
            )}

            {/* Action buttons */}
            <div className="flex gap-2 pt-1">
              <Button
                variant="default"
                size="xs"
                className="flex-1 text-[10px]"
                render={<Link href={`/movie/${movieData.id}`} />}
              >
                More Info
              </Button>
              <Button
                variant="outline"
                size="xs"
                className="flex-1 text-[10px]"
                render={<Link href={`/movie/${movieData.id}#similar`} />}
              >
                Similar
              </Button>
            </div>
          </div>
        </motion.div>
      </motion.div>
    </div>
  );
}
