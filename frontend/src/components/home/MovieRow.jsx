"use client";

import { useRef, useState, useEffect } from "react";
import { motion } from "framer-motion";
import { FiChevronLeft, FiChevronRight } from "react-icons/fi";
import MovieCard from "./MovieCard";
import { MovieCardSkeleton } from "@/components/ui/Skeleton";
import ErrorState from "@/components/ui/ErrorState";

export default function MovieRow({
  title,
  movies,
  isLoading,
  error,
  onRetry,
  showRank = false,
}) {
  const scrollRef = useRef(null);
  const [canScrollLeft, setCanScrollLeft] = useState(false);
  const [canScrollRight, setCanScrollRight] = useState(true);
  const [isRowHovered, setIsRowHovered] = useState(false);
  const [hoveredIndex, setHoveredIndex] = useState(null);

  const updateScrollState = () => {
    const el = scrollRef.current;
    if (!el) return;
    setCanScrollLeft(el.scrollLeft > 10);
    setCanScrollRight(el.scrollLeft < el.scrollWidth - el.clientWidth - 10);
  };

  useEffect(() => {
    const el = scrollRef.current;
    if (!el) return;
    updateScrollState();
    el.addEventListener("scroll", updateScrollState, { passive: true });
    return () => el.removeEventListener("scroll", updateScrollState);
  }, [movies]);

  const scroll = (direction) => {
    const el = scrollRef.current;
    if (!el) return;
    const cardWidth = el.querySelector("div")?.offsetWidth || 220;
    const scrollAmount = cardWidth * 4;
    el.scrollBy({
      left: direction === "left" ? -scrollAmount : scrollAmount,
      behavior: "smooth",
    });
  };

  // Don't render row if no movies and not loading
  if (!isLoading && (!movies || movies.length === 0) && !error) return null;

  const movieCount = movies?.length || 0;

  return (
    <motion.section
      className="relative mb-8 md:mb-10"
      initial={{ opacity: 0, y: 20 }}
      whileInView={{ opacity: 1, y: 0 }}
      viewport={{ once: true, margin: "-50px" }}
      transition={{ duration: 0.5 }}
      onMouseEnter={() => setIsRowHovered(true)}
      onMouseLeave={() => {
        setIsRowHovered(false);
        setHoveredIndex(null);
      }}
    >
      {/* Title */}
      <h2 className="text-lg md:text-xl font-semibold text-white mb-3 px-4 md:px-12">
        {title}
      </h2>

      {error ? (
        <div className="px-4 md:px-12">
          <ErrorState message="Couldn't load this section." onRetry={onRetry} />
        </div>
      ) : (
        <div className="relative group">
          {/* Left arrow */}
          {canScrollLeft && (
            <button
              onClick={() => scroll("left")}
              className={`absolute left-0 top-0 bottom-6 z-20 w-10 md:w-14
                         bg-black/60 hover:bg-black/80 flex items-center justify-center
                         transition-opacity duration-300 ${
                           isRowHovered ? "opacity-100" : "opacity-0"
                         }`}
              aria-label="Scroll left"
            >
              <FiChevronLeft size={28} className="text-white" />
            </button>
          )}

          {/* Outer wrapper clips horizontal overflow, inner allows vertical expansion */}
          <div className="overflow-x-clip overflow-y-visible">
            <div
              ref={scrollRef}
              className="flex gap-2 md:gap-3 px-4 md:px-12 overflow-x-auto overflow-y-visible scrollbar-hide scroll-smooth py-8"
            >
              {isLoading
                ? Array.from({ length: 6 }).map((_, i) => (
                    <MovieCardSkeleton key={i} />
                  ))
                : movies?.map((movie, i) => (
                    <MovieCard
                      key={movie?.movie?.id || movie?.id || i}
                      movie={movie}
                      index={i}
                      showRank={showRank}
                      isHovered={hoveredIndex === i}
                      onHoverStart={() => setHoveredIndex(i)}
                      onHoverEnd={() => setHoveredIndex(null)}
                      isFirst={i === 0}
                      isLast={i === movieCount - 1}
                      hoveredIndex={hoveredIndex}
                    />
                  ))}
            </div>
          </div>

          {/* Right arrow */}
          {canScrollRight && (
            <button
              onClick={() => scroll("right")}
              className={`absolute right-0 top-0 bottom-6 z-20 w-10 md:w-14
                         bg-black/60 hover:bg-black/80 flex items-center justify-center
                         transition-opacity duration-300 ${
                           isRowHovered ? "opacity-100" : "opacity-0"
                         }`}
              aria-label="Scroll right"
            >
              <FiChevronRight size={28} className="text-white" />
            </button>
          )}
        </div>
      )}
    </motion.section>
  );
}
