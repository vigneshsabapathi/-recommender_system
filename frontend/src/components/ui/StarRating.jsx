"use client";

import { FaStar, FaStarHalfAlt, FaRegStar } from "react-icons/fa";

export default function StarRating({ rating, size = 14, className = "" }) {
  if (rating == null) return null;

  const stars = [];
  const fullStars = Math.floor(rating);
  const hasHalf = rating - fullStars >= 0.25 && rating - fullStars < 0.75;
  const totalFull = hasHalf ? fullStars : Math.round(rating);

  for (let i = 0; i < 5; i++) {
    if (i < fullStars) {
      stars.push(
        <FaStar key={i} size={size} className="text-yellow-400" />
      );
    } else if (i === fullStars && hasHalf) {
      stars.push(
        <FaStarHalfAlt key={i} size={size} className="text-yellow-400" />
      );
    } else {
      stars.push(
        <FaRegStar key={i} size={size} className="text-yellow-400/40" />
      );
    }
  }

  return (
    <div className={`flex items-center gap-0.5 ${className}`}>
      {stars}
      <span className="ml-1.5 text-xs text-netflix-text-secondary">
        {rating.toFixed(1)}
      </span>
    </div>
  );
}
