"use client";

const GENRE_COLORS = {
  Action: "bg-red-900/60 text-red-300",
  Adventure: "bg-orange-900/60 text-orange-300",
  Animation: "bg-teal-900/60 text-teal-300",
  Children: "bg-pink-900/60 text-pink-300",
  Comedy: "bg-yellow-900/60 text-yellow-300",
  Crime: "bg-slate-700/60 text-slate-300",
  Documentary: "bg-blue-900/60 text-blue-300",
  Drama: "bg-purple-900/60 text-purple-300",
  Fantasy: "bg-violet-900/60 text-violet-300",
  "Film-Noir": "bg-gray-700/60 text-gray-300",
  Horror: "bg-emerald-900/60 text-emerald-300",
  Musical: "bg-fuchsia-900/60 text-fuchsia-300",
  Mystery: "bg-indigo-900/60 text-indigo-300",
  Romance: "bg-rose-900/60 text-rose-300",
  "Sci-Fi": "bg-cyan-900/60 text-cyan-300",
  Thriller: "bg-amber-900/60 text-amber-300",
  War: "bg-stone-700/60 text-stone-300",
  Western: "bg-orange-800/60 text-orange-200",
  IMAX: "bg-blue-800/60 text-blue-200",
};

const DEFAULT_COLOR = "bg-netflix-hover text-netflix-text-secondary";

export default function GenreBadge({ genre, onClick, small = false }) {
  const colorClass = GENRE_COLORS[genre] || DEFAULT_COLOR;
  const sizeClass = small
    ? "px-2 py-0.5 text-[10px]"
    : "px-2.5 py-1 text-xs";

  return (
    <span
      className={`inline-block rounded-full font-medium whitespace-nowrap ${colorClass} ${sizeClass} ${
        onClick ? "cursor-pointer hover:brightness-125 transition-all" : ""
      }`}
      onClick={onClick}
      role={onClick ? "button" : undefined}
    >
      {genre}
    </span>
  );
}
