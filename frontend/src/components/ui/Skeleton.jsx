"use client";

export function Skeleton({ className = "", style = {} }) {
  return (
    <div
      className={`skeleton-shimmer rounded-md ${className}`}
      style={style}
    />
  );
}

export function MovieCardSkeleton() {
  return (
    <div className="flex-shrink-0 w-[150px] sm:w-[180px] md:w-[200px] lg:w-[220px]">
      <Skeleton className="w-full aspect-[2/3] rounded-md" />
      <Skeleton className="w-3/4 h-3 mt-2 rounded" />
      <Skeleton className="w-1/2 h-3 mt-1 rounded" />
    </div>
  );
}

export function MovieRowSkeleton({ count = 6 }) {
  return (
    <div className="px-4 md:px-12 mb-8">
      <Skeleton className="w-48 h-6 mb-4 rounded" />
      <div className="flex gap-2 overflow-hidden">
        {Array.from({ length: count }).map((_, i) => (
          <MovieCardSkeleton key={i} />
        ))}
      </div>
    </div>
  );
}

export function HeroBannerSkeleton() {
  return (
    <div className="relative w-full h-[70vh] md:h-[80vh]">
      <Skeleton className="absolute inset-0 rounded-none" />
      <div className="absolute bottom-20 left-4 md:left-12 z-10 space-y-4">
        <Skeleton className="w-72 h-10 rounded" />
        <Skeleton className="w-96 h-4 rounded" />
        <Skeleton className="w-80 h-4 rounded" />
        <div className="flex gap-3 mt-4">
          <Skeleton className="w-32 h-10 rounded-md" />
          <Skeleton className="w-40 h-10 rounded-md" />
        </div>
      </div>
    </div>
  );
}

export function MovieDetailSkeleton() {
  return (
    <div className="min-h-screen bg-netflix-bg">
      <Skeleton className="w-full h-[50vh] md:h-[60vh] rounded-none" />
      <div className="px-4 md:px-12 -mt-32 relative z-10 space-y-4">
        <Skeleton className="w-80 h-10 rounded" />
        <div className="flex gap-3">
          <Skeleton className="w-16 h-6 rounded-full" />
          <Skeleton className="w-16 h-6 rounded-full" />
          <Skeleton className="w-16 h-6 rounded-full" />
        </div>
        <Skeleton className="w-full max-w-2xl h-4 rounded" />
        <Skeleton className="w-full max-w-xl h-4 rounded" />
        <Skeleton className="w-2/3 max-w-lg h-4 rounded" />
      </div>
    </div>
  );
}
