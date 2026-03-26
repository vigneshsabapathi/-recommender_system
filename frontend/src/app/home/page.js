"use client";

import { useEffect } from "react";
import { useRouter } from "next/navigation";
import HeroBanner from "@/components/home/HeroBanner";
import MovieRow from "@/components/home/MovieRow";
import AlgorithmSwitcher from "@/components/recommendation/AlgorithmSwitcher";
import ColdStartWizard from "@/components/recommendation/ColdStartWizard";
import { useRecommendations } from "@/hooks/useRecommendations";
import { useMovieList } from "@/hooks/useMovies";
import useUserStore from "@/stores/userStore";

export default function HomePage() {
  const router = useRouter();
  const userId = useUserStore((s) => s.userId);

  // Redirect to profile picker if no user selected
  useEffect(() => {
    if (userId === null) {
      router.push("/");
    }
  }, [userId, router]);

  // Recommendations for current user
  const {
    data: recData,
    isLoading: recLoading,
    error: recError,
    refetch: refetchRecs,
  } = useRecommendations();

  // Popular movies
  const {
    data: popularData,
    isLoading: popularLoading,
    error: popularError,
    refetch: refetchPopular,
  } = useMovieList({ sortBy: "num_ratings", perPage: 20 });

  // Top rated
  const {
    data: topRatedData,
    isLoading: topRatedLoading,
    error: topRatedError,
    refetch: refetchTopRated,
  } = useMovieList({ sortBy: "avg_rating", perPage: 20 });

  // Genre rows
  const {
    data: actionData,
    isLoading: actionLoading,
    error: actionError,
    refetch: refetchAction,
  } = useMovieList({ genre: "Action", perPage: 20 });

  const {
    data: comedyData,
    isLoading: comedyLoading,
    error: comedyError,
    refetch: refetchComedy,
  } = useMovieList({ genre: "Comedy", perPage: 20 });

  const {
    data: scifiData,
    isLoading: scifiLoading,
    error: scifiError,
    refetch: refetchScifi,
  } = useMovieList({ genre: "Sci-Fi", perPage: 20 });

  const {
    data: dramaData,
    isLoading: dramaLoading,
    error: dramaError,
    refetch: refetchDrama,
  } = useMovieList({ genre: "Drama", perPage: 20 });

  const {
    data: thrillerData,
    isLoading: thrillerLoading,
    error: thrillerError,
    refetch: refetchThriller,
  } = useMovieList({ genre: "Thriller", perPage: 20 });

  // Cold start wizard for new users
  if (userId === 0) {
    return (
      <div className="pt-16 bg-netflix-bg min-h-screen">
        <ColdStartWizard
          onComplete={() => {
            refetchRecs();
          }}
        />
        {/* Still show popular movies below */}
        <MovieRow
          title="Popular Right Now"
          movies={popularData?.items}
          isLoading={popularLoading}
          error={popularError}
          onRetry={refetchPopular}
        />
      </div>
    );
  }

  if (userId === null) return null;

  const recommendations = recData?.recommendations || [];

  return (
    <div className="bg-netflix-bg min-h-screen">
      {/* Hero Banner */}
      <HeroBanner
        movies={recommendations}
        isLoading={recLoading}
      />

      {/* Algorithm Switcher */}
      <div className="mt-6">
        <AlgorithmSwitcher />
      </div>

      {/* Recommendation Row */}
      <MovieRow
        title="Top Picks for You"
        movies={recommendations}
        isLoading={recLoading}
        error={recError}
        onRetry={refetchRecs}
        showRank
      />

      {/* Popular */}
      <MovieRow
        title="Popular on MovieRec"
        movies={popularData?.items}
        isLoading={popularLoading}
        error={popularError}
        onRetry={refetchPopular}
      />

      {/* Top Rated */}
      <MovieRow
        title="Critically Acclaimed"
        movies={topRatedData?.items}
        isLoading={topRatedLoading}
        error={topRatedError}
        onRetry={refetchTopRated}
      />

      {/* Genre Rows */}
      <MovieRow
        title="Action & Adventure"
        movies={actionData?.items}
        isLoading={actionLoading}
        error={actionError}
        onRetry={refetchAction}
      />

      <MovieRow
        title="Comedy"
        movies={comedyData?.items}
        isLoading={comedyLoading}
        error={comedyError}
        onRetry={refetchComedy}
      />

      <MovieRow
        title="Sci-Fi"
        movies={scifiData?.items}
        isLoading={scifiLoading}
        error={scifiError}
        onRetry={refetchScifi}
      />

      <MovieRow
        title="Drama"
        movies={dramaData?.items}
        isLoading={dramaLoading}
        error={dramaError}
        onRetry={refetchDrama}
      />

      <MovieRow
        title="Thriller"
        movies={thrillerData?.items}
        isLoading={thrillerLoading}
        error={thrillerError}
        onRetry={refetchThriller}
      />
    </div>
  );
}
