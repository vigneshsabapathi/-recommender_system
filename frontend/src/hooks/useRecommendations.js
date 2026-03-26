"use client";

import { useQuery } from "@tanstack/react-query";
import { fetchRecommendations } from "@/lib/api";
import useUserStore from "@/stores/userStore";
import useAlgorithmStore from "@/stores/algorithmStore";

export function useRecommendations(overrides = {}) {
  const userId = useUserStore((s) => s.userId);
  const algorithm = useAlgorithmStore((s) => s.algorithm);

  const finalUserId = overrides.userId ?? userId;
  const finalAlgorithm = overrides.algorithm ?? algorithm;
  const n = overrides.n ?? 20;

  return useQuery({
    queryKey: ["recommendations", finalUserId, finalAlgorithm, n],
    queryFn: () =>
      fetchRecommendations({
        userId: finalUserId,
        algorithm: finalAlgorithm,
        n,
        explain: true,
      }),
    staleTime: 5 * 60 * 1000,
    enabled: finalUserId != null,
    retry: 2,
  });
}
