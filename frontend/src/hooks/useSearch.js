"use client";

import { useState, useEffect, useRef } from "react";
import { useQuery } from "@tanstack/react-query";
import { searchMovies } from "@/lib/api";

export function useSearch(initialQuery = "", limit = 20) {
  const [query, setQuery] = useState(initialQuery);
  const [debouncedQuery, setDebouncedQuery] = useState(initialQuery);
  const timerRef = useRef(null);

  useEffect(() => {
    if (timerRef.current) clearTimeout(timerRef.current);
    timerRef.current = setTimeout(() => {
      setDebouncedQuery(query);
    }, 300);
    return () => {
      if (timerRef.current) clearTimeout(timerRef.current);
    };
  }, [query]);

  const { data, isLoading, error } = useQuery({
    queryKey: ["search", debouncedQuery, limit],
    queryFn: () => searchMovies(debouncedQuery, limit),
    staleTime: 2 * 60 * 1000,
    enabled: debouncedQuery.trim().length >= 2,
  });

  return {
    query,
    setQuery,
    results: data || [],
    isLoading: isLoading && debouncedQuery.trim().length >= 2,
    error,
  };
}
