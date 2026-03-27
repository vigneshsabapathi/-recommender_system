"use client";

import { useState, useEffect, useRef, useCallback } from "react";
import { useRouter } from "next/navigation";
import Image from "next/image";
import { FiSearch, FiLoader } from "react-icons/fi";
import { searchMovies } from "@/lib/api";
import { cn } from "@/lib/utils";
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "@/components/ui/popover";
import {
  Command,
  CommandInput,
  CommandList,
  CommandEmpty,
  CommandGroup,
  CommandItem,
} from "@/components/ui/command";

export default function SearchAutocomplete() {
  const router = useRouter();
  const [open, setOpen] = useState(false);
  const [query, setQuery] = useState("");
  const [debouncedQuery, setDebouncedQuery] = useState("");
  const [results, setResults] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const timerRef = useRef(null);

  // Debounce the query (300ms)
  useEffect(() => {
    if (timerRef.current) clearTimeout(timerRef.current);
    timerRef.current = setTimeout(() => {
      setDebouncedQuery(query);
    }, 300);
    return () => {
      if (timerRef.current) clearTimeout(timerRef.current);
    };
  }, [query]);

  // Fetch results when debounced query changes
  useEffect(() => {
    if (debouncedQuery.trim().length < 2) {
      setResults([]);
      setIsLoading(false);
      return;
    }

    let cancelled = false;
    setIsLoading(true);

    searchMovies(debouncedQuery, 6)
      .then((data) => {
        if (!cancelled) {
          setResults(data || []);
          setIsLoading(false);
        }
      })
      .catch(() => {
        if (!cancelled) {
          setResults([]);
          setIsLoading(false);
        }
      });

    return () => {
      cancelled = true;
    };
  }, [debouncedQuery]);

  const close = useCallback(() => {
    setOpen(false);
    setQuery("");
    setDebouncedQuery("");
    setResults([]);
  }, []);

  const handleSelectMovie = useCallback(
    (movieId) => {
      close();
      router.push(`/movie/${movieId}`);
    },
    [close, router]
  );

  const handleSearchAll = useCallback(() => {
    if (query.trim()) {
      close();
      router.push(`/search?q=${encodeURIComponent(query.trim())}`);
    }
  }, [query, close, router]);

  const handleKeyDown = useCallback(
    (e) => {
      if (e.key === "Escape") {
        e.preventDefault();
        close();
      }
      if (e.key === "Enter" && query.trim()) {
        // If no cmdk item is actively selected, navigate to search page
        const selected = document.querySelector(
          "[cmdk-item][data-selected=true]"
        );
        if (!selected) {
          e.preventDefault();
          handleSearchAll();
        }
      }
    },
    [query, close, handleSearchAll]
  );

  return (
    <Popover open={open} onOpenChange={setOpen}>
      <PopoverTrigger
        className="text-white hover:text-netflix-text-secondary transition-colors"
        aria-label="Search"
      >
        <FiSearch size={20} />
      </PopoverTrigger>

      <PopoverContent
        align="end"
        sideOffset={8}
        className="w-[380px] p-0 bg-netflix-dark border-netflix-border/50"
      >
        <Command
          shouldFilter={false}
          className="bg-transparent"
          onKeyDown={handleKeyDown}
        >
          <CommandInput
            placeholder="Search movies..."
            value={query}
            onValueChange={setQuery}
          />
          <CommandList className="max-h-[360px]">
            {query.trim().length >= 2 && !isLoading && results.length === 0 && (
              <CommandEmpty className="text-netflix-text-secondary">
                No movies found.
              </CommandEmpty>
            )}

            {isLoading && query.trim().length >= 2 && (
              <div className="flex items-center justify-center gap-2 py-6 text-sm text-netflix-text-secondary">
                <FiLoader className="animate-spin" size={16} />
                Searching...
              </div>
            )}

            {results.length > 0 && (
              <CommandGroup>
                {results.map((movie) => (
                  <CommandItem
                    key={movie.id}
                    value={String(movie.id)}
                    onSelect={() => handleSelectMovie(movie.id)}
                    className="flex items-center gap-3 px-2 py-1.5 cursor-pointer data-selected:bg-white/10"
                  >
                    <div className="relative w-[40px] h-[60px] flex-shrink-0 rounded overflow-hidden bg-netflix-card">
                      {movie.poster_url ? (
                        <Image
                          src={movie.poster_url}
                          alt={movie.title}
                          width={40}
                          height={60}
                          className="object-cover"
                        />
                      ) : (
                        <div className="w-full h-full flex items-center justify-center text-netflix-text-secondary text-xs">
                          N/A
                        </div>
                      )}
                    </div>
                    <div className="flex-1 min-w-0">
                      <p className="text-white text-sm font-medium truncate">
                        {movie.title}
                      </p>
                      <div className="flex items-center gap-2 mt-0.5">
                        {movie.year && (
                          <span className="text-netflix-text-secondary text-xs">
                            {movie.year}
                          </span>
                        )}
                        {movie.genres && movie.genres.length > 0 && (
                          <span className="text-netflix-text-secondary/60 text-xs truncate">
                            {movie.genres.slice(0, 3).join(", ")}
                          </span>
                        )}
                      </div>
                    </div>
                  </CommandItem>
                ))}
              </CommandGroup>
            )}

            {query.trim().length >= 2 && !isLoading && (
              <CommandGroup>
                <CommandItem
                  onSelect={handleSearchAll}
                  className="justify-center text-netflix-text-secondary hover:text-white cursor-pointer data-selected:bg-white/10"
                >
                  <FiSearch size={14} className="mr-1.5" />
                  Search for &ldquo;{query.trim()}&rdquo;...
                </CommandItem>
              </CommandGroup>
            )}
          </CommandList>
        </Command>
      </PopoverContent>
    </Popover>
  );
}
