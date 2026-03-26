"use client";

import { useState } from "react";
import { motion } from "framer-motion";
import { FiCheck, FiArrowRight } from "react-icons/fi";
import GenreBadge from "@/components/ui/GenreBadge";
import useUserStore from "@/stores/userStore";

const GENRE_LIST = [
  "Action",
  "Adventure",
  "Animation",
  "Comedy",
  "Crime",
  "Documentary",
  "Drama",
  "Fantasy",
  "Horror",
  "Musical",
  "Mystery",
  "Romance",
  "Sci-Fi",
  "Thriller",
  "War",
  "Western",
];

export default function ColdStartWizard({ onComplete }) {
  const [selectedGenres, setSelectedGenres] = useState([]);
  const [step, setStep] = useState(1);
  const setUser = useUserStore((s) => s.setUser);

  const toggleGenre = (genre) => {
    setSelectedGenres((prev) =>
      prev.includes(genre)
        ? prev.filter((g) => g !== genre)
        : [...prev, genre]
    );
  };

  const handleGetRecommendations = () => {
    // For the cold start user, we'll use user 0 and let the backend handle it
    setUser(0, "New User");
    if (onComplete) onComplete(selectedGenres);
  };

  return (
    <motion.div
      initial={{ opacity: 0, y: 40 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.6 }}
      className="max-w-3xl mx-auto px-4 py-12"
    >
      <div className="text-center mb-10">
        <h2 className="text-3xl md:text-4xl font-bold text-white mb-3">
          Welcome to MovieRec
        </h2>
        <p className="text-netflix-text-secondary text-lg">
          Tell us what you enjoy so we can personalize your experience.
        </p>
      </div>

      {step === 1 && (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.2 }}
        >
          <h3 className="text-white font-semibold text-lg mb-4">
            Select genres you enjoy
          </h3>
          <div className="flex flex-wrap gap-3 mb-8">
            {GENRE_LIST.map((genre) => {
              const isSelected = selectedGenres.includes(genre);
              return (
                <motion.button
                  key={genre}
                  onClick={() => toggleGenre(genre)}
                  whileHover={{ scale: 1.05 }}
                  whileTap={{ scale: 0.95 }}
                  className={`relative px-5 py-2.5 rounded-full text-sm font-medium
                             border transition-all duration-200 ${
                               isSelected
                                 ? "bg-netflix-red border-netflix-red text-white"
                                 : "bg-netflix-card border-netflix-border/40 text-netflix-text-secondary hover:border-white/40 hover:text-white"
                             }`}
                >
                  <span className="flex items-center gap-1.5">
                    {isSelected && <FiCheck size={14} />}
                    {genre}
                  </span>
                </motion.button>
              );
            })}
          </div>

          <div className="text-center">
            <motion.button
              onClick={handleGetRecommendations}
              disabled={selectedGenres.length === 0}
              whileHover={{ scale: 1.03 }}
              whileTap={{ scale: 0.97 }}
              className="inline-flex items-center gap-2 bg-netflix-red hover:bg-netflix-red-hover
                         disabled:bg-netflix-card disabled:text-netflix-text-secondary/50
                         text-white font-semibold px-8 py-3 rounded-md text-sm
                         transition-colors disabled:cursor-not-allowed"
            >
              Get Recommendations
              <FiArrowRight size={16} />
            </motion.button>
            <p className="text-netflix-text-secondary/60 text-xs mt-3">
              Selected {selectedGenres.length} genre{selectedGenres.length !== 1 ? "s" : ""}
            </p>
          </div>
        </motion.div>
      )}
    </motion.div>
  );
}
