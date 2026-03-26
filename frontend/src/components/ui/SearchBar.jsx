"use client";

import { useState, useRef, useEffect } from "react";
import { FiSearch, FiX } from "react-icons/fi";
import { motion, AnimatePresence } from "framer-motion";

export default function SearchBar({ value, onChange, onSubmit, placeholder = "Titles, genres, keywords..." }) {
  const [expanded, setExpanded] = useState(false);
  const inputRef = useRef(null);

  useEffect(() => {
    if (expanded && inputRef.current) {
      inputRef.current.focus();
    }
  }, [expanded]);

  const handleToggle = () => {
    if (expanded && value) {
      onChange("");
    }
    setExpanded(!expanded);
  };

  const handleKeyDown = (e) => {
    if (e.key === "Enter" && onSubmit) {
      onSubmit(value);
    }
    if (e.key === "Escape") {
      setExpanded(false);
      onChange("");
    }
  };

  return (
    <div className="relative flex items-center">
      <AnimatePresence mode="wait">
        {expanded && (
          <motion.div
            initial={{ width: 0, opacity: 0 }}
            animate={{ width: 260, opacity: 1 }}
            exit={{ width: 0, opacity: 0 }}
            transition={{ duration: 0.25, ease: "easeInOut" }}
            className="overflow-hidden"
          >
            <input
              ref={inputRef}
              type="text"
              value={value}
              onChange={(e) => onChange(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder={placeholder}
              className="w-full bg-netflix-dark/80 border border-netflix-border text-white text-sm
                         px-3 py-1.5 pr-8 outline-none placeholder:text-netflix-text-secondary/60
                         focus:border-white/50 transition-colors"
            />
          </motion.div>
        )}
      </AnimatePresence>

      <button
        onClick={handleToggle}
        className={`p-1.5 text-white hover:text-netflix-text-secondary transition-colors ${
          expanded ? "absolute right-0" : ""
        }`}
        aria-label={expanded ? "Close search" : "Open search"}
      >
        {expanded && value ? (
          <FiX size={20} />
        ) : (
          <FiSearch size={20} />
        )}
      </button>
    </div>
  );
}
