"use client";

import { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { FiChevronDown, FiChevronUp, FiZap } from "react-icons/fi";

const ALGORITHM_LABELS = {
  collaborative: "Collaborative Filtering",
  content_based: "Content-Based",
  als: "Matrix Factorization (ALS)",
  hybrid: "Hybrid Ensemble",
};

const ALGORITHM_COLORS = {
  collaborative: "#E50914",
  content_based: "#00D4FF",
  als: "#FFD700",
  hybrid: "#43e97b",
};

export default function ExplanationPanel({ explanation, algorithm }) {
  const [isOpen, setIsOpen] = useState(false);

  if (!explanation || explanation.length === 0) return null;

  // Compute max score for scaling bars
  const maxScore = Math.max(...explanation.map((e) => Math.abs(e.score)), 0.01);

  return (
    <div className="mt-3">
      <button
        onClick={() => setIsOpen(!isOpen)}
        className="flex items-center gap-1.5 text-xs text-netflix-text-secondary
                   hover:text-white transition-colors"
      >
        <FiZap size={12} className="text-yellow-400" />
        Why this was recommended
        {isOpen ? <FiChevronUp size={14} /> : <FiChevronDown size={14} />}
      </button>

      <AnimatePresence>
        {isOpen && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: "auto", opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.25, ease: "easeInOut" }}
            className="overflow-hidden"
          >
            <div className="mt-2 p-3 bg-netflix-dark/60 border border-netflix-border/30 rounded-md space-y-2.5">
              {explanation.map((item, i) => {
                const barWidth = Math.max(
                  (Math.abs(item.score) / maxScore) * 100,
                  5
                );
                const color =
                  ALGORITHM_COLORS[item.algorithm] || "#B3B3B3";

                return (
                  <div key={i}>
                    <div className="flex items-center justify-between mb-1">
                      <span className="text-xs font-medium text-white">
                        {ALGORITHM_LABELS[item.algorithm] || item.algorithm}
                      </span>
                      <span className="text-xs text-netflix-text-secondary">
                        {item.score.toFixed(3)}
                      </span>
                    </div>

                    {/* Score bar */}
                    <div className="h-1.5 bg-netflix-hover rounded-full overflow-hidden">
                      <motion.div
                        initial={{ width: 0 }}
                        animate={{ width: `${barWidth}%` }}
                        transition={{ duration: 0.5, delay: i * 0.1 }}
                        className="h-full rounded-full"
                        style={{ backgroundColor: color }}
                      />
                    </div>

                    {/* Reason text */}
                    {item.reason && (
                      <p className="text-[11px] text-netflix-text-secondary/80 mt-1">
                        {item.reason}
                      </p>
                    )}
                  </div>
                );
              })}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
