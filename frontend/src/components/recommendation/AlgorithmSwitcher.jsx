"use client";

import { motion } from "framer-motion";
import { FiLayers, FiUsers, FiFilm, FiGrid } from "react-icons/fi";
import useAlgorithmStore from "@/stores/algorithmStore";
import { Button } from "@/components/ui/button";

const ICONS = {
  layers: FiLayers,
  users: FiUsers,
  film: FiFilm,
  grid: FiGrid,
};

export default function AlgorithmSwitcher() {
  const algorithm = useAlgorithmStore((s) => s.algorithm);
  const algorithms = useAlgorithmStore((s) => s.algorithms);
  const setAlgorithm = useAlgorithmStore((s) => s.setAlgorithm);

  return (
    <div className="px-4 md:px-12 mb-6">
      <div className="flex items-center gap-3 mb-3">
        <h3 className="text-sm font-medium text-netflix-text-secondary">
          Recommendation Engine
        </h3>
        <div className="h-px flex-1 bg-netflix-border/30" />
      </div>

      <div className="flex flex-wrap gap-2">
        {algorithms.map((algo) => {
          const Icon = ICONS[algo.icon] || FiLayers;
          const isActive = algorithm === algo.id;

          return (
            <div key={algo.id} className="relative group">
              <motion.div
                whileHover={{ scale: 1.03 }}
                whileTap={{ scale: 0.97 }}
              >
                <Button
                  variant={isActive ? "default" : "outline"}
                  onClick={() => setAlgorithm(algo.id)}
                  className={
                    isActive
                      ? "gap-2 bg-netflix-red border-netflix-red text-white shadow-lg shadow-netflix-red/20 hover:bg-netflix-red-hover"
                      : "gap-2 bg-netflix-card border-netflix-border/40 text-netflix-text-secondary hover:border-white/30 hover:text-white"
                  }
                >
                  <Icon size={14} />
                  <span>{algo.name}</span>
                </Button>
              </motion.div>

              {/* Tooltip */}
              <div
                className="absolute left-1/2 -translate-x-1/2 top-full mt-2 pointer-events-none
                           opacity-0 group-hover:opacity-100 transition-opacity duration-200 z-30"
              >
                <div className="bg-netflix-dark border border-netflix-border/50 text-white text-xs
                                px-3 py-2 rounded-md shadow-xl whitespace-nowrap">
                  {algo.desc}
                  <div className="absolute -top-1 left-1/2 -translate-x-1/2 w-2 h-2
                                  bg-netflix-dark border-l border-t border-netflix-border/50
                                  rotate-45" />
                </div>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
