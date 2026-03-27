"use client";

import { motion } from "framer-motion";
import { FiGithub, FiDatabase, FiBarChart2, FiFilm } from "react-icons/fi";

const links = [
  {
    label: "GitHub",
    href: "https://github.com/vigneshsabapathi/-recommender_system",
    icon: FiGithub,
  },
  {
    label: "DagsHub",
    href: "https://dagshub.com/vigneshsabapathi/recommender_system",
    icon: FiDatabase,
  },
  {
    label: "MLflow",
    href: "https://dagshub.com/vigneshsabapathi/recommender_system.mlflow",
    icon: FiBarChart2,
  },
  {
    label: "MovieLens",
    href: "https://grouplens.org/datasets/movielens/",
    icon: FiFilm,
  },
];

export default function Footer() {
  return (
    <motion.footer
      initial={{ opacity: 0 }}
      whileInView={{ opacity: 1 }}
      viewport={{ once: true }}
      className="border-t border-white/5 mt-16"
    >
      {/* Links row */}
      <div className="max-w-6xl mx-auto px-6 py-8">
        <div className="flex flex-wrap justify-center gap-6 md:gap-10">
          {links.map((link) => (
            <a
              key={link.label}
              href={link.href}
              target="_blank"
              rel="noopener noreferrer"
              className="flex items-center gap-2 text-sm text-netflix-text-secondary hover:text-white transition-colors"
            >
              <link.icon size={16} />
              {link.label}
            </a>
          ))}
        </div>

        {/* TMDb Attribution */}
        <div className="flex items-center justify-center gap-3 mt-8 text-xs text-netflix-text-secondary/60">
          <span>Powered by</span>
          <span className="font-bold text-[#01d277]">TMDb</span>
          <span>API</span>
        </div>

        {/* Tech + Copyright */}
        <div className="text-center mt-6 space-y-2">
          <p className="text-xs text-netflix-text-secondary/40">
            Built with Next.js, FastAPI, PySpark ALS, scikit-learn
          </p>
          <p className="text-xs text-netflix-text-secondary/40">
            Portfolio project by Vignesh Sabapathi &middot; MovieLens 20M
            Dataset
          </p>
        </div>
      </div>
    </motion.footer>
  );
}
