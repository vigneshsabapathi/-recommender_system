"use client";

import { FiGithub, FiExternalLink } from "react-icons/fi";

export default function Footer() {
  return (
    <footer className="mt-20 border-t border-netflix-border/30 bg-netflix-bg">
      <div className="max-w-7xl mx-auto px-4 md:px-12 py-10">
        <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
          {/* Brand */}
          <div>
            <div className="flex items-center gap-1 mb-3">
              <span className="text-netflix-red text-xl font-black tracking-tighter">
                MOVIE
              </span>
              <span className="text-white text-xl font-light tracking-tighter">
                REC
              </span>
            </div>
            <p className="text-netflix-text-secondary text-sm leading-relaxed">
              A movie recommender system built with collaborative filtering,
              content-based filtering, matrix factorization (ALS), and hybrid
              approaches.
            </p>
          </div>

          {/* Tech Stack */}
          <div>
            <h4 className="text-white font-medium text-sm mb-3">Tech Stack</h4>
            <ul className="space-y-1.5 text-netflix-text-secondary text-sm">
              <li>React 19 + Next.js 15</li>
              <li>FastAPI + Python</li>
              <li>Spark ALS + scikit-learn</li>
              <li>MovieLens 25M Dataset</li>
            </ul>
          </div>

          {/* Links */}
          <div>
            <h4 className="text-white font-medium text-sm mb-3">Links</h4>
            <ul className="space-y-1.5">
              <li>
                <a
                  href="https://github.com"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="flex items-center gap-2 text-netflix-text-secondary text-sm
                             hover:text-white transition-colors"
                >
                  <FiGithub size={14} />
                  GitHub Repository
                </a>
              </li>
              <li>
                <a
                  href="https://grouplens.org/datasets/movielens/"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="flex items-center gap-2 text-netflix-text-secondary text-sm
                             hover:text-white transition-colors"
                >
                  <FiExternalLink size={14} />
                  MovieLens Dataset
                </a>
              </li>
            </ul>
          </div>
        </div>

        <div className="mt-8 pt-6 border-t border-netflix-border/20 flex flex-col md:flex-row items-center justify-between gap-4">
          <p className="text-netflix-text-secondary/60 text-xs">
            This product uses the TMDb API but is not endorsed or certified by TMDb.
            Movie data from MovieLens 25M research dataset.
          </p>
          <p className="text-netflix-text-secondary/60 text-xs">
            Portfolio project &mdash; not affiliated with Netflix.
          </p>
        </div>
      </div>
    </footer>
  );
}
