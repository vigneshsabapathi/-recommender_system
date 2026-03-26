import { create } from "zustand";

const ALGORITHMS = [
  {
    id: "hybrid",
    name: "Hybrid",
    desc: "Best of all approaches combined",
    icon: "layers",
  },
  {
    id: "collaborative",
    name: "Collaborative",
    desc: "Users like you also liked...",
    icon: "users",
  },
  {
    id: "content_based",
    name: "Content-Based",
    desc: "Based on movie features & genres",
    icon: "film",
  },
  {
    id: "als",
    name: "Matrix Factorization",
    desc: "Latent factor model (ALS)",
    icon: "grid",
  },
];

const useAlgorithmStore = create((set) => ({
  algorithm: "hybrid",
  algorithms: ALGORITHMS,
  setAlgorithm: (algo) => set({ algorithm: algo }),
}));

export default useAlgorithmStore;
