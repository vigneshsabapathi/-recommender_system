import { create } from "zustand";
import { persist } from "zustand/middleware";

const SAMPLE_USERS = [
  { id: 1, name: "Action Fan", avatar: "\uD83C\uDFAC", color: "#E50914" },
  { id: 42, name: "Comedy Lover", avatar: "\uD83D\uDE02", color: "#FFD700" },
  { id: 999, name: "Sci-Fi Nerd", avatar: "\uD83D\uDE80", color: "#00D4FF" },
  { id: 0, name: "New User", avatar: "\uD83D\uDC64", color: "#808080" },
];

const useUserStore = create(
  persist(
    (set) => ({
      userId: null,
      userName: null,
      sampleUsers: SAMPLE_USERS,
      setUser: (id, name) => set({ userId: id, userName: name }),
      clearUser: () => set({ userId: null, userName: null }),
    }),
    {
      name: "movierec-user",
    }
  )
);

export default useUserStore;
