"use client";

import { useRouter } from "next/navigation";
import { motion } from "framer-motion";
import useUserStore from "@/stores/userStore";

const container = {
  hidden: { opacity: 0 },
  show: {
    opacity: 1,
    transition: { staggerChildren: 0.1, delayChildren: 0.3 },
  },
};

const item = {
  hidden: { opacity: 0, y: 30 },
  show: { opacity: 1, y: 0, transition: { duration: 0.4 } },
};

export default function ProfilePicker() {
  const router = useRouter();
  const sampleUsers = useUserStore((s) => s.sampleUsers);
  const setUser = useUserStore((s) => s.setUser);

  const handleSelect = (user) => {
    setUser(user.id, user.name);
    router.push("/home");
  };

  return (
    <div className="min-h-screen bg-netflix-bg flex flex-col items-center justify-center px-4">
      {/* Logo */}
      <motion.div
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5 }}
        className="mb-10"
      >
        <div className="flex items-center gap-1">
          <span className="text-netflix-red text-4xl md:text-5xl font-black tracking-tighter">
            MOVIE
          </span>
          <span className="text-white text-4xl md:text-5xl font-light tracking-tighter">
            REC
          </span>
        </div>
      </motion.div>

      <motion.h1
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 0.2, duration: 0.5 }}
        className="text-3xl md:text-4xl text-white font-normal mb-8"
      >
        Who&apos;s watching?
      </motion.h1>

      <motion.div
        variants={container}
        initial="hidden"
        animate="show"
        className="grid grid-cols-2 md:grid-cols-4 gap-4 md:gap-6"
      >
        {sampleUsers.map((user) => (
          <motion.button
            key={user.id}
            variants={item}
            onClick={() => handleSelect(user)}
            whileHover={{ scale: 1.08 }}
            whileTap={{ scale: 0.95 }}
            className="flex flex-col items-center gap-3 group cursor-pointer"
          >
            <div
              className="w-24 h-24 md:w-32 md:h-32 rounded-md flex items-center justify-center
                         text-4xl md:text-5xl border-2 border-transparent
                         group-hover:border-white/80 transition-all duration-200
                         shadow-lg group-hover:shadow-xl"
              style={{ backgroundColor: user.color }}
            >
              {user.avatar}
            </div>
            <span className="text-netflix-text-secondary group-hover:text-white text-sm transition-colors">
              {user.name}
            </span>
          </motion.button>
        ))}
      </motion.div>

      <motion.p
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 0.8, duration: 0.5 }}
        className="mt-10 text-netflix-text-secondary/50 text-xs text-center max-w-md"
      >
        Each profile uses a different user from the MovieLens dataset to
        demonstrate how recommendations change based on viewing history.
      </motion.p>
    </div>
  );
}
