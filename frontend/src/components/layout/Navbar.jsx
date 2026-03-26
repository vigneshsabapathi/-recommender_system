"use client";

import { useState, useEffect } from "react";
import Link from "next/link";
import { useRouter, usePathname } from "next/navigation";
import { motion } from "framer-motion";
import { FiSearch } from "react-icons/fi";
import useUserStore from "@/stores/userStore";

export default function Navbar() {
  const [scrolled, setScrolled] = useState(false);
  const [searchOpen, setSearchOpen] = useState(false);
  const [searchQuery, setSearchQuery] = useState("");
  const router = useRouter();
  const pathname = usePathname();
  const userId = useUserStore((s) => s.userId);
  const userName = useUserStore((s) => s.userName);
  const sampleUsers = useUserStore((s) => s.sampleUsers);

  const currentUser = sampleUsers.find((u) => u.id === userId);

  useEffect(() => {
    const handleScroll = () => setScrolled(window.scrollY > 10);
    window.addEventListener("scroll", handleScroll, { passive: true });
    return () => window.removeEventListener("scroll", handleScroll);
  }, []);

  const handleSearch = (e) => {
    if (e.key === "Enter" && searchQuery.trim()) {
      router.push(`/search?q=${encodeURIComponent(searchQuery.trim())}`);
      setSearchOpen(false);
    }
    if (e.key === "Escape") {
      setSearchOpen(false);
      setSearchQuery("");
    }
  };

  const navLinks = [
    { href: "/home", label: "Home" },
    { href: "/browse", label: "Browse" },
  ];

  // Don't show on profile picker page
  if (pathname === "/" || pathname === "/profile") return null;

  return (
    <motion.nav
      className={`fixed top-0 left-0 right-0 z-50 transition-all duration-500 ${
        scrolled
          ? "bg-netflix-bg/95 backdrop-blur-sm shadow-lg"
          : "bg-gradient-to-b from-black/80 to-transparent"
      }`}
      initial={{ y: -80 }}
      animate={{ y: 0 }}
      transition={{ duration: 0.3 }}
    >
      <div className="flex items-center justify-between px-4 md:px-12 h-16">
        {/* Left: Logo + Nav links */}
        <div className="flex items-center gap-8">
          <Link href="/home" className="flex items-center gap-1 group">
            <span className="text-netflix-red text-2xl font-black tracking-tighter">
              MOVIE
            </span>
            <span className="text-white text-2xl font-light tracking-tighter">
              REC
            </span>
          </Link>

          <div className="hidden md:flex items-center gap-6">
            {navLinks.map((link) => (
              <Link
                key={link.href}
                href={link.href}
                className={`text-sm font-medium transition-colors hover:text-white ${
                  pathname === link.href
                    ? "text-white"
                    : "text-netflix-text-secondary"
                }`}
              >
                {link.label}
              </Link>
            ))}
          </div>
        </div>

        {/* Right: Search + Profile */}
        <div className="flex items-center gap-4">
          {/* Search */}
          <div className="relative flex items-center">
            {searchOpen ? (
              <motion.div
                initial={{ width: 0, opacity: 0 }}
                animate={{ width: 260, opacity: 1 }}
                exit={{ width: 0, opacity: 0 }}
                transition={{ duration: 0.25 }}
                className="flex items-center bg-black/90 border border-white/30"
              >
                <FiSearch size={16} className="ml-2 text-white/60" />
                <input
                  type="text"
                  autoFocus
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                  onKeyDown={handleSearch}
                  onBlur={() => {
                    if (!searchQuery) setSearchOpen(false);
                  }}
                  placeholder="Titles, genres..."
                  className="w-full bg-transparent text-white text-sm px-2 py-1.5
                             outline-none placeholder:text-white/40"
                />
              </motion.div>
            ) : (
              <button
                onClick={() => setSearchOpen(true)}
                className="text-white hover:text-netflix-text-secondary transition-colors"
                aria-label="Search"
              >
                <FiSearch size={20} />
              </button>
            )}
          </div>

          {/* User Profile */}
          <Link
            href="/profile"
            className="flex items-center gap-2 group cursor-pointer"
          >
            <div
              className="w-8 h-8 rounded flex items-center justify-center text-lg
                         group-hover:ring-2 ring-white/30 transition-all"
              style={{ backgroundColor: currentUser?.color || "#808080" }}
            >
              {currentUser?.avatar || "\uD83D\uDC64"}
            </div>
            <span className="hidden lg:block text-sm text-netflix-text-secondary group-hover:text-white transition-colors">
              {userName || "Profile"}
            </span>
          </Link>
        </div>
      </div>
    </motion.nav>
  );
}
