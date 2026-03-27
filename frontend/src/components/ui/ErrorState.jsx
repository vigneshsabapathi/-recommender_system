"use client";

import { FiAlertTriangle, FiRefreshCw } from "react-icons/fi";
import { Button } from "@/components/ui/button";

export default function ErrorState({ message, onRetry }) {
  return (
    <div className="flex flex-col items-center justify-center py-20 px-4 text-center">
      <FiAlertTriangle className="text-netflix-red mb-4" size={48} />
      <h3 className="text-xl font-semibold text-white mb-2">
        Something went wrong
      </h3>
      <p className="text-netflix-text-secondary text-sm mb-6 max-w-md">
        {message || "We're having trouble loading this content. Please try again."}
      </p>
      {onRetry && (
        <Button variant="destructive" onClick={onRetry} className="gap-2 px-6 py-2.5 h-auto">
          <FiRefreshCw size={16} />
          Try Again
        </Button>
      )}
    </div>
  );
}
