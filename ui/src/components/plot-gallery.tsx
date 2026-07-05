"use client";

/**
 * PlotGallery — EEG plot viewer with click-to-expand fullscreen lightbox.
 */

import { useState } from "react";

interface PlotGalleryProps {
  base64: string;
  index: number;
}

export function PlotGallery({ base64, index }: PlotGalleryProps) {
  const [fullscreen, setFullscreen] = useState(false);
  const src = `data:image/png;base64,${base64}`;

  return (
    <>
      {/* Thumbnail */}
      <div className="rounded-xl border border-white/10 bg-slate-800/60 backdrop-blur overflow-hidden">
        <div className="px-4 py-2 border-b border-white/5 flex items-center justify-between">
          <span className="text-xs text-slate-400 font-medium">
            📊 Plot {index + 1}
          </span>
          <button
            type="button"
            onClick={() => setFullscreen(true)}
            className="text-xs text-indigo-400 hover:text-indigo-300 transition-colors"
          >
            🔍 Expand
          </button>
        </div>
        <button
          type="button"
          onClick={() => setFullscreen(true)}
          className="w-full cursor-zoom-in"
        >
          <img
            src={src}
            alt={`EEG Plot ${index + 1}`}
            className="w-full h-auto"
          />
        </button>
      </div>

      {/* Fullscreen lightbox */}
      {fullscreen && (
        <div
          className="fixed inset-0 z-50 bg-black/90 flex items-center justify-center p-8"
          onClick={() => setFullscreen(false)}
        >
          <button
            type="button"
            className="absolute top-4 right-4 text-white/70 hover:text-white text-2xl transition-colors"
            onClick={() => setFullscreen(false)}
          >
            ✕
          </button>
          <img
            src={src}
            alt={`EEG Plot ${index + 1} (fullscreen)`}
            className="max-w-full max-h-full object-contain rounded-lg shadow-2xl"
            onClick={(e) => e.stopPropagation()}
          />
        </div>
      )}
    </>
  );
}
