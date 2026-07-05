"use client";

/**
 * RunHistory — Sidebar listing past runs from the output/ directory.
 */

import { useEffect, useState } from "react";
import Link from "next/link";
import { listRuns } from "@/lib/api";
import type { RunSummary } from "@/lib/types";

export function RunHistory() {
  const [runs, setRuns] = useState<RunSummary[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    listRuns()
      .then(setRuns)
      .catch(() => {})
      .finally(() => setLoading(false));
  }, []);

  if (loading) {
    return (
      <div className="p-4 text-center">
        <div className="animate-spin rounded-full h-5 w-5 border-2 border-indigo-400 border-t-transparent mx-auto" />
      </div>
    );
  }

  if (runs.length === 0) {
    return (
      <div className="p-4 text-center text-sm text-slate-500">
        No past runs yet.
      </div>
    );
  }

  return (
    <div className="space-y-1 p-2">
      <p className="text-[10px] uppercase tracking-wider text-slate-500 font-semibold px-2 pb-1">
        Past Runs
      </p>
      {runs.map((run) => (
        <Link
          key={run.run_id}
          href={`/run/${run.run_id}`}
          className="block px-3 py-2 rounded-lg text-sm hover:bg-white/5 transition-colors group"
        >
          <div className="flex items-center gap-2">
            <span className="text-xs">
              {run.is_approved === true
                ? "✅"
                : run.is_approved === false
                  ? "❌"
                  : "⏳"}
            </span>
            <span className="text-slate-300 font-mono text-xs truncate group-hover:text-indigo-300 transition-colors">
              {run.run_id}
            </span>
          </div>
          <p className="text-xs text-slate-500 truncate mt-0.5 pl-5">
            {run.directive?.slice(0, 50)}
          </p>
        </Link>
      ))}
    </div>
  );
}
