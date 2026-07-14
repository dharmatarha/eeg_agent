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

  const activeRuns = runs.filter(
    (run) =>
      run.status &&
      run.status !== "completed" &&
      run.status !== "failed"
  );
  const pastRuns = runs.filter(
    (run) =>
      !run.status ||
      run.status === "completed" ||
      run.status === "failed"
  );

  if (runs.length === 0) {
    return (
      <div className="p-4 text-center text-sm text-slate-500">
        No runs found.
      </div>
    );
  }

  return (
    <div className="space-y-4 p-2">
      {activeRuns.length > 0 && (
        <div className="space-y-1">
          <p className="text-[10px] uppercase tracking-wider text-emerald-400 font-semibold px-2 pb-1 flex items-center gap-1.5">
            <span className="relative flex h-1.5 w-1.5">
              <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75" />
              <span className="relative inline-flex rounded-full h-1.5 w-1.5 bg-emerald-500" />
            </span>
            Active Runs
          </p>
          {activeRuns.map((run) => (
            <Link
              key={run.run_id}
              href={`/run/${run.run_id}`}
              className="block px-3 py-2 rounded-lg text-sm hover:bg-white/5 transition-colors group bg-emerald-500/5 border border-emerald-500/10"
            >
              <div className="flex items-center justify-between gap-2">
                <span className="text-slate-300 font-mono text-xs truncate group-hover:text-indigo-300 transition-colors">
                  {run.run_id}
                </span>
                <span className="text-[10px] font-medium px-1.5 py-0.5 rounded bg-emerald-500/10 text-emerald-400 border border-emerald-500/20 capitalize whitespace-nowrap">
                  {run.status.replace("_", " ")}
                </span>
              </div>
              <p className="text-xs text-slate-400 truncate mt-1 pl-1">
                {run.directive?.slice(0, 50)}
              </p>
            </Link>
          ))}
        </div>
      )}

      <div className="space-y-1">
        <p className="text-[10px] uppercase tracking-wider text-slate-500 font-semibold px-2 pb-1">
          Past Runs
        </p>
        {pastRuns.length === 0 ? (
          <div className="text-[11px] text-slate-600 px-2 py-1 italic">
            No past runs yet.
          </div>
        ) : (
          pastRuns.map((run) => (
            <Link
              key={run.run_id}
              href={`/run/${run.run_id}`}
              className="block px-3 py-2 rounded-lg text-sm hover:bg-white/5 transition-colors group"
            >
              <div className="flex items-center gap-2">
                <span className="text-xs">
                  {run.status === "failed"
                    ? "❌"
                    : run.is_approved === true
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
          ))
        )}
      </div>
    </div>
  );
}
