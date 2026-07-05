"use client";

/**
 * Session page — drives a single analysis run.
 *
 * Connects to the WebSocket stream on mount, renders the assistant-ui
 * Thread with domain-specific components, and shows run status.
 */

import { useEffect, useCallback } from "react";
import { useParams } from "next/navigation";
import Link from "next/link";
import { EegRuntimeProvider, useEegSession } from "@/providers/eeg-runtime";
import { EegThread } from "@/components/eeg-thread";
import { RunHistory } from "@/components/run-history";
import { ReportViewer } from "@/components/report-viewer";
import { cancelRun } from "@/lib/api";

// Phase badge styling
const phaseBadge: Record<string, { label: string; color: string }> = {
  planner: { label: "Planning", color: "bg-blue-500/20 text-blue-400" },
  awaiting_hitl: {
    label: "Awaiting Review",
    color: "bg-amber-500/20 text-amber-400 animate-pulse",
  },
  executor: { label: "Executing", color: "bg-purple-500/20 text-purple-400" },
  critic: { label: "Reviewing", color: "bg-cyan-500/20 text-cyan-400" },
  completed: {
    label: "Completed",
    color: "bg-emerald-500/20 text-emerald-400",
  },
  failed: { label: "Failed", color: "bg-red-500/20 text-red-400" },
};

function SessionContent() {
  const params = useParams();
  const runId = params.runId as string;
  const { connect, phase, cancel } = useEegSession();

  // Connect on mount
  useEffect(() => {
    if (runId) {
      connect(runId);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [runId]);

  const handleCancel = useCallback(async () => {
    cancel();
    try {
      await cancelRun(runId);
    } catch {
      // best-effort
    }
  }, [cancel, runId]);

  const badge = phase ? phaseBadge[phase] : null;
  const isFinished = phase === "completed" || phase === "failed";

  return (
    <div className="min-h-screen flex">
      {/* Sidebar */}
      <aside className="w-64 border-r border-white/5 bg-slate-900/50 flex flex-col">
        <div className="p-4 border-b border-white/5">
          <Link
            href="/"
            className="text-sm text-indigo-400 hover:text-indigo-300 transition-colors flex items-center gap-1"
          >
            ← New Analysis
          </Link>
        </div>
        <div className="flex-1 overflow-y-auto">
          <RunHistory />
        </div>
      </aside>

      {/* Main content */}
      <div className="flex-1 flex flex-col">
        {/* Header */}
        <header className="border-b border-white/5 bg-slate-900/30 px-6 py-3 flex items-center gap-4">
          <div className="flex items-center gap-3 flex-1 min-w-0">
            <h1 className="text-sm font-semibold text-slate-300 truncate">
              <span className="text-slate-500">Run:</span>{" "}
              <code className="font-mono text-indigo-400">{runId}</code>
            </h1>
            {badge && (
              <span
                className={`text-xs font-medium px-2.5 py-1 rounded-full ${badge.color}`}
              >
                {badge.label}
              </span>
            )}
          </div>
          {!isFinished && (
            <button
              onClick={handleCancel}
              className="text-xs px-3 py-1.5 rounded-lg bg-red-600/20 text-red-400 hover:bg-red-600/30 border border-red-500/30 transition-all font-medium"
            >
              Cancel
            </button>
          )}
        </header>

        {/* Thread */}
        <div className="flex-1 overflow-hidden">
          {phase === "completed" ? (
            <div className="p-6 overflow-y-auto h-full">
              <ReportViewer runId={runId} />
            </div>
          ) : (
            <div className="h-full">
              <EegThread />
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

export default function RunPage() {
  return (
    <EegRuntimeProvider>
      <SessionContent />
    </EegRuntimeProvider>
  );
}
