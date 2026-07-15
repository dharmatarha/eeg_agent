"use client";

/**
 * Session page — drives a single analysis run.
 *
 * Connects to the WebSocket stream on mount, renders the assistant-ui
 * Thread with domain-specific components, and shows run status.
 */

import { useEffect, useCallback, useState } from "react";
import { useParams } from "next/navigation";
import Link from "next/link";
import { EegRuntimeProvider, useEegSession } from "@/providers/eeg-runtime";
import { EegThread } from "@/components/eeg-thread";
import { RunHistory } from "@/components/run-history";
import { ReportViewer } from "@/components/report-viewer";
import { cancelRun } from "@/lib/api";
import { CodeTrace } from "@/components/code-trace";
import { PlotGallery } from "@/components/plot-gallery";
import { MessageSquare, Terminal, Image as ImageIcon, FileText, Lock } from "lucide-react";

// Phase badge styling
const phaseBadge: Record<string, { label: string; color: string }> = {
  ingest: {
    label: "Ingesting Metadata",
    color: "bg-indigo-500/20 text-indigo-400 animate-pulse",
  },
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
  const { connect, phase, cancel, codeBlocks, plots, messages } = useEegSession();
  const [activeTab, setActiveTab] = useState<"chat" | "code" | "plots" | "report">("chat");

  const lastStatusMessage = [...messages]
    .reverse()
    .find((m) => m.type === "status");

  // Connect on mount
  useEffect(() => {
    if (runId) {
      connect(runId);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [runId]);

  // Auto-switch tabs based on phase transitions
  /* eslint-disable react-hooks/set-state-in-effect */
  useEffect(() => {
    if (phase === "completed") {
      setActiveTab("report");
    } else if (phase === "executor") {
      setActiveTab("code");
    } else if (phase === "critic") {
      setActiveTab("plots");
    } else if (phase === "awaiting_hitl" || phase === "planner" || phase === "ingest") {
      setActiveTab("chat");
    }
  }, [phase]);
  /* eslint-enable react-hooks/set-state-in-effect */

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
      <div className="flex-1 flex flex-col h-screen overflow-hidden bg-slate-950">
        {/* Header */}
        <header className="border-b border-white/5 bg-slate-900/30 px-6 py-3 flex items-center gap-4 shrink-0">
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

        {/* Tab Navigation */}
        <div className="border-b border-white/5 bg-slate-900/10 px-6 py-2 flex items-center gap-2 shrink-0">
          <nav className="flex space-x-1" aria-label="Tabs">
            {/* Chat Tab */}
            <button
              onClick={() => setActiveTab("chat")}
              className={`flex items-center gap-2 px-4 py-2 rounded-lg text-xs font-semibold uppercase tracking-wider transition-all duration-200 border cursor-pointer ${
                activeTab === "chat"
                  ? "bg-indigo-500/10 text-indigo-400 border-indigo-500/20 shadow-[0_0_12px_rgba(99,102,241,0.15)]"
                  : "text-slate-400 hover:text-slate-200 hover:bg-slate-800/40 border-transparent"
              }`}
            >
              <MessageSquare className="w-4 h-4" />
              Chat & Planning
              {phase === "awaiting_hitl" && (
                <span className="w-1.5 h-1.5 rounded-full bg-amber-500 animate-pulse" />
              )}
            </button>

            {/* Code Execution Tab */}
            <button
              onClick={() => setActiveTab("code")}
              className={`flex items-center gap-2 px-4 py-2 rounded-lg text-xs font-semibold uppercase tracking-wider transition-all duration-200 border cursor-pointer ${
                activeTab === "code"
                  ? "bg-purple-500/10 text-purple-400 border-purple-500/20 shadow-[0_0_12px_rgba(168,85,247,0.15)]"
                  : "text-slate-400 hover:text-slate-200 hover:bg-slate-800/40 border-transparent"
              }`}
            >
              <Terminal className="w-4 h-4" />
              Code Execution
              {codeBlocks.length > 0 && (
                <span className={`text-[10px] px-1.5 py-0.5 rounded font-mono font-bold leading-none ${
                  codeBlocks.some(b => b.metadata?.error)
                    ? "bg-red-500/20 text-red-400 border border-red-500/30 animate-pulse"
                    : "bg-purple-500/20 text-purple-400 border border-purple-500/10"
                }`}>
                  {codeBlocks.length}
                </span>
              )}
            </button>

            {/* Generated Plots Tab */}
            <button
              onClick={() => setActiveTab("plots")}
              className={`flex items-center gap-2 px-4 py-2 rounded-lg text-xs font-semibold uppercase tracking-wider transition-all duration-200 border cursor-pointer ${
                activeTab === "plots"
                  ? "bg-emerald-500/10 text-emerald-400 border-emerald-500/20 shadow-[0_0_12px_rgba(16,185,129,0.15)]"
                  : "text-slate-400 hover:text-slate-200 hover:bg-slate-800/40 border-transparent"
              }`}
            >
              <ImageIcon className="w-4 h-4" />
              Generated Plots
              {plots.length > 0 && (
                <span className="text-[10px] bg-emerald-500/20 text-emerald-400 border border-emerald-500/10 px-1.5 py-0.5 rounded font-mono font-bold leading-none">
                  {plots.length}
                </span>
              )}
            </button>

            {/* Final Report Tab */}
            <button
              onClick={() => {
                if (phase === "completed") {
                  setActiveTab("report");
                }
              }}
              disabled={phase !== "completed"}
              className={`flex items-center gap-2 px-4 py-2 rounded-lg text-xs font-semibold uppercase tracking-wider transition-all duration-200 border ${
                phase !== "completed"
                  ? "text-slate-600 cursor-not-allowed border-transparent opacity-50"
                  : activeTab === "report"
                    ? "bg-cyan-500/10 text-cyan-400 border-cyan-500/20 shadow-[0_0_12px_rgba(6,182,212,0.15)] cursor-pointer"
                    : "text-slate-400 hover:text-slate-200 hover:bg-slate-800/40 border-transparent cursor-pointer"
              }`}
              title={phase !== "completed" ? "Available when the analysis finishes" : undefined}
            >
              {phase !== "completed" ? (
                <Lock className="w-4 h-4 text-slate-600" />
              ) : (
                <FileText className="w-4 h-4" />
              )}
              Final Report
            </button>
          </nav>
        </div>

        {/* Tab Content */}
        <div className="flex-1 overflow-hidden relative">
          {activeTab === "chat" && (
            <div className="h-full flex flex-col max-w-4xl mx-auto px-4 py-6">
              <div className="flex-1 overflow-hidden rounded-xl border border-white/5 bg-slate-900/20 backdrop-blur-sm shadow-xl flex flex-col">
                <EegThread />
              </div>
            </div>
          )}

          {activeTab === "code" && (
            <div className="h-full overflow-y-auto p-6 space-y-4 max-w-5xl mx-auto">
              {codeBlocks.length === 0 ? (
                <div className="flex flex-col items-center justify-center h-64 text-slate-500 text-sm space-y-3">
                  <div className="h-8 w-8 rounded-full border border-dashed border-slate-600 animate-pulse flex items-center justify-center">
                    <Terminal className="w-4 h-4 text-slate-500" />
                  </div>
                  <p>Waiting for code execution to start...</p>
                </div>
              ) : (
                codeBlocks.map((block) => (
                  <CodeTrace
                    key={block.id}
                    code={block.metadata?.code as string}
                    logs={block.metadata?.logs as string}
                    error={block.metadata?.error as boolean}
                    index={block.metadata?.index as number}
                  />
                ))
              )}

              {phase === "executor" && lastStatusMessage && (
                <div className="flex items-center gap-3 p-4 rounded-xl border border-purple-500/20 bg-purple-500/5 text-purple-300 font-mono text-xs animate-pulse">
                  <div className="w-2 h-2 rounded-full bg-purple-500 animate-ping shrink-0" />
                  <span>{lastStatusMessage.content}</span>
                </div>
              )}
            </div>
          )}

          {activeTab === "plots" && (
            <div className="h-full overflow-y-auto p-6 max-w-7xl mx-auto">
              {plots.length === 0 ? (
                <div className="flex flex-col items-center justify-center h-64 text-slate-500 text-sm space-y-3">
                  <span className="text-3xl">📊</span>
                  <p>No plots generated yet...</p>
                </div>
              ) : (
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                  {plots.map((plot) => (
                    <PlotGallery
                      key={plot.id}
                      base64={plot.metadata?.base64 as string}
                      index={plot.metadata?.index as number}
                    />
                  ))}
                </div>
              )}

              {phase === "critic" && lastStatusMessage && (
                <div className="mt-6 flex items-center gap-3 p-4 rounded-xl border border-cyan-500/20 bg-cyan-500/5 text-cyan-300 font-mono text-xs animate-pulse">
                  <div className="w-2 h-2 rounded-full bg-cyan-500 animate-ping shrink-0" />
                  <span>{lastStatusMessage.content}</span>
                </div>
              )}
            </div>
          )}

          {activeTab === "report" && (
            <div className="h-full overflow-y-auto p-6 max-w-4xl mx-auto">
              <ReportViewer runId={runId} />
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
