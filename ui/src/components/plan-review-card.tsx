"use client";

/**
 * PlanReviewCard — HITL component for reviewing the analysis plan.
 *
 * Renders the plan as markdown with three action buttons:
 * ✅ Approve, ✏️ Edit (with inline feedback), ❌ Reject
 */

import { useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import remarkMath from "remark-math";
import rehypeKatex from "rehype-katex";
import { useEegSession } from "@/providers/eeg-runtime";

interface PlanReviewCardProps {
  plan: string;
  requiresAction: boolean;
}

export function PlanReviewCard({ plan, requiresAction }: PlanReviewCardProps) {
  const { approve, reject, phase } = useEegSession();
  const [editing, setEditing] = useState(false);
  const [feedback, setFeedback] = useState("");
  const isWaiting = phase === "awaiting_hitl" && requiresAction;

  return (
    <div className="rounded-xl border border-white/10 bg-slate-800/60 backdrop-blur overflow-hidden">
      {/* Header */}
      <div className="px-5 py-3 bg-gradient-to-r from-indigo-600/20 to-purple-600/20 border-b border-white/10 flex items-center justify-between">
        <h3 className="text-sm font-semibold text-indigo-300 flex items-center gap-2">
          <span className="text-lg">📋</span>
          Analysis Plan
        </h3>
        {isWaiting && (
          <span className="text-xs px-2 py-1 rounded-full bg-amber-500/20 text-amber-400 animate-pulse font-medium">
            Awaiting Review
          </span>
        )}
        {!isWaiting && phase !== "awaiting_hitl" && (
          <span className="text-xs px-2 py-1 rounded-full bg-emerald-500/20 text-emerald-400 font-medium">
            ✅ Approved
          </span>
        )}
      </div>

      {/* Plan content */}
      <div className="px-5 py-4 max-h-96 overflow-y-auto prose prose-invert prose-sm prose-slate max-w-none">
        <ReactMarkdown remarkPlugins={[remarkGfm, remarkMath]} rehypePlugins={[rehypeKatex]}>{plan}</ReactMarkdown>
      </div>

      {/* Actions */}
      {isWaiting && (
        <div className="px-5 py-4 border-t border-white/10 space-y-3">
          {editing && (
            <div className="space-y-2">
              <textarea
                value={feedback}
                onChange={(e) => setFeedback(e.target.value)}
                placeholder="Provide feedback or modifications to the plan..."
                rows={3}
                className="w-full bg-slate-900/50 border border-white/10 rounded-lg px-3 py-2 text-sm text-slate-200 placeholder-slate-500 outline-none focus:ring-2 focus:ring-indigo-500/50 resize-y"
                autoFocus
              />
            </div>
          )}
          <div className="flex gap-2">
            <button
              onClick={() => approve(editing ? feedback : undefined)}
              className="flex-1 py-2 px-4 rounded-lg text-sm font-semibold bg-gradient-to-r from-emerald-600 to-emerald-500 text-white hover:from-emerald-500 hover:to-emerald-400 transition-all shadow-lg shadow-emerald-500/20"
            >
              ✅ {editing ? "Approve with Feedback" : "Approve"}
            </button>
            {!editing && (
              <button
                onClick={() => setEditing(true)}
                className="py-2 px-4 rounded-lg text-sm font-semibold bg-slate-700 text-slate-300 hover:bg-slate-600 transition-all"
              >
                ✏️ Edit
              </button>
            )}
            <button
              onClick={reject}
              className="py-2 px-4 rounded-lg text-sm font-semibold bg-red-600/20 text-red-400 hover:bg-red-600/30 border border-red-500/30 transition-all"
            >
              ❌ Reject
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
