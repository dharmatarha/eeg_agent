"use client";

/**
 * CriticVerdict — Styled card showing the Critic agent's review.
 *
 * Displays a verdict badge (APPROVED/REJECTED) and the full feedback
 * rendered as markdown.
 */

import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import remarkMath from "remark-math";
import rehypeKatex from "rehype-katex";

interface CriticVerdictProps {
  approved: boolean;
  feedback: string;
}

export function CriticVerdict({ approved, feedback }: CriticVerdictProps) {
  return (
    <div
      className={`rounded-xl border backdrop-blur overflow-hidden ${
        approved
          ? "border-emerald-500/30 bg-emerald-500/5"
          : "border-red-500/30 bg-red-500/5"
      }`}
    >
      {/* Header */}
      <div
        className={`px-5 py-3 border-b flex items-center gap-3 ${
          approved
            ? "border-emerald-500/20 bg-emerald-500/10"
            : "border-red-500/20 bg-red-500/10"
        }`}
      >
        <span className="text-2xl">{approved ? "✅" : "❌"}</span>
        <div>
          <h3
            className={`text-sm font-bold ${
              approved ? "text-emerald-400" : "text-red-400"
            }`}
          >
            Critic Verdict: {approved ? "APPROVED" : "REJECTED"}
          </h3>
          <p className="text-xs text-slate-400">
            Quality assurance & methodology review
          </p>
        </div>
      </div>

      {/* Feedback */}
      <div className="px-5 py-4 prose prose-invert prose-sm prose-slate max-w-none max-h-96 overflow-y-auto">
        <ReactMarkdown remarkPlugins={[remarkGfm, remarkMath]} rehypePlugins={[rehypeKatex]}>{feedback}</ReactMarkdown>
      </div>
    </div>
  );
}
