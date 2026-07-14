"use client";

/**
 * ReportViewer — Renders a final_report.md as styled markdown.
 */

import { useEffect, useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import remarkMath from "remark-math";
import rehypeKatex from "rehype-katex";
import { getRunReport } from "@/lib/api";

interface ReportViewerProps {
  runId: string;
}

export function ReportViewer({ runId }: ReportViewerProps) {
  const [report, setReport] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    getRunReport(runId)
      .then(setReport)
      .catch((e) => setError(e.message))
      .finally(() => setLoading(false));
  }, [runId]);

  if (loading) {
    return (
      <div className="flex items-center justify-center py-12">
        <div className="animate-spin rounded-full h-8 w-8 border-2 border-indigo-400 border-t-transparent" />
      </div>
    );
  }

  if (error) {
    return (
      <div className="bg-red-500/10 border border-red-500/30 rounded-lg px-4 py-3 text-sm text-red-400">
        Failed to load report: {error}
      </div>
    );
  }

  return (
    <div className="rounded-xl border border-white/10 bg-slate-800/60 backdrop-blur overflow-hidden">
      <div className="px-5 py-3 bg-gradient-to-r from-indigo-600/20 to-purple-600/20 border-b border-white/10">
        <h3 className="text-sm font-semibold text-indigo-300 flex items-center gap-2">
          <span className="text-lg">📄</span>
          Final Report
        </h3>
      </div>
      <div className="px-6 py-5 prose prose-invert prose-sm prose-slate max-w-none overflow-y-auto max-h-[600px]">
        <ReactMarkdown remarkPlugins={[remarkGfm, remarkMath]} rehypePlugins={[rehypeKatex]}>
          {report || ""}
        </ReactMarkdown>
      </div>
    </div>
  );
}
