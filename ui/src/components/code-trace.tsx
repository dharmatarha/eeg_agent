"use client";

/**
 * CodeTrace — Collapsible code execution block.
 *
 * Displays:
 * - Status badge (success/failed)
 * - Syntax-highlighted Python code
 * - Expandable execution logs
 */

import { useState } from "react";
import { Highlight, themes } from "prism-react-renderer";

interface CodeTraceProps {
  code: string;
  logs: string;
  error: boolean;
  index: number;
}

export function CodeTrace({ code, logs, error, index }: CodeTraceProps) {
  const [expanded, setExpanded] = useState(error);

  return (
    <div className="rounded-xl border border-white/10 bg-slate-800/60 backdrop-blur overflow-hidden">
      {/* Header */}
      <button
        type="button"
        onClick={() => setExpanded(!expanded)}
        className="w-full px-4 py-2.5 flex items-center gap-3 hover:bg-white/5 transition-colors"
      >
        <span
          className={`text-xs font-mono px-2 py-0.5 rounded ${
            error
              ? "bg-red-500/20 text-red-400"
              : "bg-emerald-500/20 text-emerald-400"
          }`}
        >
          {error ? "❌ Failed" : "✅ Success"}
        </span>
        <span className="text-sm text-slate-400 font-medium">
          Code Block {index + 1}
        </span>
        <span className="ml-auto text-slate-500 text-xs">
          {expanded ? "▼" : "▶"}
        </span>
      </button>

      {/* Content */}
      {expanded && (
        <div className="border-t border-white/5">
          {/* Code */}
          <div className="overflow-x-auto">
            <Highlight
              theme={themes.nightOwl}
              code={code.trim()}
              language="python"
            >
              {({ style, tokens, getLineProps, getTokenProps }) => (
                <pre
                  className="text-xs leading-relaxed p-4 m-0"
                  style={{ ...style, background: "transparent" }}
                >
                  {tokens.map((line, i) => (
                    <div key={i} {...getLineProps({ line })}>
                      <span className="inline-block w-8 text-right mr-4 text-slate-600 select-none">
                        {i + 1}
                      </span>
                      {line.map((token, key) => (
                        <span key={key} {...getTokenProps({ token })} />
                      ))}
                    </div>
                  ))}
                </pre>
              )}
            </Highlight>
          </div>

          {/* Logs */}
          {logs && (
            <div className="border-t border-white/5 px-4 py-3">
              <p className="text-[10px] uppercase tracking-wider text-slate-500 font-semibold mb-1.5">
                Output Logs
              </p>
              <pre
                className={`text-xs leading-relaxed font-mono whitespace-pre-wrap ${
                  error ? "text-red-400" : "text-slate-400"
                }`}
              >
                {logs}
              </pre>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
