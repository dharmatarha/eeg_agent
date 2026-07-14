"use client";

/**
 * DataIntakeForm — Landing page form for configuring a new analysis run.
 *
 * Features:
 * - Interactive file tree browser (from GET /api/data/browse)
 * - Directive text area with drag-and-drop file support
 * - Reference run selector dropdown
 * - Submit button with concurrency guard
 */

import { useCallback, useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import { browseData, createRun, listRuns } from "@/lib/api";
import type { FileTreeNode, RunSummary } from "@/lib/types";

// ---------------------------------------------------------------------------
// File tree sub-component
// ---------------------------------------------------------------------------

function FileNode({
  node,
  depth,
  selectedPath,
  onSelect,
}: {
  node: FileTreeNode;
  depth: number;
  selectedPath: string | null;
  onSelect: (path: string) => void;
}) {
  const [expanded, setExpanded] = useState(depth < 1);
  const isDir = node.type === "directory";
  const isSelected = selectedPath === node.path;

  const icon = isDir
    ? expanded
      ? "📂"
      : "📁"
    : node.name.endsWith(".fif")
      ? "🧠"
      : node.name.endsWith(".set")
        ? "📊"
        : "📄";

  return (
    <div>
      <button
        type="button"
        onClick={() => {
          if (isDir) setExpanded(!expanded);
          onSelect(node.path);
        }}
        className={`
          w-full text-left flex items-center gap-2 px-3 py-1.5 rounded-md text-sm
          transition-colors duration-150
          ${isSelected ? "bg-indigo-500/20 text-indigo-300 ring-1 ring-indigo-500/40" : "text-slate-300 hover:bg-white/5"}
        `}
        style={{ paddingLeft: `${depth * 16 + 12}px` }}
      >
        <span className="flex-shrink-0">{icon}</span>
        <span className="truncate font-medium">{node.name}</span>
        {node.is_bids && (
          <span className="ml-auto text-[10px] font-bold px-1.5 py-0.5 rounded bg-emerald-500/20 text-emerald-400 uppercase tracking-wider">
            BIDS
          </span>
        )}
        {!isDir && node.size != null && (
          <span className="ml-auto text-xs text-slate-500">
            {(node.size / 1024).toFixed(0)} KB
          </span>
        )}
      </button>
      {isDir && expanded && node.children && (
        <div>
          {node.children.map((child) => (
            <FileNode
              key={child.path}
              node={child}
              depth={depth + 1}
              selectedPath={selectedPath}
              onSelect={onSelect}
            />
          ))}
          {node.children.length === 0 && (
            <p
              className="text-xs text-slate-500 italic"
              style={{ paddingLeft: `${(depth + 1) * 16 + 12}px` }}
            >
              (empty)
            </p>
          )}
        </div>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Main form component
// ---------------------------------------------------------------------------

export function DataIntakeForm() {
  const router = useRouter();

  const [tree, setTree] = useState<FileTreeNode[]>([]);
  const [runs, setRuns] = useState<RunSummary[]>([]);
  const [loading, setLoading] = useState(true);

  const [selectedPath, setSelectedPath] = useState<string | null>(null);
  const [directive, setDirective] = useState("");
  const [referenceRun, setReferenceRun] = useState("");
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Fetch data tree and past runs on mount
  useEffect(() => {
    async function load() {
      try {
        const [browseResult, runList] = await Promise.all([
          browseData(),
          listRuns(),
        ]);
        setTree(browseResult.tree);
        setRuns(runList);
      } catch (e) {
        setError(`Failed to connect to server: ${e}`);
      } finally {
        setLoading(false);
      }
    }
    load();
  }, []);

  // Drag-and-drop directive file
  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    const file = e.dataTransfer.files[0];
    if (file && (file.name.endsWith(".txt") || file.name.endsWith(".md"))) {
      const reader = new FileReader();
      reader.onload = () => {
        setDirective(reader.result as string);
      };
      reader.readAsText(file);
    }
  }, []);

  // Submit handler
  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!selectedPath || !directive.trim()) return;

    setSubmitting(true);
    setError(null);

    try {
      const result = await createRun({
        data_path: selectedPath,
        directive: directive.trim(),
        reference_run_id: referenceRun || undefined,
      });
      router.push(`/run/${result.run_id}`);
    } catch (e) {
      setError(
        e instanceof Error ? e.message : "Failed to create run"
      );
      setSubmitting(false);
    }
  };

  return (
    <form onSubmit={handleSubmit} className="space-y-6">
      {/* --- Data Path Browser --- */}
      <div>
        <label className="block text-sm font-semibold text-slate-300 mb-2">
          📁 Select EEG Data
        </label>
        <div className="bg-slate-800/50 rounded-lg border border-white/10 max-h-72 overflow-y-auto p-2">
          {loading ? (
            <div className="flex items-center justify-center py-8">
              <div className="animate-spin rounded-full h-6 w-6 border-2 border-indigo-400 border-t-transparent" />
            </div>
          ) : tree.length === 0 ? (
            <p className="text-sm text-slate-500 text-center py-8">
              No EEG files found in data/ directory.
            </p>
          ) : (
            tree.map((node) => (
              <FileNode
                key={node.path}
                node={node}
                depth={0}
                selectedPath={selectedPath}
                onSelect={setSelectedPath}
              />
            ))
          )}
        </div>
        {selectedPath && (
          <p className="mt-1.5 text-xs text-indigo-400">
            Selected: <code className="font-mono">{selectedPath}</code>
          </p>
        )}
      </div>

      {/* --- Directive --- */}
      <div>
        <label
          htmlFor="directive"
          className="block text-sm font-semibold text-slate-300 mb-2"
        >
          📝 Analysis Directive
        </label>
        <textarea
          id="directive"
          value={directive}
          onChange={(e) => setDirective(e.target.value)}
          onDrop={handleDrop}
          onDragOver={(e) => e.preventDefault()}
          placeholder="Describe your analysis goals, e.g. 'Perform ERP analysis on auditory oddball data, focusing on the N400 component...'"
          rows={4}
          className="w-full bg-slate-800/50 border border-white/10 rounded-lg px-4 py-3 text-sm text-slate-200 placeholder-slate-500 focus:ring-2 focus:ring-indigo-500/50 focus:border-indigo-500/50 outline-none transition-all resize-y"
        />
        <p className="mt-1 text-xs text-slate-500">
          Drag & drop a .txt or .md file, or type directly
        </p>
      </div>

      {/* --- Reference Run Selector --- */}
      <div>
        <label
          htmlFor="reference-run"
          className="block text-sm font-semibold text-slate-300 mb-2"
        >
          🔗 Reference Run{" "}
          <span className="font-normal text-slate-500">(optional)</span>
        </label>
        <select
          id="reference-run"
          value={referenceRun}
          onChange={(e) => setReferenceRun(e.target.value)}
          className="w-full bg-slate-800/50 border border-white/10 rounded-lg px-4 py-2.5 text-sm text-slate-200 focus:ring-2 focus:ring-indigo-500/50 focus:border-indigo-500/50 outline-none"
        >
          <option value="">None</option>
          {runs
            .filter((run) => run.status === "completed")
            .map((run) => (
              <option key={run.run_id} value={run.run_id}>
                {run.run_id} — {run.directive?.slice(0, 60)}
                {run.is_approved ? " ✅" : ""}
              </option>
            ))}
        </select>
      </div>

      {/* --- Error Display --- */}
      {error && (
        <div className="bg-red-500/10 border border-red-500/30 rounded-lg px-4 py-3 text-sm text-red-400">
          {error}
        </div>
      )}

      {/* --- Submit --- */}
      <button
        type="submit"
        disabled={!selectedPath || !directive.trim() || submitting}
        className={`
          w-full py-3 px-6 rounded-lg text-sm font-semibold transition-all duration-200
          ${
            !selectedPath || !directive.trim() || submitting
              ? "bg-slate-700 text-slate-500 cursor-not-allowed"
              : "bg-gradient-to-r from-indigo-600 to-purple-600 text-white hover:from-indigo-500 hover:to-purple-500 shadow-lg shadow-indigo-500/25 hover:shadow-indigo-500/40"
          }
        `}
      >
        {submitting ? (
          <span className="flex items-center justify-center gap-2">
            <span className="animate-spin rounded-full h-4 w-4 border-2 border-white border-t-transparent" />
            Starting analysis...
          </span>
        ) : (
          "🚀 Start Analysis"
        )}
      </button>
    </form>
  );
}
