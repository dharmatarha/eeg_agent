/**
 * HTTP client for the FastAPI bridge server REST endpoints.
 */

import type {
  BrowseResponse,
  CreateRunRequest,
  CreateRunResponse,
  RunSummary,
} from "./types";

export interface RunStateResponse {
  run_id: string;
  phase: string;
  state?: {
    user_directive?: string;
    data_path?: string;
    raw_metadata?: string;
    analysis_plan?: string;
    executed_code_blocks?: Array<{
      code: string;
      logs: string;
      error: boolean;
    }>;
    generated_plots?: string[];
    critic_feedback?: string;
    is_approved?: boolean;
    error_count?: number;
  };
  memory?: unknown;
  /** True when the run was interrupted by a backend crash (e.g. OOM). */
  interrupted?: boolean;
}

const API_URL =
  process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

async function fetchJson<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${API_URL}${path}`, {
    ...init,
    headers: {
      "Content-Type": "application/json",
      ...init?.headers,
    },
  });
  if (!res.ok) {
    const body = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(body.detail || `HTTP ${res.status}`);
  }
  return res.json();
}

/** Browse the data/ directory on the server. */
export async function browseData(): Promise<BrowseResponse> {
  return fetchJson<BrowseResponse>("/api/data/browse");
}

/** List past completed runs. */
export async function listRuns(): Promise<RunSummary[]> {
  const data = await fetchJson<{ runs: RunSummary[] }>("/api/runs");
  return data.runs;
}

/** Create a new analysis run. */
export async function createRun(
  req: CreateRunRequest
): Promise<CreateRunResponse> {
  return fetchJson<CreateRunResponse>("/api/runs", {
    method: "POST",
    body: JSON.stringify(req),
  });
}

/** Get a completed run's report markdown. */
export async function getRunReport(
  runId: string
): Promise<string> {
  const data = await fetchJson<{ report: string }>(
    `/api/runs/${runId}/report`
  );
  return data.report;
}

/** Cancel the active run. */
export async function cancelRun(runId: string): Promise<void> {
  await fetchJson(`/api/runs/${runId}`, { method: "DELETE" });
}

/** Get the WebSocket URL for streaming a run. */
export function getStreamUrl(runId: string): string {
  const wsBase = API_URL.replace(/^http/, "ws");
  return `${wsBase}/api/runs/${runId}/stream`;
}

/** Get the URL for a plot image. */
export function getPlotUrl(runId: string, filename: string): string {
  return `${API_URL}/api/runs/${runId}/plots/${filename}`;
}

/** Get the active or completed state of a run. */
export async function getRunState(runId: string): Promise<RunStateResponse> {
  return fetchJson<RunStateResponse>(`/api/runs/${runId}/state`);
}
