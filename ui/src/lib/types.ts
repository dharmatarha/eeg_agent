/**
 * Wire protocol types for communication between the Next.js frontend
 * and the FastAPI bridge server.
 */

// ---------------------------------------------------------------------------
// Server → Client events (sent over WebSocket)
// ---------------------------------------------------------------------------

export type ServerEvent =
  | { type: "status"; phase: RunPhase; message: string }
  | { type: "plan_ready"; plan: string }
  | { type: "hitl_required"; plan: string }
  | {
      type: "code_block";
      index: number;
      code: string;
      logs: string;
      error: boolean;
    }
  | { type: "plot"; index: number; base64: string }
  | { type: "critic_verdict"; approved: boolean; feedback: string }
  | {
      type: "completed";
      thread_id: string;
      artifacts?: {
        report?: string;
        pipeline?: string;
        plots?: string[];
      };
    }
  | { type: "error"; message: string };

// ---------------------------------------------------------------------------
// Client → Server commands (sent over WebSocket)
// ---------------------------------------------------------------------------

export type ClientCommand =
  | {
      type: "hitl_response";
      decision: "approve" | "reject";
      feedback?: string;
    }
  | { type: "cancel" };

// ---------------------------------------------------------------------------
// Run phases
// ---------------------------------------------------------------------------

export type RunPhase =
  | "ingest"
  | "planner"
  | "awaiting_hitl"
  | "executor"
  | "critic"
  | "completed"
  | "failed";

// ---------------------------------------------------------------------------
// REST API types
// ---------------------------------------------------------------------------

export interface FileTreeNode {
  name: string;
  path: string;
  type: "file" | "directory";
  size?: number;
  is_bids?: boolean;
  children?: FileTreeNode[];
}

export interface BrowseResponse {
  data_dir: string;
  tree: FileTreeNode[];
}

export interface RunSummary {
  run_id: string;
  timestamp: string;
  directive: string;
  status: string;
  is_approved: boolean | null;
}

export interface CreateRunRequest {
  data_path: string;
  directive: string;
  reference_run_id?: string;
}

export interface CreateRunResponse {
  run_id: string;
  data_path: string;
  container_data_path: string;
  is_bids: boolean;
}

// ---------------------------------------------------------------------------
// Internal UI message types (our "external store" format)
// ---------------------------------------------------------------------------

export interface EegMessage {
  id: string;
  role: "user" | "assistant";
  type: ServerEvent["type"] | "user_submission";
  content: string;
  metadata?: Record<string, unknown>;
  timestamp: Date;
}
