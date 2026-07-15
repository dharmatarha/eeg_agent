"use client";

/**
 * EEG Runtime Provider — bridges the FastAPI WebSocket stream with
 * assistant-ui's ExternalStoreRuntime.
 *
 * Maintains a local messages array, connects to the WebSocket on mount,
 * and converts incoming ServerEvents into EegMessages that feed the
 * ExternalStoreRuntime.
 */

import {
  type ReactNode,
  createContext,
  useCallback,
  useContext,
  useRef,
  useState,
} from "react";
import {
  useExternalStoreRuntime,
  AssistantRuntimeProvider,
} from "@assistant-ui/react";
import { convertMessage } from "@/lib/convert-message";
import { connectStream, type WsClient } from "@/lib/ws";
import { getRunState } from "@/lib/api";
import type { EegMessage, RunPhase, ServerEvent } from "@/lib/types";

// ---------------------------------------------------------------------------
// Context for imperative actions (HITL, cancel, phase tracking)
// ---------------------------------------------------------------------------

interface EegSessionContext {
  runId: string | null;
  phase: RunPhase | null;
  isConnected: boolean;
  /** Connect to a run's WebSocket stream. */
  connect: (runId: string) => Promise<void>;
  /** Send an HITL approval. */
  approve: (feedback?: string) => void;
  /** Send an HITL rejection. */
  reject: () => void;
  /** Cancel the current run. */
  cancel: () => void;
  codeBlocks: EegMessage[];
  plots: EegMessage[];
  messages: EegMessage[];
}

const SessionCtx = createContext<EegSessionContext>({
  runId: null,
  phase: null,
  isConnected: false,
  connect: async () => {},
  approve: () => {},
  reject: () => {},
  cancel: () => {},
  codeBlocks: [],
  plots: [],
  messages: [],
});

export const useEegSession = () => useContext(SessionCtx);

// ---------------------------------------------------------------------------
// Provider component
// ---------------------------------------------------------------------------

let msgCounter = 0;
function nextId(): string {
  return `msg-${Date.now()}-${++msgCounter}`;
}

export function EegRuntimeProvider({ children }: { children: ReactNode }) {
  const [messages, setMessages] = useState<EegMessage[]>([]);
  const [isRunning, setIsRunning] = useState(false);
  const [phase, setPhase] = useState<RunPhase | null>(null);
  const [runId, setRunId] = useState<string | null>(null);
  const [isConnected, setIsConnected] = useState(false);
  const wsRef = useRef<WsClient | null>(null);

  // --- Helper to append or update a message (deduplication) ---
  const addOrUpdateMessage = useCallback((msg: EegMessage) => {
    setMessages((prev) => {
      // 1. For code blocks, deduplicate by index
      if (msg.type === "code_block") {
        const idx = msg.metadata?.index;
        const existing = prev.findIndex(
          (m) => m.type === "code_block" && m.metadata?.index === idx
        );
        if (existing !== -1) {
          const next = [...prev];
          next[existing] = {
            ...next[existing],
            content: msg.content,
            metadata: msg.metadata,
          };
          return next;
        }
      }
      // 2. For plots, deduplicate by index
      if (msg.type === "plot") {
        const idx = msg.metadata?.index;
        const existing = prev.findIndex(
          (m) => m.type === "plot" && m.metadata?.index === idx
        );
        if (existing !== -1) {
          const next = [...prev];
          next[existing] = {
            ...next[existing],
            content: msg.content,
            metadata: msg.metadata,
          };
          return next;
        }
      }
      // 3. For plan_ready and hitl_required, deduplicate if in the same planning turn
      if (msg.type === "plan_ready" || msg.type === "hitl_required") {
        let lastPlanIdx = -1;
        for (let i = prev.length - 1; i >= 0; i--) {
          if (prev[i].type === "plan_ready" || prev[i].type === "hitl_required") {
            lastPlanIdx = i;
            break;
          }
        }

        if (lastPlanIdx !== -1) {
          const hasUserMsgAfter = prev.slice(lastPlanIdx + 1).some((m) => m.role === "user");
          if (!hasUserMsgAfter) {
            const next = [...prev];
            next[lastPlanIdx] = {
              ...next[lastPlanIdx],
              type: msg.type,
              content: msg.content,
            };
            return next;
          }
        }
      }
      // 4. For critic verdicts, deduplicate
      if (msg.type === "critic_verdict") {
        const existing = prev.findIndex((m) => m.type === "critic_verdict");
        if (existing !== -1) {
          const next = [...prev];
          next[existing] = {
            ...next[existing],
            content: msg.content,
            metadata: msg.metadata,
          };
          return next;
        }
      }
      // 5. For user submissions, check if already exists
      if (msg.type === "user_submission") {
        const existing = prev.find(
          (m) => m.type === "user_submission" && m.content === msg.content
        );
        if (existing) return prev;
      }
      // 6. Otherwise, append
      return [...prev, msg];
    });
  }, []);

  // --- Handle incoming server events ---
  const handleEvent = useCallback((event: ServerEvent) => {
    const id = nextId();
    const now = new Date();

    switch (event.type) {
      case "status":
        setPhase(event.phase);
        addOrUpdateMessage({
          id,
          role: "assistant",
          type: "status",
          content: event.message,
          metadata: { phase: event.phase },
          timestamp: now,
        });
        break;

      case "plan_ready":
        addOrUpdateMessage({
          id,
          role: "assistant",
          type: "plan_ready",
          content: event.plan,
          timestamp: now,
        });
        break;

      case "hitl_required":
        setPhase("awaiting_hitl");
        addOrUpdateMessage({
          id,
          role: "assistant",
          type: "hitl_required",
          content: event.plan,
          timestamp: now,
        });
        break;

      case "code_block":
        addOrUpdateMessage({
          id,
          role: "assistant",
          type: "code_block",
          content: event.code,
          metadata: {
            code: event.code,
            logs: event.logs,
            error: event.error,
            index: event.index,
          },
          timestamp: now,
        });
        break;

      case "plot":
        addOrUpdateMessage({
          id,
          role: "assistant",
          type: "plot",
          content: `Plot ${event.index + 1}`,
          metadata: { base64: event.base64, index: event.index },
          timestamp: now,
        });
        break;

      case "critic_verdict":
        setPhase("critic");
        addOrUpdateMessage({
          id,
          role: "assistant",
          type: "critic_verdict",
          content: event.feedback,
          metadata: {
            approved: event.approved,
            feedback: event.feedback,
          },
          timestamp: now,
        });
        break;

      case "completed":
        setPhase("completed");
        setIsRunning(false);
        addOrUpdateMessage({
          id,
          role: "assistant",
          type: "completed",
          content: "Analysis complete",
          metadata: {
            thread_id: event.thread_id,
            artifacts: event.artifacts,
          },
          timestamp: now,
        });
        break;

      case "error":
        setPhase("failed");
        setIsRunning(false);
        addOrUpdateMessage({
          id,
          role: "assistant",
          type: "error",
          content: event.message,
          timestamp: now,
        });
        break;
    }
  }, [addOrUpdateMessage]);

  // --- Connect to a run's WebSocket ---
  const connect = useCallback(
    async (newRunId: string) => {
      // Close any existing connection
      wsRef.current?.close();
      setRunId(newRunId);

      try {
        // Fetch current state from the bridge server
        const runState = await getRunState(newRunId);
        const now = new Date();
        const initialMessages: EegMessage[] = [];

        // Always hydrate the user directive if available
        // Always hydrate the user directive if available
        if (runState.state?.user_directive) {
          initialMessages.push({
            id: `msg-dir-${Date.now()}`,
            role: "user",
            type: "user_submission",
            content: runState.state.user_directive,
            timestamp: now,
          });
        }

        // Hydrate all history fields from state if available
        if (runState.state?.analysis_plan) {
          const isApproved = runState.state.is_approved || false;
          initialMessages.push({
            id: `msg-plan-${Date.now()}`,
            role: "assistant",
            type: isApproved ? "plan_ready" : "hitl_required",
            content: runState.state.analysis_plan,
            timestamp: now,
          });
        }

        if (runState.state?.is_approved) {
          initialMessages.push({
            id: `msg-appr-${Date.now()}`,
            role: "user",
            type: "user_submission",
            content: "✅ Plan approved",
            timestamp: now,
          });
        }

        if (runState.state?.executed_code_blocks) {
          runState.state.executed_code_blocks.forEach((block, idx) => {
            initialMessages.push({
              id: `msg-code-${idx}-${Date.now()}`,
              role: "assistant",
              type: "code_block",
              content: block.code,
              metadata: {
                code: block.code,
                logs: block.logs,
                error: block.error,
                index: idx,
              },
              timestamp: now,
            });
          });
        }

        if (runState.state?.generated_plots) {
          runState.state.generated_plots.forEach((plot, idx) => {
            initialMessages.push({
              id: `msg-plot-${idx}-${Date.now()}`,
              role: "assistant",
              type: "plot",
              content: `Plot ${idx + 1}`,
              metadata: { base64: plot, index: idx },
              timestamp: now,
            });
          });
        }

        if (runState.state?.critic_feedback) {
          initialMessages.push({
            id: `msg-critic-${Date.now()}`,
            role: "assistant",
            type: "critic_verdict",
            content: runState.state.critic_feedback,
            metadata: {
              approved: runState.state.is_approved || false,
              feedback: runState.state.critic_feedback,
            },
            timestamp: now,
          });
        }

        // Case A: Run is already completed
        if (runState.phase === "completed") {
          setMessages(initialMessages);
          setPhase("completed");
          setIsRunning(false);
          setIsConnected(false);
          return;
        }

        // Case B: Run is failed or was interrupted (backend crash / OOM)
        if (runState.phase === "failed") {
          const interruptMsg = runState.interrupted
            ? "⚠️ This run was interrupted (the backend process was terminated, likely due to an out-of-memory condition). The run cannot be resumed from this point — please start a new analysis."
            : "❌ This run failed.";
          initialMessages.push({
            id: `msg-fail-${Date.now()}`,
            role: "assistant",
            type: "error",
            content: interruptMsg,
            timestamp: now,
          });
          setMessages(initialMessages);
          setPhase("failed");
          setIsRunning(false);
          setIsConnected(false);
          return;
        }

        // Case C: Active run (Planning, executing, etc.) -> Hydrate history and start WebSocket stream
        setMessages(initialMessages);
        setPhase(runState.phase as RunPhase);
        setIsRunning(true);

        // Start WebSocket stream for real-time updates
        const client = connectStream(
          newRunId,
          handleEvent,
          () => {
            setIsConnected(false);
          },
          () => {
            setIsConnected(false);
            setIsRunning(false);
            setPhase("failed");
          }
        );

        wsRef.current = client;
        setIsConnected(true);
      } catch (err) {
        console.error("Failed to connect or hydrate state:", err);
        const now = new Date();
        setMessages([
          {
            id: `msg-err-${Date.now()}`,
            role: "assistant",
            type: "error",
            content: `❌ Could not load run state: ${err instanceof Error ? err.message : "Unknown error"}`,
            timestamp: now,
          },
        ]);
        setPhase("failed");
        setIsRunning(false);
        setIsConnected(false);
      }
    },
    [handleEvent]
  );

  // --- HITL actions ---
  const approve = useCallback((feedback?: string) => {
    wsRef.current?.send({
      type: "hitl_response",
      decision: "approve",
      feedback,
    });
    setPhase("executor");
    addOrUpdateMessage({
      id: nextId(),
      role: "user",
      type: "user_submission",
      content: feedback
        ? `✅ Plan approved with feedback: ${feedback}`
        : "✅ Plan approved",
      timestamp: new Date(),
    });
  }, [addOrUpdateMessage]);

  const reject = useCallback(() => {
    wsRef.current?.send({ type: "hitl_response", decision: "reject" });
    setPhase("failed");
    setIsRunning(false);
    addOrUpdateMessage({
      id: nextId(),
      role: "user",
      type: "user_submission",
      content: "❌ Plan rejected",
      timestamp: new Date(),
    });
  }, [addOrUpdateMessage]);

  const cancel = useCallback(() => {
    wsRef.current?.send({ type: "cancel" });
    wsRef.current?.close();
    setIsRunning(false);
    setPhase("failed");
  }, []);

  const codeBlocks = messages.filter((m) => m.type === "code_block");
  const plots = messages.filter((m) => m.type === "plot");
  const chatMessages = messages.filter(
    (m) => m.type !== "code_block" &&
           m.type !== "plot" &&
           !(m.type === "status" && (m.metadata?.phase === "executor" || m.metadata?.phase === "critic"))
  );

  // --- assistant-ui runtime ---
  const onNew = useCallback(async () => {
    // The composer is disabled during EEG runs; the only "new message"
    // interaction is the initial form submission, handled outside the runtime.
  }, []);

  const runtime = useExternalStoreRuntime({
    isRunning,
    messages: chatMessages,
    convertMessage,
    onNew,
  });

  const sessionCtx: EegSessionContext = {
    runId,
    phase,
    isConnected,
    connect,
    approve,
    reject,
    cancel,
    codeBlocks,
    plots,
    messages,
  };

  return (
    <SessionCtx.Provider value={sessionCtx}>
      <AssistantRuntimeProvider runtime={runtime}>
        {children}
      </AssistantRuntimeProvider>
    </SessionCtx.Provider>
  );
}
