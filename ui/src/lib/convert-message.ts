/**
 * Converts EegMessage objects (our internal format) into assistant-ui's
 * ThreadMessageLike format for rendering in the Thread component.
 */

import type { ThreadMessageLike } from "@assistant-ui/react";
import type { EegMessage } from "./types";

export function convertMessage(message: EegMessage): ThreadMessageLike {
  const base = {
    id: message.id,
    createdAt: message.timestamp,
  };

  if (message.role === "user") {
    return {
      ...base,
      role: "user" as const,
      content: [{ type: "text" as const, text: message.content }],
    };
  }

  // Assistant messages — map by event type
  const meta = message.metadata || {};
  const eventType = message.type;

  switch (eventType) {
    case "plan_ready":
    case "hitl_required":
      return {
        ...base,
        role: "assistant" as const,
        content: [
          {
            type: "tool-call" as const,
            toolCallId: `plan-${message.id}`,
            toolName: "plan_review",
            args: {
              id: message.id,
              plan: message.content,
              requiresAction: eventType === "hitl_required",
            },
          },
        ],
      };

    case "code_block":
      return {
        ...base,
        role: "assistant" as const,
        content: [
          {
            type: "tool-call" as const,
            toolCallId: `code-${message.id}`,
            toolName: "code_execution",
            args: {
              code: meta.code as string,
              logs: meta.logs as string,
              error: meta.error as boolean,
              index: meta.index as number,
            },
          },
        ],
      };

    case "plot":
      return {
        ...base,
        role: "assistant" as const,
        content: [
          {
            type: "tool-call" as const,
            toolCallId: `plot-${message.id}`,
            toolName: "plot_display",
            args: {
              base64: meta.base64 as string,
              index: meta.index as number,
            },
          },
        ],
      };

    case "critic_verdict":
      return {
        ...base,
        role: "assistant" as const,
        content: [
          {
            type: "tool-call" as const,
            toolCallId: `critic-${message.id}`,
            toolName: "critic_verdict",
            args: {
              approved: meta.approved as boolean,
              feedback: message.content,
            },
          },
        ],
      };

    case "status":
      return {
        ...base,
        role: "assistant" as const,
        content: [{ type: "text" as const, text: message.content }],
        metadata: { custom: { phase: meta.phase } },
      };

    case "completed":
      return {
        ...base,
        role: "assistant" as const,
        content: [
          {
            type: "text" as const,
            text: `✅ Analysis complete! Thread ID: \`${meta.thread_id}\``,
          },
        ],
        metadata: { custom: { artifacts: meta.artifacts } },
      };

    case "error":
      return {
        ...base,
        role: "assistant" as const,
        content: [
          { type: "text" as const, text: `❌ Error: ${message.content}` },
        ],
      };

    default:
      return {
        ...base,
        role: "assistant" as const,
        content: [{ type: "text" as const, text: message.content }],
      };
  }
}
