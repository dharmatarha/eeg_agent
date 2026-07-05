"use client";

/**
 * EegThread — Custom Thread component built from assistant-ui primitives.
 *
 * Uses ThreadPrimitive to render the message stream, with custom tool UIs
 * for domain-specific EEG components (plan review, code blocks, plots, critic).
 */

import {
  ThreadPrimitive,
  MessagePrimitive,
  useAssistantToolUI,
} from "@assistant-ui/react";
import { PlanReviewCard } from "@/components/plan-review-card";
import { CodeTrace } from "@/components/code-trace";
import { PlotGallery } from "@/components/plot-gallery";
import { CriticVerdict } from "@/components/critic-verdict";

// --- Register Tool UIs ---

function ToolUIs() {
  useAssistantToolUI({
    toolName: "plan_review",
    render: ({ args }) => (
      <PlanReviewCard
        plan={args.plan as string}
        requiresAction={args.requiresAction as boolean}
      />
    ),
  });

  useAssistantToolUI({
    toolName: "code_execution",
    render: ({ args }) => (
      <CodeTrace
        code={args.code as string}
        logs={args.logs as string}
        error={args.error as boolean}
        index={args.index as number}
      />
    ),
  });

  useAssistantToolUI({
    toolName: "plot_display",
    render: ({ args }) => (
      <PlotGallery
        base64={args.base64 as string}
        index={args.index as number}
      />
    ),
  });

  useAssistantToolUI({
    toolName: "critic_verdict",
    render: ({ args }) => (
      <CriticVerdict
        approved={args.approved as boolean}
        feedback={args.feedback as string}
      />
    ),
  });

  return null;
}

// --- Message components ---

function AssistantMessage() {
  return (
    <MessagePrimitive.Root className="py-3 px-4">
      <div className="space-y-3 max-w-3xl">
        <MessagePrimitive.Content
          components={{
            Text: ({ text }) => (
              <p className="text-sm text-slate-300 leading-relaxed whitespace-pre-wrap">
                {text}
              </p>
            ),
          }}
        />
      </div>
    </MessagePrimitive.Root>
  );
}

function UserMessage() {
  return (
    <MessagePrimitive.Root className="py-3 px-4 flex justify-end">
      <div className="max-w-md bg-indigo-600/20 border border-indigo-500/30 rounded-xl px-4 py-2.5">
        <MessagePrimitive.Content
          components={{
            Text: ({ text }) => (
              <p className="text-sm text-indigo-200">{text}</p>
            ),
          }}
        />
      </div>
    </MessagePrimitive.Root>
  );
}

// --- Main Thread ---

export function EegThread() {
  return (
    <ThreadPrimitive.Root className="flex flex-col h-full">
      <ToolUIs />

      <ThreadPrimitive.Viewport className="flex-1 overflow-y-auto">
        <ThreadPrimitive.Empty>
          <div className="flex items-center justify-center h-full">
            <div className="text-center space-y-3 p-8">
              <div className="animate-spin rounded-full h-8 w-8 border-2 border-indigo-400 border-t-transparent mx-auto" />
              <p className="text-sm text-slate-400">
                Connecting to analysis pipeline...
              </p>
            </div>
          </div>
        </ThreadPrimitive.Empty>

        <ThreadPrimitive.Messages
          components={{
            AssistantMessage,
            UserMessage,
          }}
        />

        <ThreadPrimitive.ViewportFooter>
          <div className="h-4" />
        </ThreadPrimitive.ViewportFooter>
      </ThreadPrimitive.Viewport>
    </ThreadPrimitive.Root>
  );
}
