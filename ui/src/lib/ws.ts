/**
 * WebSocket client for real-time graph execution streaming.
 *
 * Connects to the FastAPI bridge server and dispatches ServerEvent
 * objects to a callback. Supports sending ClientCommand messages
 * for HITL responses and cancellation.
 */

import type { ClientCommand, ServerEvent } from "./types";
import { getStreamUrl } from "./api";

export type EventCallback = (event: ServerEvent) => void;

export interface WsClient {
  /** Send a command to the server (HITL response or cancel). */
  send: (command: ClientCommand) => void;
  /** Close the WebSocket connection. */
  close: () => void;
  /** Whether the connection is currently open. */
  isConnected: () => boolean;
}

/**
 * Connect to the run's WebSocket stream.
 *
 * @param runId - The run/thread ID to stream.
 * @param onEvent - Callback invoked for each ServerEvent.
 * @param onClose - Optional callback when the connection closes.
 * @param onError - Optional callback for connection errors.
 * @returns A WsClient handle for sending commands and closing.
 */
export function connectStream(
  runId: string,
  onEvent: EventCallback,
  onClose?: () => void,
  onError?: (error: Event) => void
): WsClient {
  const url = getStreamUrl(runId);
  const ws = new WebSocket(url);

  ws.onmessage = (event) => {
    try {
      const data: ServerEvent = JSON.parse(event.data);
      onEvent(data);
    } catch (e) {
      console.error("[ws] Failed to parse server event:", e);
    }
  };

  ws.onclose = () => {
    onClose?.();
  };

  ws.onerror = (event) => {
    console.error("[ws] WebSocket error:", event);
    onError?.(event);
  };

  return {
    send: (command: ClientCommand) => {
      if (ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify(command));
      } else {
        console.warn("[ws] Cannot send — socket not open");
      }
    },
    close: () => {
      ws.close();
    },
    isConnected: () => ws.readyState === WebSocket.OPEN,
  };
}
