export type Route = "retrieve" | "direct" | "out_of_scope";

export interface SourceRef {
  source: string;
  section: string;
}

export type AgentEvent =
  | { type: "route"; route: Route; reason: string }
  | { type: "retrieve"; query: string; count: number; sections: SourceRef[] }
  | { type: "grade"; sufficient: boolean; reason: string }
  | { type: "rewrite"; query: string }
  | { type: "token"; text: string }
  | { type: "done"; answer: string; sources: SourceRef[]; route: Route }
  | { type: "error"; detail: string };

export type TraceEvent = Extract<AgentEvent, { type: "route" | "retrieve" | "grade" | "rewrite" }>;

export async function askStream(
  question: string,
  onEvent: (event: AgentEvent) => void,
): Promise<void> {
  const response = await fetch("/api/ask/stream", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question }),
  });
  if (!response.ok || response.body === null) {
    throw new Error(`Request failed (${response.status})`);
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const frames = buffer.split("\n\n");
    buffer = frames.pop() ?? "";
    for (const frame of frames) {
      const data = frame.split("\n").find((line) => line.startsWith("data: "));
      if (data) onEvent(JSON.parse(data.slice(6)) as AgentEvent);
    }
  }
}
