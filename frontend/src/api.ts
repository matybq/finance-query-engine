export type Route = "retrieve" | "direct" | "out_of_scope";

export interface SourceRef {
  source: string;
  section: string;
}

export interface AskResponse {
  answer: string;
  sources: SourceRef[];
  route: Route;
}

export async function ask(question: string): Promise<AskResponse> {
  const response = await fetch("/api/ask", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question }),
  });
  if (!response.ok) {
    const detail = await response
      .json()
      .then((body: { detail?: unknown }) => body.detail)
      .catch(() => null);
    throw new Error(typeof detail === "string" ? detail : `Request failed (${response.status})`);
  }
  return response.json();
}
