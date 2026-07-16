import { useState, type FormEvent } from "react";
import { ask, type AskResponse, type Route } from "./api";

const SAMPLE_QUESTIONS = [
  "What is AirCover for Hosts?",
  "How does Airbnb describe its competition?",
  "What drove revenue growth in fiscal 2025?",
];

const ROUTE_LABELS: Record<Route, string> = {
  retrieve: "grounded in filing",
  direct: "direct answer",
  out_of_scope: "out of scope",
};

type Result =
  | { kind: "answer"; question: string; response: AskResponse }
  | { kind: "error"; question: string; message: string };

export function App() {
  const [question, setQuestion] = useState("");
  const [pending, setPending] = useState<string | null>(null);
  const [result, setResult] = useState<Result | null>(null);

  async function submit(text: string) {
    const trimmed = text.trim();
    if (!trimmed || pending !== null) return;
    setPending(trimmed);
    setResult(null);
    try {
      const response = await ask(trimmed);
      setResult({ kind: "answer", question: trimmed, response });
    } catch (error) {
      const message = error instanceof Error ? error.message : "The request failed.";
      setResult({ kind: "error", question: trimmed, message });
    } finally {
      setPending(null);
    }
  }

  function onSubmit(event: FormEvent) {
    event.preventDefault();
    void submit(question);
  }

  return (
    <div className="page">
      <header className="masthead">
        <p className="overline">Finance Query Engine</p>
        <h1>
          Ask the filing,
          <br />
          not the model.
        </h1>
        <p className="lede">
          Evidence-grounded answers over the <strong>Airbnb 10-K</strong> (fiscal year 2025,
          SEC&nbsp;EDGAR). When the filing doesn&rsquo;t support an answer, the agent refuses
          instead of guessing.
        </p>
      </header>

      <form className="ask" onSubmit={onSubmit}>
        <label className="overline" htmlFor="question">
          Question
        </label>
        <div className="ask-row">
          <input
            id="question"
            type="text"
            value={question}
            onChange={(event) => setQuestion(event.target.value)}
            placeholder="e.g. What risks does Airbnb list around regulation?"
            autoComplete="off"
            disabled={pending !== null}
          />
          <button type="submit" disabled={pending !== null || question.trim() === ""}>
            Ask
          </button>
        </div>
        <ul className="samples">
          {SAMPLE_QUESTIONS.map((sample) => (
            <li key={sample}>
              <button
                type="button"
                onClick={() => {
                  setQuestion(sample);
                  void submit(sample);
                }}
                disabled={pending !== null}
              >
                {sample}
              </button>
            </li>
          ))}
        </ul>
      </form>

      {pending !== null && (
        <section className="card pending" aria-live="polite">
          <div className="progress" />
          <p className="mono">Consulting the filing — retrieval, grading, generation…</p>
        </section>
      )}

      {result?.kind === "answer" && (
        <section className="card answer" aria-live="polite">
          <div className="answer-meta">
            <span className={`badge badge-${result.response.route}`}>
              {ROUTE_LABELS[result.response.route]}
            </span>
          </div>
          <p className="asked">{result.question}</p>
          <p className="answer-text">{result.response.answer}</p>
          {result.response.sources.length > 0 && (
            <footer className="sources">
              <p className="overline">Sources</p>
              <ol>
                {result.response.sources.map((ref, index) => (
                  <li key={`${ref.source}-${ref.section}-${index}`} className="mono">
                    {ref.source} · {ref.section}
                  </li>
                ))}
              </ol>
            </footer>
          )}
        </section>
      )}

      {result?.kind === "error" && (
        <section className="card error" aria-live="polite">
          <p className="asked">{result.question}</p>
          <p className="mono">{result.message}</p>
        </section>
      )}

      <footer className="colophon mono">
        <a href="/api/docs">API docs</a>
        <span aria-hidden="true">/</span>
        <a href="https://github.com/matybq/finance-query-engine">Source</a>
      </footer>
    </div>
  );
}
