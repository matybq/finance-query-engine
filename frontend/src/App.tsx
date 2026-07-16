import { useState, type FormEvent } from "react";
import { AgentGraph } from "./AgentGraph";
import { askStream, type Route, type SourceRef, type TraceEvent } from "./api";

const SAMPLE_QUESTIONS = [
  "What is AirCover for Hosts?",
  "How does Airbnb describe its competition?",
  "What drove revenue growth in fiscal 2025?",
  "What was Tesla's revenue in 2025?",
];

const ROUTE_LABELS: Record<Route, string> = {
  retrieve: "grounded in filing",
  direct: "direct answer",
  out_of_scope: "out of scope",
};

type Phase = "idle" | "running" | "done" | "error";

interface TraceLine {
  label: string;
  detail: string;
  tone?: "ok" | "warn";
}

function traceLine(step: TraceEvent): TraceLine {
  switch (step.type) {
    case "route":
      return { label: `router → ${step.route}`, detail: step.reason };
    case "retrieve":
      return {
        label: `retrieve → ${step.count} chunks`,
        detail: [`“${step.query}”`, ...step.sections.map((ref) => ref.section)].join(" · "),
      };
    case "grade":
      return {
        label: `grade → ${step.sufficient ? "sufficient" : "insufficient"}`,
        detail: step.reason,
        tone: step.sufficient ? "ok" : "warn",
      };
    case "rewrite":
      return { label: `rewrite → “${step.query}”`, detail: "" };
  }
}

export function App() {
  const [question, setQuestion] = useState("");
  const [phase, setPhase] = useState<Phase>("idle");
  const [asked, setAsked] = useState("");
  const [trace, setTrace] = useState<TraceEvent[]>([]);
  const [route, setRoute] = useState<Route | null>(null);
  const [streamed, setStreamed] = useState("");
  const [answer, setAnswer] = useState("");
  const [sources, setSources] = useState<SourceRef[]>([]);
  const [errorDetail, setErrorDetail] = useState("");

  async function submit(text: string) {
    const trimmed = text.trim();
    if (!trimmed || phase === "running") return;
    setPhase("running");
    setAsked(trimmed);
    setTrace([]);
    setRoute(null);
    setStreamed("");
    setAnswer("");
    setSources([]);

    let settled = false;
    try {
      await askStream(trimmed, (event) => {
        switch (event.type) {
          case "route":
            setRoute(event.route);
            setTrace((steps) => [...steps, event]);
            break;
          case "retrieve":
          case "grade":
          case "rewrite":
            setTrace((steps) => [...steps, event]);
            break;
          case "token":
            setStreamed((current) => current + event.text);
            break;
          case "done":
            settled = true;
            setRoute(event.route);
            setAnswer(event.answer);
            setSources(event.sources);
            setPhase("done");
            break;
          case "error":
            settled = true;
            setErrorDetail(event.detail);
            setPhase("error");
            break;
        }
      });
      if (!settled) throw new Error("The stream ended unexpectedly.");
    } catch (error) {
      if (!settled) {
        setErrorDetail(error instanceof Error ? error.message : "The request failed.");
        setPhase("error");
      }
    }
  }

  function onSubmit(event: FormEvent) {
    event.preventDefault();
    void submit(question);
  }

  const running = phase === "running";

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
          instead of guessing &mdash; and you can watch it decide, step by step.
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
            disabled={running}
          />
          <button type="submit" disabled={running || question.trim() === ""}>
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
                disabled={running}
              >
                {sample}
              </button>
            </li>
          ))}
        </ul>
      </form>

      {phase !== "idle" && (
        <section className={phase === "error" ? "card error" : "card"} aria-live="polite">
          {running && <div className="progress" />}
          {route !== null && (
            <div className="answer-meta">
              <span className={`badge badge-${route}`}>{ROUTE_LABELS[route]}</span>
            </div>
          )}
          <p className="asked">{asked}</p>

          {phase !== "error" && (
            <div className="run-body">
              <AgentGraph
                trace={trace}
                route={route}
                hasTokens={streamed !== ""}
                finished={phase === "done"}
              />
              <div className="run-side">
                {trace.length > 0 ? (
                  <ol className="trace">
                    {trace.map((step, index) => {
                      const line = traceLine(step);
                      return (
                        <li key={index} className={line.tone}>
                          <span className="trace-label">{line.label}</span>
                          {line.detail && <span className="trace-detail">{line.detail}</span>}
                        </li>
                      );
                    })}
                  </ol>
                ) : (
                  <p className="mono waiting">routing the question…</p>
                )}
              </div>
            </div>
          )}
          {running && streamed !== "" && (
            <p className="answer-text">
              {streamed}
              <span className="cursor" aria-hidden="true" />
            </p>
          )}
          {phase === "done" && <p className="answer-text">{answer}</p>}
          {phase === "error" && <p className="mono">{errorDetail}</p>}

          {phase === "done" && sources.length > 0 && (
            <footer className="sources">
              <p className="overline">Sources</p>
              <ol>
                {sources.map((ref, index) => (
                  <li key={`${ref.source}-${ref.section}-${index}`} className="mono">
                    {ref.source} · {ref.section}
                  </li>
                ))}
              </ol>
            </footer>
          )}
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
