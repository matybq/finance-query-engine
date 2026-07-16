import type { Route, TraceEvent } from "./api";

interface NodeSpec {
  x: number;
  y: number;
  w: number;
  label: string;
  tone?: "ok" | "warn";
}

const NODE_H = 20;

const NODES: Record<string, NodeSpec> = {
  router: { x: 130, y: 18, w: 50, label: "router" },
  direct: { x: 36, y: 78, w: 48, label: "direct" },
  out_of_scope: { x: 122, y: 78, w: 80, label: "out of scope", tone: "warn" },
  retrieve: { x: 212, y: 78, w: 60, label: "retrieve" },
  grade: { x: 212, y: 140, w: 48, label: "grade" },
  rewrite: { x: 122, y: 140, w: 56, label: "rewrite" },
  generate: { x: 212, y: 202, w: 60, label: "generate", tone: "ok" },
  insufficient_evidence: { x: 92, y: 202, w: 118, label: "insufficient evidence", tone: "warn" },
};

interface EdgeSpec {
  from: string;
  to: string;
  d: string;
}

const EDGES: EdgeSpec[] = [
  { from: "router", to: "direct", d: "M116,28 L44,65" },
  { from: "router", to: "out_of_scope", d: "M130,28 L123,65" },
  { from: "router", to: "retrieve", d: "M144,28 L206,65" },
  { from: "retrieve", to: "grade", d: "M212,88 L212,127" },
  { from: "grade", to: "rewrite", d: "M188,140 L153,140" },
  { from: "rewrite", to: "retrieve", d: "M122,130 C122,106 156,96 178,88" },
  { from: "grade", to: "generate", d: "M212,150 L212,189" },
  { from: "grade", to: "insufficient_evidence", d: "M196,150 L112,189" },
];

function visitedSequence(
  trace: TraceEvent[],
  route: Route | null,
  hasTokens: boolean,
  finished: boolean,
): string[] {
  const sequence = ["router"];
  for (const step of trace) {
    switch (step.type) {
      case "route":
        sequence.push(step.route);
        break;
      case "retrieve":
        if (sequence[sequence.length - 1] !== "retrieve") sequence.push("retrieve");
        break;
      case "grade":
        sequence.push("grade");
        break;
      case "rewrite":
        sequence.push("rewrite");
        break;
    }
  }
  if (route === "retrieve") {
    const lastGrade = [...trace].reverse().find((step) => step.type === "grade");
    const generating = hasTokens || (finished && lastGrade?.sufficient);
    if (generating && sequence[sequence.length - 1] !== "generate") sequence.push("generate");
    if (finished && lastGrade && !lastGrade.sufficient) sequence.push("insufficient_evidence");
  }
  return sequence;
}

interface AgentGraphProps {
  trace: TraceEvent[];
  route: Route | null;
  hasTokens: boolean;
  finished: boolean;
}

export function AgentGraph({ trace, route, hasTokens, finished }: AgentGraphProps) {
  const sequence = visitedSequence(trace, route, hasTokens, finished);
  const visited = new Set(sequence);
  const active = finished ? null : sequence[sequence.length - 1];

  const traversed = new Set<string>();
  for (let i = 1; i < sequence.length; i++) {
    traversed.add(`${sequence[i - 1]}->${sequence[i]}`);
  }

  function nodeClass(id: string): string {
    const spec = NODES[id];
    const parts = ["gnode"];
    if (visited.has(id)) {
      parts.push("on");
      if (spec.tone) parts.push(spec.tone);
    } else if (finished) {
      parts.push("ghost");
    }
    if (id === active) parts.push("active");
    return parts.join(" ");
  }

  function edgeClass(edge: EdgeSpec): string {
    if (traversed.has(`${edge.from}->${edge.to}`)) return "gedge on";
    return finished ? "gedge ghost" : "gedge";
  }

  return (
    <svg className="agent-graph" viewBox="0 0 260 224" role="img" aria-label="Agent graph trace">
      <defs>
        <marker id="arrow" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
          <path d="M0,0 L8,4 L0,8 z" />
        </marker>
      </defs>
      {EDGES.map((edge) => (
        <path
          key={`${edge.from}-${edge.to}`}
          className={edgeClass(edge)}
          d={edge.d}
          pathLength={1}
          markerEnd="url(#arrow)"
        />
      ))}
      {Object.entries(NODES).map(([id, spec]) => (
        <g key={id} className={nodeClass(id)}>
          <rect
            x={spec.x - spec.w / 2}
            y={spec.y - NODE_H / 2}
            width={spec.w}
            height={NODE_H}
            rx={3}
          />
          <text x={spec.x} y={spec.y + 3}>
            {spec.label}
          </text>
        </g>
      ))}
    </svg>
  );
}
