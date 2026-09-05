import type { NodeName, RunEndEvent } from '../lib/events'
import { NODE_LABELS, NODE_ORDER } from '../lib/events'

const NODE_BAR: Record<NodeName, string> = {
  gatekeeper: 'bg-sky-400',
  planner: 'bg-violet-400',
  tool_executor: 'bg-amber-400',
  auditor: 'bg-rose-400',
  synthesizer: 'bg-emerald-400',
}

/**
 * Per-node wall time for the run. The point is to make the expensive node
 * obvious at a glance — "the Auditor is 40% of latency" is only actionable if
 * you can see it without reading logs.
 */
export function LatencyBreakdown({ summary }: { summary: RunEndEvent }) {
  const total = summary.duration_ms || 1
  const nodes = NODE_ORDER.filter((node) => (summary.node_timings_ms[node] ?? 0) > 0)

  return (
    <section className="rounded-xl border border-edge bg-panel p-4">
      <header className="flex items-baseline gap-2">
        <h3 className="text-xs font-semibold uppercase tracking-wider text-slate-400">
          Latency breakdown
        </h3>
        <span className="ml-auto font-mono text-sm font-semibold text-slate-200">
          {(summary.duration_ms / 1000).toFixed(1)}s
        </span>
      </header>

      <div className="mt-3 flex h-2 overflow-hidden rounded-full bg-canvas">
        {nodes.map((node) => (
          <div
            key={node}
            className={NODE_BAR[node]}
            style={{ width: `${((summary.node_timings_ms[node] ?? 0) / total) * 100}%` }}
            title={`${NODE_LABELS[node]}: ${summary.node_timings_ms[node]}ms`}
          />
        ))}
      </div>

      <ul className="mt-3 space-y-1.5">
        {nodes
          .slice()
          .sort((a, b) => (summary.node_timings_ms[b] ?? 0) - (summary.node_timings_ms[a] ?? 0))
          .map((node) => {
            const ms = summary.node_timings_ms[node] ?? 0
            return (
              <li key={node} className="flex items-center gap-2 text-xs">
                <span className={`size-2 rounded-sm ${NODE_BAR[node]}`} />
                <span className="text-slate-400">{NODE_LABELS[node]}</span>
                <span className="ml-auto font-mono text-slate-300">
                  {(ms / 1000).toFixed(1)}s
                </span>
                <span className="w-10 text-right font-mono text-slate-500">
                  {Math.round((ms / total) * 100)}%
                </span>
              </li>
            )
          })}
      </ul>

      <dl className="mt-4 grid grid-cols-3 gap-2 border-t border-edge pt-3">
        {[
          ['LLM calls', summary.llm_calls],
          ['Out tokens', summary.output_tokens],
          ['Replans', summary.replans],
        ].map(([label, value]) => (
          <div key={label as string}>
            <dt className="text-[10px] uppercase tracking-wider text-slate-500">{label}</dt>
            <dd className="font-mono text-sm text-slate-200">{value}</dd>
          </div>
        ))}
      </dl>
    </section>
  )
}
