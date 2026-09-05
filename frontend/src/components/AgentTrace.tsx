import type { NodeName } from '../lib/events'
import { NODE_LABELS, TOOL_LABELS } from '../lib/events'
import type { TraceEntry } from '../lib/useAgentStream'

const NODE_ACCENT: Record<NodeName, string> = {
  gatekeeper: 'text-sky-300 border-sky-400/30 bg-sky-400/10',
  planner: 'text-violet-300 border-violet-400/30 bg-violet-400/10',
  tool_executor: 'text-amber-300 border-amber-400/30 bg-amber-400/10',
  auditor: 'text-rose-300 border-rose-400/30 bg-rose-400/10',
  synthesizer: 'text-emerald-300 border-emerald-400/30 bg-emerald-400/10',
}

function formatMs(ms?: number) {
  if (ms === undefined) return null
  return ms < 1000 ? `${ms}ms` : `${(ms / 1000).toFixed(1)}s`
}

function Spinner() {
  return (
    <span className="inline-block size-3 animate-spin rounded-full border-2 border-current border-t-transparent" />
  )
}

function ToolRow({ tool }: { tool: TraceEntry['tools'][number] }) {
  return (
    <li className="animate-trace-in rounded-lg border border-edge/70 bg-canvas/60 p-2.5">
      <div className="flex items-center gap-2">
        {tool.status === 'running' ? (
          <span className="text-amber-300">
            <Spinner />
          </span>
        ) : tool.status === 'failed' ? (
          <span className="text-rose-400">✕</span>
        ) : (
          <span className="text-emerald-400">✓</span>
        )}
        <span className="font-mono text-xs font-medium text-amber-200">
          {TOOL_LABELS[tool.toolName] ?? tool.toolName}
        </span>
        <span className="ml-auto font-mono text-[11px] text-slate-500">
          {formatMs(tool.durationMs)}
          {tool.resultCount != null && tool.resultCount > 0 && (
            <span className="ml-1.5">· {tool.resultCount} hits</span>
          )}
        </span>
      </div>
      <p className="mt-1.5 truncate font-mono text-[11px] text-slate-500" title={tool.toolInput}>
        “{tool.toolInput}”
      </p>
      {tool.preview && (
        <p className="mt-1.5 line-clamp-2 text-[11px] leading-relaxed text-slate-400">
          {tool.preview}
        </p>
      )}
      {tool.error && <p className="mt-1.5 text-[11px] text-rose-400">{tool.error}</p>}
    </li>
  )
}

function AuditVerdict({ audit }: { audit: NonNullable<TraceEntry['audit']> }) {
  const replan = audit.action === 'replan'
  return (
    <div
      className={`animate-trace-in mt-2 rounded-lg border p-2.5 ${
        replan ? 'border-rose-500/40 bg-rose-500/10' : 'border-emerald-500/40 bg-emerald-500/10'
      }`}
    >
      <div className="flex items-center gap-2">
        <span className="font-mono text-[11px] tracking-wide text-slate-300">
          confidence
        </span>
        <span className="flex gap-0.5">
          {[1, 2, 3, 4, 5].map((n) => (
            <span
              key={n}
              className={`h-3.5 w-1.5 rounded-sm ${
                n <= audit.confidenceScore
                  ? replan
                    ? 'bg-rose-400'
                    : 'bg-emerald-400'
                  : 'bg-slate-700'
              }`}
            />
          ))}
        </span>
        <span className="font-mono text-[11px] text-slate-400">{audit.confidenceScore}/5</span>
        <span
          className={`ml-auto rounded px-1.5 py-0.5 font-mono text-[10px] font-semibold uppercase tracking-wider ${
            replan ? 'bg-rose-500/20 text-rose-300' : 'bg-emerald-500/20 text-emerald-300'
          }`}
        >
          {replan ? 'replan' : 'accept'}
        </span>
      </div>
      <p className="mt-2 text-[11px] leading-relaxed text-slate-400">{audit.reasoning}</p>
    </div>
  )
}

function TraceCard({ entry, isLast }: { entry: TraceEntry; isLast: boolean }) {
  const running = entry.status === 'running'
  return (
    <li className="animate-trace-in relative pl-8">
      {/* Connector down to the next node. */}
      {!isLast && <span className="absolute left-[11px] top-6 h-[calc(100%-0.5rem)] w-px bg-edge" />}
      <span
        className={`absolute left-0 top-1 flex size-[23px] items-center justify-center rounded-full border text-[10px] ${
          NODE_ACCENT[entry.node]
        }`}
      >
        {running ? <Spinner /> : '✓'}
      </span>

      <div className="flex items-baseline gap-2">
        <h3 className="text-sm font-semibold text-slate-200">{NODE_LABELS[entry.node]}</h3>
        {entry.attempt > 1 && (
          <span className="rounded bg-rose-500/15 px-1.5 py-0.5 font-mono text-[10px] font-semibold text-rose-300">
            attempt {entry.attempt}
          </span>
        )}
        <span className="ml-auto font-mono text-[11px] text-slate-500">
          {running ? 'running…' : formatMs(entry.durationMs)}
        </span>
      </div>

      {entry.plan && (
        <ol className="mt-2 space-y-1">
          {entry.plan.map((step, i) => (
            <li key={`${step.tool_name}-${i}`} className="flex items-center gap-2 text-[11px]">
              <span className="font-mono text-slate-600">{i + 1}.</span>
              <span
                className={`font-mono ${
                  step.tool_name === 'FINISH' ? 'text-slate-500' : 'text-violet-300'
                }`}
              >
                {TOOL_LABELS[step.tool_name] ?? step.tool_name}
              </span>
              {step.tool_input && (
                <span className="truncate text-slate-500" title={step.tool_input}>
                  {step.tool_input}
                </span>
              )}
            </li>
          ))}
        </ol>
      )}

      {entry.tools.length > 0 && (
        <ul className="mt-2 space-y-1.5">
          {entry.tools.map((tool, i) => (
            <ToolRow key={`${tool.toolName}-${i}`} tool={tool} />
          ))}
        </ul>
      )}

      {entry.audit && <AuditVerdict audit={entry.audit} />}
    </li>
  )
}

export function AgentTrace({ trace }: { trace: TraceEntry[] }) {
  if (trace.length === 0) {
    return (
      <p className="px-1 text-xs leading-relaxed text-slate-500">
        Node transitions appear here as the supervisor graph runs — plans, tool calls, and the
        Auditor's verdict on each one.
      </p>
    )
  }
  return (
    <ol className="space-y-5">
      {trace.map((entry, i) => (
        <TraceCard key={entry.id} entry={entry} isLast={i === trace.length - 1} />
      ))}
    </ol>
  )
}
