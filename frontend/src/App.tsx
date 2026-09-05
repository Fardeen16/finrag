import { useEffect, useRef, useState } from 'react'

import { AgentTrace } from './components/AgentTrace'
import { LatencyBreakdown } from './components/LatencyBreakdown'
import { useAgentStream } from './lib/useAgentStream'

const EXAMPLES = [
  'Analyze Alphabet FY2024 revenue and how it relates to the competitive risks in the 10-K',
  'What was total revenue in FY2024?',
  'How is the company doing?',
]

const SPEEDS = [
  { label: '1×', value: 1, hint: 'Real latency (~22s)' },
  { label: '5×', value: 5, hint: 'Compressed' },
  { label: '20×', value: 20, hint: 'Fast iteration' },
]

export default function App() {
  const { state, send, cancel } = useAgentStream()
  const [draft, setDraft] = useState(EXAMPLES[0])
  const [speed, setSpeed] = useState(5)
  const [agentMode, setAgentMode] = useState<'live' | 'mock'>('mock')
  const answerRef = useRef<HTMLDivElement>(null)
  const streaming = state.status === 'streaming'

  useEffect(() => {
    void fetch('/api/health')
      .then((res) => res.json())
      .then((body: { agent?: string }) => {
        if (body.agent === 'live') setAgentMode('live')
      })
      .catch(() => undefined)
  }, [])

  useEffect(() => {
    answerRef.current?.scrollTo({ top: answerRef.current.scrollHeight })
  }, [state.answer])

  function submit() {
    const query = draft.trim()
    if (query && !streaming) void send(query, speed)
  }

  return (
    <div className="mx-auto flex h-full max-w-[1400px] flex-col gap-4 p-5">
      <header className="flex flex-wrap items-center gap-3">
        <div>
          <h1 className="text-lg font-semibold tracking-tight text-slate-100">
            FinRAG
            <span className="ml-2 text-sm font-normal text-slate-500">
              Agentic analysis of Alphabet Inc.
            </span>
          </h1>
        </div>
        <div className="ml-auto flex items-center gap-3">
          <span
            className={`rounded-full border px-2.5 py-1 font-mono text-[11px] ${
              agentMode === 'live'
                ? 'border-emerald-400/30 bg-emerald-400/10 text-emerald-300'
                : 'border-amber-400/30 bg-amber-400/10 text-amber-300'
            }`}
          >
            {agentMode === 'live' ? 'live · RunPod' : 'mock agent'}
          </span>
          {agentMode === 'mock' && (
          <div className="flex items-center gap-1 rounded-lg border border-edge bg-panel p-1">
            {SPEEDS.map((option) => (
              <button
                key={option.value}
                type="button"
                title={option.hint}
                onClick={() => setSpeed(option.value)}
                className={`rounded px-2 py-1 font-mono text-[11px] transition ${
                  speed === option.value
                    ? 'bg-slate-700 text-slate-100'
                    : 'text-slate-500 hover:text-slate-300'
                }`}
              >
                {option.label}
              </button>
            ))}
          </div>
          )}
        </div>
      </header>

      <div className="grid min-h-0 flex-1 gap-4 lg:grid-cols-[1fr_420px]">
        {/* --- Conversation ------------------------------------------------ */}
        <main className="flex min-h-0 flex-col gap-3">
          <div className="flex min-h-0 flex-1 flex-col rounded-xl border border-edge bg-panel">
            {state.query && (
              <div className="border-b border-edge px-5 py-3.5">
                <p className="text-sm leading-relaxed text-slate-300">{state.query}</p>
              </div>
            )}

            <div ref={answerRef} className="min-h-0 flex-1 overflow-y-auto px-5 py-4">
              {state.status === 'idle' && (
                <div className="flex h-full flex-col items-center justify-center gap-4 text-center">
                  <p className="max-w-sm text-sm leading-relaxed text-slate-500">
                    Ask a question about Alphabet's filings. The supervisor graph plans, calls
                    tools, audits its own output, and replans when the audit fails.
                  </p>
                  <div className="flex flex-col gap-1.5">
                    {EXAMPLES.map((example) => (
                      <button
                        key={example}
                        type="button"
                        onClick={() => setDraft(example)}
                        className="max-w-md rounded-lg border border-edge px-3 py-2 text-left text-xs text-slate-400 transition hover:border-slate-600 hover:text-slate-200"
                      >
                        {example}
                      </button>
                    ))}
                  </div>
                </div>
              )}

              {state.clarification && (
                <div className="rounded-lg border border-sky-500/40 bg-sky-500/10 p-4">
                  <p className="font-mono text-[10px] uppercase tracking-wider text-sky-300">
                    Gatekeeper needs clarification
                  </p>
                  <p className="mt-2 text-sm leading-relaxed text-slate-200">
                    {state.clarification}
                  </p>
                </div>
              )}

              {state.answer && (
                <div className="space-y-4">
                  {state.answer.split('\n\n').map((para, i) => (
                    <p key={i} className="text-sm leading-relaxed text-slate-200">
                      {para}
                      {streaming && i === state.answer.split('\n\n').length - 1 && (
                        <span className="ml-0.5 inline-block h-4 w-1.5 translate-y-0.5 animate-pulse bg-emerald-400" />
                      )}
                    </p>
                  ))}
                </div>
              )}

              {streaming && !state.answer && !state.clarification && (
                <p className="text-xs text-slate-500">
                  Working — first question after idle can take a few minutes while
                  the GPU worker starts. Watch the trace panel.
                </p>
              )}

              {state.errorMessage && (
                <div className="rounded-lg border border-rose-500/40 bg-rose-500/10 p-4">
                  <p className="font-mono text-[10px] uppercase tracking-wider text-rose-300">
                    Stream failed
                  </p>
                  <p className="mt-2 text-sm text-slate-300">{state.errorMessage}</p>
                </div>
              )}
            </div>
          </div>

          {/* --- Composer -------------------------------------------------- */}
          <div className="rounded-xl border border-edge bg-panel p-2.5">
            <textarea
              value={draft}
              onChange={(event) => setDraft(event.target.value)}
              onKeyDown={(event) => {
                if (event.key === 'Enter' && !event.shiftKey) {
                  event.preventDefault()
                  submit()
                }
              }}
              rows={2}
              placeholder="Ask about revenue, margins, risk factors…"
              className="w-full resize-none bg-transparent px-2 py-1.5 text-sm text-slate-200 placeholder:text-slate-600 focus:outline-none"
            />
            <div className="flex items-center gap-2 px-1">
              <span className="font-mono text-[10px] text-slate-600">
                Enter to send · Shift+Enter for newline
              </span>
              {streaming ? (
                <button
                  type="button"
                  onClick={cancel}
                  className="ml-auto rounded-lg border border-edge px-3.5 py-1.5 text-xs font-medium text-slate-300 transition hover:border-rose-500/50 hover:text-rose-300"
                >
                  Stop
                </button>
              ) : (
                <button
                  type="button"
                  onClick={submit}
                  disabled={!draft.trim()}
                  className="ml-auto rounded-lg bg-slate-100 px-3.5 py-1.5 text-xs font-semibold text-slate-900 transition hover:bg-white disabled:cursor-not-allowed disabled:opacity-40"
                >
                  Send
                </button>
              )}
            </div>
          </div>
        </main>

        {/* --- Trace panel -------------------------------------------------- */}
        <aside className="flex min-h-0 flex-col gap-3">
          <div className="flex min-h-0 flex-1 flex-col rounded-xl border border-edge bg-panel">
            <header className="flex items-center gap-2 border-b border-edge px-4 py-3">
              <h2 className="text-xs font-semibold uppercase tracking-wider text-slate-400">
                Agent trace
              </h2>
              {streaming && (
                <span className="size-1.5 animate-pulse rounded-full bg-emerald-400" />
              )}
              {state.runId && (
                <span className="ml-auto font-mono text-[10px] text-slate-600">{state.runId}</span>
              )}
            </header>
            <div className="min-h-0 flex-1 overflow-y-auto p-4">
              <AgentTrace trace={state.trace} />
            </div>
          </div>
          {state.summary && <LatencyBreakdown summary={state.summary} />}
        </aside>
      </div>
    </div>
  )
}
