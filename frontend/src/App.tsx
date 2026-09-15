import { useEffect, useRef, useState } from 'react'

import { AgentTrace } from './components/AgentTrace'
import { LatencyBreakdown } from './components/LatencyBreakdown'
import { useAgentStream } from './lib/useAgentStream'

/** Swap these if you pick a different product name. */
const PRODUCT_NAME = 'Financial Analyst Agent'
const PRODUCT_TAGLINE = 'Alphabet Inc. · FY2024 10-K'
const ASSISTANT_LABEL = 'Analyst'

const EXAMPLES = [
  {
    label: 'Revenue vs. competition',
    query:
      'Analyze Alphabet FY2024 revenue and how it relates to the competitive risks in the 10-K',
  },
  {
    label: 'A single metric',
    query: 'What was total revenue in FY2024?',
  },
  {
    label: 'Open-ended',
    query: 'How is the company doing?',
  },
]

const SPEEDS = [
  { label: '1×', value: 1, hint: 'Real latency (~22s)' },
  { label: '5×', value: 5, hint: 'Compressed' },
  { label: '20×', value: 20, hint: 'Fast iteration' },
]

function Mark() {
  return (
    <span className="flex size-9 items-center justify-center rounded-lg border border-sky-400/30 bg-sky-400/10 text-sky-200">
      <svg viewBox="0 0 24 24" className="size-[18px]" fill="none" aria-hidden="true">
        <path
          d="M7 4.5h7.2L19 9.2V19.5H7V4.5Z"
          stroke="currentColor"
          strokeWidth="1.6"
          strokeLinejoin="round"
        />
        <path d="M14.2 4.5V9.2H19" stroke="currentColor" strokeWidth="1.6" strokeLinejoin="round" />
        <path d="M9.2 13h5.6M9.2 16.2h3.8" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" />
      </svg>
    </span>
  )
}

function ConversationTurn({
  query,
  answer,
  clarification,
  errorMessage,
  streaming = false,
}: {
  query: string
  answer: string
  clarification?: string
  errorMessage?: string
  streaming?: boolean
}) {
  const paragraphs = answer ? answer.split('\n\n') : []

  return (
    <article className="space-y-4">
      <div className="flex justify-end">
        <div className="max-w-[85%]">
          <p className="mb-1.5 text-right font-sans text-[11px] font-medium tracking-wide text-slate-500">
            You
          </p>
          <p className="rounded-2xl rounded-tr-md bg-slate-800/80 px-4 py-3 text-[15px] leading-relaxed text-slate-100">
            {query}
          </p>
        </div>
      </div>

      {clarification && (
        <div className="max-w-[90%] rounded-2xl border border-sky-500/35 bg-sky-500/10 px-4 py-3">
          <p className="font-sans text-[11px] font-medium tracking-wide text-sky-300">
            Needs a more specific question
          </p>
          <p className="mt-1.5 text-[15px] leading-relaxed text-slate-200">{clarification}</p>
        </div>
      )}

      {(answer || streaming || errorMessage) && (
        <div className="max-w-[92%]">
          <p className="mb-1.5 font-sans text-[11px] font-medium tracking-wide text-emerald-400/80">
            {ASSISTANT_LABEL}
          </p>
          {answer && (
            <div className="space-y-3 rounded-2xl rounded-tl-md border border-edge bg-canvas/50 px-4 py-3">
              {paragraphs.map((para, i) => (
                <p key={i} className="text-[15px] leading-relaxed text-slate-200">
                  {para}
                  {streaming && i === paragraphs.length - 1 && (
                    <span className="ml-0.5 inline-block h-4 w-1.5 translate-y-0.5 animate-pulse bg-emerald-400" />
                  )}
                </p>
              ))}
            </div>
          )}
          {streaming && !answer && !clarification && (
            <p className="rounded-2xl border border-edge bg-canvas/40 px-4 py-3 text-sm text-slate-500">
              Working — the first question after idle can take a few minutes while the GPU
              worker starts. Watch the reasoning panel.
            </p>
          )}
          {errorMessage && (
            <div className="mt-2 rounded-2xl border border-rose-500/40 bg-rose-500/10 px-4 py-3">
              <p className="font-sans text-[11px] font-medium tracking-wide text-rose-300">
                Stream failed
              </p>
              <p className="mt-1.5 text-sm text-slate-300">{errorMessage}</p>
            </div>
          )}
        </div>
      )}
    </article>
  )
}

export default function App() {
  const { state, send, cancel } = useAgentStream()
  const [draft, setDraft] = useState(EXAMPLES[0].query)
  const [speed, setSpeed] = useState(5)
  const [agentMode, setAgentMode] = useState<'live' | 'mock'>('mock')
  const answerRef = useRef<HTMLDivElement>(null)
  const streaming = state.status === 'streaming'
  const turns = [
    ...state.history,
    ...(state.query
      ? [
          {
            id: state.runId ?? 'current',
            query: state.query,
            answer: state.answer,
            clarification: state.clarification,
            errorMessage: state.errorMessage,
          },
        ]
      : []),
  ]

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
  }, [turns, streaming])

  function submit() {
    const query = draft.trim()
    if (query && !streaming) void send(query, speed)
  }

  return (
    <div className="mx-auto flex h-full max-w-[1440px] flex-col gap-5 p-5 md:p-6">
      <header className="flex flex-wrap items-center gap-3">
        <div className="flex items-center gap-3">
          <Mark />
          <div>
            <h1 className="font-serif text-[1.65rem] leading-none tracking-tight text-slate-50">
              {PRODUCT_NAME}
            </h1>
            <p className="mt-1 text-sm text-slate-500">{PRODUCT_TAGLINE}</p>
          </div>
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

      <div className="grid min-h-0 flex-1 gap-4 lg:grid-cols-[minmax(0,1fr)_400px]">
        <main className="flex min-h-0 flex-col gap-3">
          <div className="flex min-h-0 flex-1 flex-col rounded-2xl border border-edge bg-panel/90 shadow-[inset_0_1px_0_rgba(255,255,255,0.04)]">
            <div
              ref={answerRef}
              data-chat-thread="keep"
              className="min-h-0 flex-1 overflow-y-auto px-5 py-6 sm:px-8"
            >
              {turns.length === 0 && (
                <div className="flex h-full flex-col items-center justify-center text-center">
                  <Mark />
                  <h2 className="mt-4 font-serif text-2xl text-slate-100">
                    Ask the Agent
                  </h2>
                  <p className="mt-2 max-w-md text-[15px] leading-relaxed text-slate-500">
                    The agent plans tools, audits its own evidence, and replans when a
                    call actually fails. Answers stay grounded in Alphabet's filing.
                  </p>
                  <div className="mt-6 flex w-full max-w-lg flex-col gap-2">
                    {EXAMPLES.map((example) => (
                      <button
                        key={example.query}
                        type="button"
                        onClick={() => setDraft(example.query)}
                        className="group rounded-xl border border-edge bg-canvas/40 px-4 py-3 text-left transition hover:border-sky-400/40 hover:bg-sky-400/5"
                      >
                        <span className="block text-[12px] font-medium tracking-wide text-slate-500 group-hover:text-sky-300">
                          {example.label}
                        </span>
                        <span className="mt-1 block text-[15px] leading-snug text-slate-300">
                          {example.query}
                        </span>
                      </button>
                    ))}
                  </div>
                </div>
              )}

              {turns.length > 0 && (
                <div className="mx-auto max-w-3xl space-y-8">
                  {turns.map((turn, index) => (
                    <ConversationTurn
                      key={turn.id}
                      query={turn.query}
                      answer={turn.answer}
                      clarification={turn.clarification}
                      errorMessage={turn.errorMessage}
                      streaming={streaming && index === turns.length - 1}
                    />
                  ))}
                </div>
              )}
            </div>
          </div>

          <div className="rounded-2xl border border-edge bg-panel p-3 shadow-[inset_0_1px_0_rgba(255,255,255,0.04)] focus-within:border-sky-400/35">
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
              className="w-full resize-none bg-transparent px-3 py-2 text-[15px] text-slate-200 placeholder:text-slate-600 focus:outline-none"
            />
            <div className="flex items-center gap-2 px-2 pb-0.5">
              <span className="text-[11px] text-slate-600">
                Enter to send · Shift+Enter for newline
              </span>
              {streaming ? (
                <button
                  type="button"
                  onClick={cancel}
                  className="ml-auto rounded-lg border border-edge px-4 py-1.5 text-sm font-medium text-slate-300 transition hover:border-rose-500/50 hover:text-rose-300"
                >
                  Stop
                </button>
              ) : (
                <button
                  type="button"
                  onClick={submit}
                  disabled={!draft.trim()}
                  className="ml-auto rounded-lg bg-sky-200 px-4 py-1.5 text-sm font-semibold text-slate-950 transition hover:bg-white disabled:cursor-not-allowed disabled:opacity-40"
                >
                  Send
                </button>
              )}
            </div>
          </div>
        </main>

        <aside className="flex min-h-0 flex-col gap-3">
          <div className="flex min-h-0 flex-1 flex-col rounded-2xl border border-edge bg-panel/90">
            <header className="flex items-center gap-2 border-b border-edge px-4 py-3">
              <h2 className="text-[11px] font-semibold uppercase tracking-[0.14em] text-slate-400">
                Reasoning
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
