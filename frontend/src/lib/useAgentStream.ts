import { useCallback, useReducer, useRef } from 'react'

import type {
  AgentEvent,
  NodeName,
  PlanStep,
  RunEndEvent,
} from './events'
import { createSSEParser } from './sse'

export interface ToolCall {
  toolName: string
  toolInput: string
  status: 'running' | 'done' | 'failed'
  durationMs?: number
  preview?: string | null
  resultCount?: number | null
  error?: string | null
}

/** One visit to a node. A node re-entered after a replan gets its own entry. */
export interface TraceEntry {
  id: string
  node: NodeName
  attempt: number
  status: 'running' | 'done'
  durationMs?: number
  plan?: PlanStep[]
  audit?: {
    confidenceScore: number
    isRelevant: boolean
    reasoning: string
    action: 'accept' | 'replan'
  }
  tools: ToolCall[]
}

export interface ChatTurn {
  id: string
  query: string
  answer: string
  clarification?: string
  errorMessage?: string
}

export interface RunState {
  status: 'idle' | 'streaming' | 'done' | 'error'
  query: string
  runId?: string
  trace: TraceEntry[]
  answer: string
  clarification?: string
  errorMessage?: string
  summary?: RunEndEvent
  lastSeq: number
  /** Completed turns. The in-flight query lives in `query` / `answer`. */
  history: ChatTurn[]
}

const initialState: RunState = {
  status: 'idle',
  query: '',
  trace: [],
  answer: '',
  lastSeq: 0,
  history: [],
}

function archiveCurrent(state: RunState): ChatTurn[] {
  if (!state.query) return state.history
  return [
    ...state.history,
    {
      id: state.runId ?? `turn-${state.history.length}`,
      query: state.query,
      answer: state.answer,
      clarification: state.clarification,
      errorMessage: state.errorMessage,
    },
  ]
}

type Action =
  | { kind: 'reset'; query: string }
  | { kind: 'event'; event: AgentEvent }
  | { kind: 'closed' }
  | { kind: 'error'; message: string }

/** Mutate the most recent trace entry for `node`, returning a new array. */
function patchLatest(
  trace: TraceEntry[],
  node: NodeName,
  patch: (entry: TraceEntry) => TraceEntry,
): TraceEntry[] {
  for (let i = trace.length - 1; i >= 0; i -= 1) {
    if (trace[i].node === node) {
      const next = trace.slice()
      next[i] = patch(trace[i])
      return next
    }
  }
  return trace
}

function reducer(state: RunState, action: Action): RunState {
  switch (action.kind) {
    case 'reset':
      return {
        ...initialState,
        status: 'streaming',
        query: action.query,
        history: archiveCurrent(state),
      }

    case 'closed':
      return state.status === 'streaming' ? { ...state, status: 'done' } : state

    case 'error':
      return {
        ...state,
        status: 'error',
        errorMessage: action.message,
        trace: state.trace.map((entry) =>
          entry.status === 'running' ? { ...entry, status: 'done' } : entry,
        ),
      }

    case 'event': {
      const event = action.event
      // Duplicates and replays are dropped rather than applied twice; an
      // out-of-order node transition would render the graph walk wrong.
      if (event.seq <= state.lastSeq) return state
      const base = { ...state, lastSeq: event.seq }

      switch (event.type) {
        case 'run_start':
          return { ...base, runId: event.run_id }

        case 'node_start':
          return {
            ...base,
            trace: [
              ...base.trace,
              {
                id: `${event.node}-${event.attempt}-${event.seq}`,
                node: event.node,
                attempt: event.attempt,
                status: 'running',
                tools: [],
              },
            ],
          }

        case 'node_end':
          return {
            ...base,
            trace: patchLatest(base.trace, event.node, (entry) => ({
              ...entry,
              status: 'done',
              durationMs: event.duration_ms,
            })),
          }

        case 'plan':
          return {
            ...base,
            trace: patchLatest(base.trace, 'planner', (entry) => ({
              ...entry,
              plan: event.steps,
            })),
          }

        case 'tool_start':
          return {
            ...base,
            trace: patchLatest(base.trace, 'tool_executor', (entry) => ({
              ...entry,
              tools: [
                ...entry.tools,
                {
                  toolName: event.tool_name,
                  toolInput: event.tool_input,
                  status: 'running',
                },
              ],
            })),
          }

        case 'tool_end':
          return {
            ...base,
            trace: patchLatest(base.trace, 'tool_executor', (entry) => {
              const tools = entry.tools.slice()
              for (let i = tools.length - 1; i >= 0; i -= 1) {
                if (tools[i].toolName === event.tool_name && tools[i].status === 'running') {
                  tools[i] = {
                    ...tools[i],
                    status: event.ok ? 'done' : 'failed',
                    durationMs: event.duration_ms,
                    preview: event.preview,
                    resultCount: event.result_count,
                    error: event.error,
                  }
                  break
                }
              }
              return { ...entry, tools }
            }),
          }

        case 'audit':
          return {
            ...base,
            trace: patchLatest(base.trace, 'auditor', (entry) => ({
              ...entry,
              audit: {
                confidenceScore: event.confidence_score,
                isRelevant: event.is_relevant,
                reasoning: event.reasoning,
                action: event.action,
              },
            })),
          }

        case 'clarification':
          return { ...base, clarification: event.question }

        case 'token':
          return { ...base, answer: base.answer + event.text }

        case 'run_end':
          return { ...base, status: 'done', summary: event }

        case 'heartbeat':
          return base

        case 'error':
          return {
            ...base,
            status: 'error',
            errorMessage: event.message,
            trace: base.trace.map((entry) =>
              entry.status === 'running' ? { ...entry, status: 'done' } : entry,
            ),
          }

        default:
          return base
      }
    }

    default:
      return state
  }
}

export function useAgentStream() {
  const [state, dispatch] = useReducer(reducer, initialState)
  const abortRef = useRef<AbortController | null>(null)

  const cancel = useCallback(() => {
    abortRef.current?.abort()
    abortRef.current = null
  }, [])

  const send = useCallback(async (query: string, speed = 1) => {
    abortRef.current?.abort()
    const controller = new AbortController()
    abortRef.current = controller
    dispatch({ kind: 'reset', query })

    try {
      const response = await fetch(`/api/chat/stream?speed=${speed}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query }),
        signal: controller.signal,
      })

      if (!response.ok) {
        throw new Error(`Server returned ${response.status} ${response.statusText}`)
      }
      if (!response.body) {
        throw new Error('Response carried no readable body')
      }

      const reader = response.body.getReader()
      const decoder = new TextDecoder()
      const parse = createSSEParser()

      for (;;) {
        const { done, value } = await reader.read()
        if (done) break
        for (const frame of parse(decoder.decode(value, { stream: true }))) {
          try {
            dispatch({ kind: 'event', event: JSON.parse(frame.data) as AgentEvent })
          } catch {
            // A malformed frame should not tear down a run that is otherwise fine.
          }
        }
      }
      dispatch({ kind: 'closed' })
    } catch (error) {
      if (error instanceof DOMException && error.name === 'AbortError') {
        dispatch({ kind: 'closed' })
        return
      }
      const raw = error instanceof Error ? error.message : String(error)
      const message =
        /connection error|failed to fetch|networkerror|load failed/i.test(raw)
          ? 'The browser dropped the stream after a long wait. Hard-refresh and send the question again — the GPU is usually warm on the second try.'
          : raw
      dispatch({
        kind: 'error',
        message,
      })
    } finally {
      if (abortRef.current === controller) abortRef.current = null
    }
  }, [])

  return { state, send, cancel }
}
