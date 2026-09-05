/**
 * Mirror of `backend/app/events.py`. Keep the two in step.
 */

export type NodeName =
  | 'gatekeeper'
  | 'planner'
  | 'tool_executor'
  | 'auditor'
  | 'synthesizer'

export const NODE_ORDER: NodeName[] = [
  'gatekeeper',
  'planner',
  'tool_executor',
  'auditor',
  'synthesizer',
]

export const NODE_LABELS: Record<NodeName, string> = {
  gatekeeper: 'Gatekeeper',
  planner: 'Planner',
  tool_executor: 'Tool Executor',
  auditor: 'Auditor',
  synthesizer: 'Synthesizer',
}

export const TOOL_LABELS: Record<string, string> = {
  librarian_rag_tool: 'Librarian',
  analyst_sql_tool: 'SQL Analyst',
  analyst_trend_tool: 'Trend Analyst',
  scout_web_search_tool: 'Scout',
  FINISH: 'Finish',
}

interface BaseEvent {
  seq: number
  ts: number
}

export interface PlanStep {
  tool_name: string
  tool_input: string
}

export interface RunStartEvent extends BaseEvent {
  type: 'run_start'
  run_id: string
  conversation_id: string
  query: string
}

export interface NodeStartEvent extends BaseEvent {
  type: 'node_start'
  node: NodeName
  attempt: number
}

export interface NodeEndEvent extends BaseEvent {
  type: 'node_end'
  node: NodeName
  attempt: number
  duration_ms: number
}

export interface PlanEvent extends BaseEvent {
  type: 'plan'
  attempt: number
  steps: PlanStep[]
}

export interface ToolStartEvent extends BaseEvent {
  type: 'tool_start'
  tool_name: string
  tool_input: string
}

export interface ToolEndEvent extends BaseEvent {
  type: 'tool_end'
  tool_name: string
  duration_ms: number
  ok: boolean
  preview: string | null
  result_count: number | null
  error: string | null
}

export interface AuditEvent extends BaseEvent {
  type: 'audit'
  attempt: number
  confidence_score: number
  is_relevant: boolean
  reasoning: string
  action: 'accept' | 'replan'
}

export interface ClarificationEvent extends BaseEvent {
  type: 'clarification'
  question: string
}

export interface TokenEvent extends BaseEvent {
  type: 'token'
  text: string
}

export interface RunEndEvent extends BaseEvent {
  type: 'run_end'
  run_id: string
  duration_ms: number
  node_timings_ms: Record<string, number>
  llm_calls: number
  output_tokens: number
  replans: number
}

export interface ErrorEvent extends BaseEvent {
  type: 'error'
  code: string
  message: string
  retryable: boolean
}

export interface HeartbeatEvent extends BaseEvent {
  type: 'heartbeat'
}

export type AgentEvent =
  | RunStartEvent
  | NodeStartEvent
  | NodeEndEvent
  | PlanEvent
  | ToolStartEvent
  | ToolEndEvent
  | AuditEvent
  | ClarificationEvent
  | TokenEvent
  | RunEndEvent
  | ErrorEvent
  | HeartbeatEvent
