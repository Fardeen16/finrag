/**
 * Incremental SSE frame parser.
 *
 * Needed because we read the stream with `fetch` + `ReadableStream` instead of
 * `EventSource`. `EventSource` would parse frames for us, but it only issues GET
 * requests and cannot set headers — so it can carry neither the query body nor
 * the Authorization header the API will require. Parsing by hand is the price.
 *
 * Network chunks do not align with frame boundaries, so state has to persist
 * across calls to `push`.
 */

export interface SSEFrame {
  event?: string
  data: string
}

export function createSSEParser() {
  let buffer = ''

  return function push(chunk: string): SSEFrame[] {
    buffer += chunk

    // A trailing CR may be the first half of a CRLF split across two chunks.
    // Normalising it now would fabricate a frame boundary, so hold it back.
    let held = ''
    let work = buffer
    if (work.endsWith('\r')) {
      held = '\r'
      work = work.slice(0, -1)
    }
    work = work.replace(/\r\n/g, '\n').replace(/\r/g, '\n')

    const parts = work.split('\n\n')
    buffer = (parts.pop() ?? '') + held

    const frames: SSEFrame[] = []
    for (const part of parts) {
      if (!part.trim()) continue
      let event: string | undefined
      const data: string[] = []
      for (const line of part.split('\n')) {
        if (!line || line.startsWith(':')) continue // comment / heartbeat
        const colon = line.indexOf(':')
        const field = colon === -1 ? line : line.slice(0, colon)
        let value = colon === -1 ? '' : line.slice(colon + 1)
        if (value.startsWith(' ')) value = value.slice(1)
        if (field === 'event') event = value
        else if (field === 'data') data.push(value)
      }
      if (data.length) frames.push({ event, data: data.join('\n') })
    }
    return frames
  }
}
