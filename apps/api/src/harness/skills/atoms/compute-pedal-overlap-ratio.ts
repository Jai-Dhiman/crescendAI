import type { ToolDefinition } from '../../loop/types'

export type CcEvent = { time_ms: number; value: number }

function pedaledDuration(
  noteOnset: number,
  noteDuration: number,
  pedalCc: CcEvent[],
): number {
  const noteEnd = noteOnset + noteDuration
  const sorted = [...pedalCc].sort((a, b) => a.time_ms - b.time_ms)
  let pedaled = 0
  let isPedaled = false
  let segStart = noteOnset
  for (const ev of sorted) {
    const t = Math.max(noteOnset, Math.min(noteEnd, ev.time_ms))
    if (isPedaled) {
      pedaled += Math.max(0, t - segStart)
    }
    isPedaled = ev.value >= 64
    segStart = Math.max(noteOnset, ev.time_ms)
  }
  // Close last segment if pedal still down at note end
  if (isPedaled && segStart < noteEnd) {
    pedaled += Math.max(0, noteEnd - segStart)
  }
  return pedaled
}

export const computePedalOverlapRatio: ToolDefinition = {
  name: 'compute-pedal-overlap-ratio',
  description:
    'Computes the time-weighted fraction of note duration during which the sustain pedal (CC64 >= 64) is depressed. Returns 0 when no notes are present.',
  input_schema: {
    type: 'object',
    properties: {
      notes: {
        type: 'array',
        items: {
          type: 'object',
          properties: {
            onset_ms: { type: 'number' },
            duration_ms: { type: 'number', minimum: 0 },
          },
          required: ['onset_ms', 'duration_ms'],
        },
      },
      pedal_cc: {
        type: 'array',
        items: {
          type: 'object',
          properties: {
            time_ms: { type: 'number' },
            value: { type: 'number', minimum: 0, maximum: 127 },
          },
          required: ['time_ms', 'value'],
        },
        description: 'Sustain pedal CC64 events. value >= 64 means depressed.',
      },
    },
    required: ['notes', 'pedal_cc'],
  },
  invoke: async (input: unknown): Promise<number> => {
    const { notes, pedal_cc } = input as {
      notes: { onset_ms: number; duration_ms: number }[]
      pedal_cc: CcEvent[]
    }
    const totalDuration = notes.reduce((s, n) => s + n.duration_ms, 0)
    if (totalDuration === 0) return 0
    const totalPedaled = notes.reduce(
      (s, n) => s + pedaledDuration(n.onset_ms, n.duration_ms, pedal_cc),
      0,
    )
    return totalPedaled / totalDuration
  },
}
