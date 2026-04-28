import type { ToolDefinition } from '../../loop/types'
import type { CcEvent } from './compute-pedal-overlap-ratio'
import type { Alignment } from './align-performance-to-score'

export type MidiNote = { pitch: number; onset_ms: number; duration_ms: number; velocity: number; bar?: number }
export type SignalBundle = {
  muq_scores: number[][]
  midi_notes: MidiNote[]
  pedal_cc: CcEvent[]
  alignment: Alignment[]
}

function overlaps(chunkCoverage: [number, number], barRange: [number, number]): boolean {
  return chunkCoverage[0] <= barRange[1] && chunkCoverage[1] >= barRange[0]
}

export const extractBarRangeSignals: ToolDefinition = {
  name: 'extract-bar-range-signals',
  description:
    'Slices the enrichment cache to return all signals overlapping a bar range. Returns a SignalBundle with muq_scores (one 6-vector per overlapping chunk), midi_notes filtered to bar_range, pedal_cc from overlapping chunks, and alignment entries filtered to bar_range.',
  input_schema: {
    type: 'object',
    properties: {
      bar_range: {
        type: 'array',
        items: { type: 'integer' },
        minItems: 2,
        maxItems: 2,
      },
      chunks: {
        type: 'array',
        items: {
          type: 'object',
          properties: {
            chunk_id: { type: 'string' },
            bar_coverage: { type: 'array', items: { type: 'integer' }, minItems: 2, maxItems: 2 },
            muq_scores: { type: 'array', items: { type: 'number' }, minItems: 6, maxItems: 6 },
            midi_notes: { type: 'array' },
            pedal_cc: { type: 'array' },
            alignment: { type: 'array' },
          },
          required: ['chunk_id', 'bar_coverage', 'muq_scores', 'midi_notes', 'pedal_cc', 'alignment'],
        },
      },
    },
    required: ['bar_range', 'chunks'],
  },
  invoke: async (input: unknown): Promise<SignalBundle> => {
    const { bar_range, chunks } = input as {
      bar_range: [number, number]
      chunks: {
        chunk_id: string
        bar_coverage: [number, number]
        muq_scores: number[]
        midi_notes: MidiNote[]
        pedal_cc: CcEvent[]
        alignment: Alignment[]
      }[]
    }
    const relevant = chunks.filter((c) => overlaps(c.bar_coverage, bar_range))
    const seenChunks = new Set<string>()
    const muq_scores: number[][] = []
    const midi_notes: MidiNote[] = []
    const pedal_cc: CcEvent[] = []
    const alignment: Alignment[] = []

    for (const chunk of relevant) {
      if (!seenChunks.has(chunk.chunk_id)) {
        seenChunks.add(chunk.chunk_id)
        muq_scores.push(chunk.muq_scores)
        pedal_cc.push(...chunk.pedal_cc)
      }
      midi_notes.push(...chunk.midi_notes.filter((n) => n.bar === undefined || (n.bar >= bar_range[0] && n.bar <= bar_range[1])))
      alignment.push(...chunk.alignment.filter((a) => a.bar >= bar_range[0] && a.bar <= bar_range[1]))
    }
    return { muq_scores, midi_notes, pedal_cc, alignment }
  },
}
