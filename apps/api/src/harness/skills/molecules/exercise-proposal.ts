import type { ToolDefinition } from '../../loop/types'
import { ExerciseArtifactSchema } from '../../artifacts/exercise'
import type { ExerciseArtifact } from '../../artifacts/exercise'
import type { DiagnosisArtifact } from '../../artifacts/diagnosis'
import { fetchSimilarPastObservation } from '../atoms/fetch-similar-past-observation'
import type { PastObservation } from '../atoms/fetch-similar-past-observation'

type ExerciseInput = {
  diagnosis: DiagnosisArtifact
  diagnosis_ref: string
  midi_notes: { pitch: number; onset_ms: number; duration_ms: number; velocity: number; bar: number }[]
  past_diagnoses: { artifact_id: string; session_id: string; created_at: number; primary_dimension: string; bar_range: [number, number] | null; piece_id: string }[]
  piece_id: string
  now_ms: number
}

const EXERCISE_MAP: Record<string, Record<string, { type: string; subtype: string | null }>> = {
  pedaling: {
    minor:       { type: 'pedal_isolation', subtype: null },
    moderate:    { type: 'pedal_isolation', subtype: null },
    significant: { type: 'pedal_isolation', subtype: 'no-pedal-pass' },
  },
  timing: {
    minor:       { type: 'segment_loop', subtype: 'metronome-locked' },
    moderate:    { type: 'segment_loop', subtype: 'metronome-locked' },
    significant: { type: 'segment_loop', subtype: 'metronome-locked' },
  },
  dynamics: {
    minor:       { type: 'dynamic_exaggeration', subtype: null },
    moderate:    { type: 'dynamic_exaggeration', subtype: null },
    significant: { type: 'dynamic_exaggeration', subtype: 'extreme-contrast' },
  },
  articulation: {
    minor:       { type: 'isolated_hands', subtype: null },
    moderate:    { type: 'isolated_hands', subtype: null },
    significant: { type: 'isolated_hands', subtype: null },
  },
  phrasing: {
    minor:       { type: 'slow_practice', subtype: 'shape-vocally' },
    moderate:    { type: 'slow_practice', subtype: 'shape-vocally' },
    significant: { type: 'slow_practice', subtype: 'shape-vocally' },
  },
  interpretation: {
    minor:       { type: 'slow_practice', subtype: 'imitate-reference' },
    moderate:    { type: 'slow_practice', subtype: 'imitate-reference' },
    significant: { type: 'slow_practice', subtype: 'imitate-reference' },
  },
}

const MINUTES_MAP: Record<string, number> = { minor: 2, moderate: 5, significant: 8 }

const INSTRUCTION_MAP: Record<string, string> = {
  pedal_isolation: `Play bars {start}-{end} three times with no sustain pedal. Listen for whether the line still sustains itself in your fingers.`,
  segment_loop: `Loop bars {start}-{end} with a metronome. Match the click exactly. Do not accelerate or slow down.`,
  dynamic_exaggeration: `Play bars {start}-{end} with exaggerated dynamics. Make the loud parts louder and the soft parts softer than you think is right.`,
  isolated_hands: `Play bars {start}-{end} hands separately. Focus on even articulation between fingers.`,
  slow_practice: `Play bars {start}-{end} at half tempo. Listen to the shape of the phrase as you play.`,
}

const SUCCESS_MAP: Record<string, string> = {
  pedal_isolation: 'Three consecutive clean repetitions with no pedal where harmonies remain audibly distinct.',
  segment_loop: 'Five consecutive repetitions matching the metronome within 20ms.',
  dynamic_exaggeration: 'The loudest and softest moments are clearly audible as different from each other.',
  isolated_hands: 'Each hand plays with consistent articulation for three repetitions.',
  slow_practice: 'The phrase shape is clear and intentional at half tempo.',
}

const ACTION_BINDING_TOOLS: Record<string, string> = {
  pedal_isolation: 'mute_pedal',
  segment_loop: 'segment_loop',
  isolated_hands: 'isolated_hands',
}

export const exerciseProposal: ToolDefinition = {
  name: 'exercise-proposal',
  description: 'Generates one targeted ExerciseArtifact from a single DiagnosisArtifact. Requires finding_type="issue" and severity in {moderate, significant}.',
  input_schema: {
    type: 'object',
    properties: {
      diagnosis: { type: 'object' },
      diagnosis_ref: { type: 'string' },
      midi_notes: { type: 'array', items: { type: 'object' } },
      past_diagnoses: { type: 'array', items: { type: 'object' } },
      piece_id: { type: 'string' },
      now_ms: { type: 'number' },
    },
    required: ['diagnosis', 'diagnosis_ref', 'midi_notes', 'past_diagnoses', 'piece_id', 'now_ms'],
  },
  invoke: async (input: unknown): Promise<ExerciseArtifact> => {
    const i = input as ExerciseInput
    const d = i.diagnosis

    if (d.finding_type !== 'issue') {
      throw new Error(`exercise-proposal: diagnosis must have finding_type "issue", got "${d.finding_type}"`)
    }
    if (d.bar_range === null) {
      throw new Error('exercise-proposal: diagnosis bar_range must not be null')
    }
    if (d.severity === 'minor') {
      throw new Error(`exercise-proposal: severity must be "moderate" or "significant", got "minor"`)
    }

    const prior = await fetchSimilarPastObservation.invoke({
      dimension: d.primary_dimension,
      piece_id: i.piece_id,
      bar_range: d.bar_range,
      past_diagnoses: i.past_diagnoses,
      now_ms: i.now_ms,
    }) as PastObservation | null

    if (prior && prior.days_ago < 3) {
      throw new Error(`exercise-proposal: diagnosis already addressed by exercise ${prior.artifact_id} ${prior.days_ago} day(s) ago`)
    }

    const mapping = EXERCISE_MAP[d.primary_dimension]?.[d.severity]
    if (!mapping) throw new Error(`exercise-proposal: no mapping for dimension "${d.primary_dimension}" severity "${d.severity}"`)
    const estimatedMinutes = MINUTES_MAP[d.severity] ?? 5

    const actionTool = ACTION_BINDING_TOOLS[mapping.type]
    const action_binding = actionTool
      ? { tool: actionTool, args: { bars: d.bar_range } }
      : null

    const instructionTemplate = INSTRUCTION_MAP[mapping.type] ?? INSTRUCTION_MAP.slow_practice
    const instruction = instructionTemplate
      .replace('{start}', String(d.bar_range[0]))
      .replace('{end}', String(d.bar_range[1]))
      .slice(0, 400)

    const success_criterion = (SUCCESS_MAP[mapping.type] ?? SUCCESS_MAP.slow_practice).slice(0, 200)

    return ExerciseArtifactSchema.parse({
      diagnosis_ref: i.diagnosis_ref,
      diagnosis_summary: d.one_sentence_finding,
      target_dimension: d.primary_dimension,
      exercise_type: mapping.type,
      exercise_subtype: mapping.subtype,
      bar_range: d.bar_range,
      instruction,
      success_criterion,
      estimated_minutes: estimatedMinutes,
      action_binding,
    })
  },
}
