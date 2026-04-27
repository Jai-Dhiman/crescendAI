import { describe, test, expect } from 'vitest'
import { DiagnosisArtifactSchema, type DiagnosisArtifact } from './diagnosis'

const baseValid: DiagnosisArtifact = {
  primary_dimension: 'pedaling',
  dimensions: ['pedaling'],
  severity: 'moderate',
  scope: 'stop_moment',
  bar_range: [12, 16],
  evidence_refs: ['cache:muq:abc123'],
  one_sentence_finding: 'Over-pedaled through the slow passage at bars 12-16.',
  confidence: 'high',
  finding_type: 'issue',
}

describe('DiagnosisArtifactSchema', () => {
  test('accepts a fully valid baseline artifact', () => {
    expect(() => DiagnosisArtifactSchema.parse(baseValid)).not.toThrow()
  })

  test('rejects when primary_dimension is not in dimensions list', () => {
    const invalid = { ...baseValid, primary_dimension: 'timing' as const, dimensions: ['pedaling'] as const }
    const result = DiagnosisArtifactSchema.safeParse(invalid)
    expect(result.success).toBe(false)
    expect(result.error?.issues.some(i => i.message.includes('primary_dimension'))).toBe(true)
  })

  test('rejects when one_sentence_finding exceeds 200 chars', () => {
    const invalid = { ...baseValid, one_sentence_finding: 'x'.repeat(201) }
    expect(DiagnosisArtifactSchema.safeParse(invalid).success).toBe(false)
  })

  test('rejects when evidence_refs is empty', () => {
    const invalid = { ...baseValid, evidence_refs: [] as string[] }
    expect(DiagnosisArtifactSchema.safeParse(invalid).success).toBe(false)
  })

  test('rejects when bar_range is null but scope is not "session"', () => {
    const invalid = { ...baseValid, bar_range: null, scope: 'stop_moment' as const }
    const result = DiagnosisArtifactSchema.safeParse(invalid)
    expect(result.success).toBe(false)
    expect(result.error?.issues.some(i => i.message.includes('bar_range'))).toBe(true)
  })

  test('accepts bar_range null when scope is "session"', () => {
    const valid = { ...baseValid, bar_range: null, scope: 'session' as const }
    expect(() => DiagnosisArtifactSchema.parse(valid)).not.toThrow()
  })

  test('rejects when bar_range start > end', () => {
    const invalid = { ...baseValid, bar_range: [16, 12] as [number, number] }
    expect(DiagnosisArtifactSchema.safeParse(invalid).success).toBe(false)
  })

  test('accepts strength finding_type with minor severity', () => {
    const valid = { ...baseValid, finding_type: 'strength' as const, severity: 'minor' as const }
    expect(() => DiagnosisArtifactSchema.parse(valid)).not.toThrow()
  })
})
