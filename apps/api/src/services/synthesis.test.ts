import { PgDialect } from 'drizzle-orm/pg-core'
import { describe, expect, it, vi } from 'vitest'
import { loadBaselinesFromDb, persistAccumulatedMoments, persistDiagnosisArtifacts } from './synthesis'
import type { AccumulatedMoment } from './accumulator'
import type { BarAnalysisFacts } from './bar-analysis-facts'

const VALID_RESULT = {
  primary_dimension: 'pedaling' as const,
  dimensions: ['pedaling'] as const,
  severity: 'moderate' as const,
  scope: 'passage' as const,
  bar_range: [3, 5] as [number, number],
  evidence_refs: ['chunk:0'],
  one_sentence_finding: 'Pedal overlap at bars 3-5 reduces note separation.',
  confidence: 'high' as const,
  finding_type: 'issue' as const,
}

const INVALID_RESULT = { not_a: 'diagnosis' }

describe('persistDiagnosisArtifacts', () => {
  it('inserts exactly 1 row when given 1 valid and 1 invalid result', async () => {
    const insertedValues: unknown[] = []
    const mockDb = {
      insert: vi.fn().mockReturnThis(),
      values: vi.fn().mockImplementation((v: unknown) => {
        insertedValues.push(v)
        return { onConflictDoNothing: vi.fn().mockResolvedValue(undefined) }
      }),
    }

    await persistDiagnosisArtifacts(
      mockDb as never,
      [
        { tool: 'compute-pedal-overlap-ratio', output: VALID_RESULT },
        { tool: 'extract-bar-range-signals', output: INVALID_RESULT },
      ],
      'sess-1',
      'stu-1',
      null,
    )

    expect(mockDb.insert).toHaveBeenCalledTimes(1)
    expect(insertedValues[0]).toMatchObject({
      sessionId: 'sess-1',
      studentId: 'stu-1',
      pieceId: null,
      primaryDimension: 'pedaling',
      barRangeStart: 3,
      barRangeEnd: 5,
    })
  })

  it('inserts 0 rows when all results are invalid', async () => {
    const mockDb = {
      insert: vi.fn().mockReturnThis(),
      values: vi.fn().mockReturnValue({ onConflictDoNothing: vi.fn().mockResolvedValue(undefined) }),
    }

    await persistDiagnosisArtifacts(mockDb as never, [{ tool: 'x', output: INVALID_RESULT }], 'sess-1', 'stu-1', null)

    expect(mockDb.insert).not.toHaveBeenCalled()
  })

  it('does not throw when DB insert rejects', async () => {
    const mockDb = {
      insert: vi.fn().mockReturnThis(),
      values: vi.fn().mockReturnValue({
        onConflictDoNothing: vi.fn().mockRejectedValue(new Error('DB error')),
      }),
    }

    await expect(
      persistDiagnosisArtifacts(mockDb as never, [{ tool: 'x', output: VALID_RESULT }], 's', 's', null)
    ).resolves.not.toThrow()
  })
})

describe('loadBaselinesFromDb date filter encoding', () => {
  // Regression: the 30-day cutoff was interpolated as a raw Date into a sql
  // template, bypassing the timestamp column encoder. postgres-js then received
  // a Date and threw "Received an instance of Date" on every chunk. The typed
  // gt() operator routes it through PgTimestamp.mapToDriverValue (-> ISO string).
  it('never passes a raw Date to the driver params', async () => {
    let capturedWhere: unknown = null
    const fakeDb = {
      select: () => fakeDb,
      from: () => fakeDb,
      where: (w: unknown) => {
        capturedWhere = w
        return fakeDb
      },
      groupBy: () => Promise.resolve([]),
    }

    const result = await loadBaselinesFromDb(fakeDb as never, 'stu-1')
    expect(result).toBeNull() // groupBy resolved to []
    expect(capturedWhere).not.toBeNull()

    const { params } = new PgDialect().sqlToQuery(capturedWhere as never)
    // No param may be a raw Date (that is exactly what broke serialization).
    for (const p of params) {
      expect(p instanceof Date).toBe(false)
    }
    // The cutoff must survive as an encoded ISO-8601 string.
    expect(
      params.some((p) => typeof p === 'string' && /^\d{4}-\d{2}-\d{2}T/.test(p)),
    ).toBe(true)
  })
})

describe('persistAccumulatedMoments reasoning_trace', () => {
  const baseMoment: AccumulatedMoment = {
    chunkIndex: 0,
    dimension: 'timing',
    score: 0.3,
    baseline: 0.5,
    deviation: -0.2,
    isPositive: false,
    reasoning: 'rushing ahead of beat',
    barRange: [4, 7],
    analysisTier: 1,
    timestampMs: 1,
    llmAnalysis: null,
  }

  function makeMockDb() {
    const insertedValues: Record<string, unknown>[] = []
    const mockDb = {
      insert: vi.fn().mockReturnThis(),
      values: vi.fn().mockImplementation((v: Record<string, unknown>) => {
        insertedValues.push(v)
        return { onConflictDoNothing: vi.fn().mockResolvedValue(undefined) }
      }),
    }
    return { mockDb, insertedValues }
  }

  it('falls back to moment.reasoning when llmAnalysis is null', async () => {
    const { mockDb, insertedValues } = makeMockDb()
    await persistAccumulatedMoments(mockDb as never, 'stu-1', 'sess-1', null, [baseMoment])
    expect(insertedValues[0]?.reasoningTrace).toBe('rushing ahead of beat')
  })

  it('writes discriminated-union JSON with kind="facts" when llmAnalysis is non-null', async () => {
    const facts: BarAnalysisFacts = {
      tier: 1,
      bar_range: '4-7',
      selected: { dimension: 'timing', analysis: 'rushing 45ms' },
      correlated: [],
    }
    const { mockDb, insertedValues } = makeMockDb()
    await persistAccumulatedMoments(mockDb as never, 'stu-1', 'sess-1', null, [
      { ...baseMoment, llmAnalysis: facts },
    ])
    const trace = insertedValues[0]?.reasoningTrace as string
    const parsed = JSON.parse(trace)
    // Discriminant must be present so eval harnesses can detect structured facts
    expect(parsed.kind).toBe('facts')
    // All BarAnalysisFacts fields must survive round-trip
    expect(parsed).toMatchObject(facts)
  })

  it('round-trip: kind="facts" JSON is not raw prose and contains all BarAnalysisFacts fields', async () => {
    const facts: BarAnalysisFacts = {
      tier: 2,
      bar_range: '1-3',
      selected: { dimension: 'dynamics', analysis: 'forte marking ignored' },
      correlated: [{ dimension: 'phrasing', analysis: 'phrase shape flat' }],
    }
    const { mockDb, insertedValues } = makeMockDb()
    await persistAccumulatedMoments(mockDb as never, 'stu-1', 'sess-1', 'conv-1', [
      { ...baseMoment, dimension: 'dynamics', llmAnalysis: facts },
    ])
    const trace = insertedValues[0]?.reasoningTrace as string
    const parsed = JSON.parse(trace)
    expect(parsed.kind).toBe('facts')
    expect(parsed.bar_range).toBe('1-3')
    expect(parsed.selected.dimension).toBe('dynamics')
    expect(parsed.correlated).toHaveLength(1)
    // Must not be raw prose string (would corrupt eval prompt)
    expect(typeof trace).toBe('string')
    expect(trace).not.toMatch(/^Selected dimension/)
  })
})
