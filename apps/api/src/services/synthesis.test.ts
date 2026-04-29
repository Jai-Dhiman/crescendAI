import { describe, expect, it, vi } from 'vitest'
import { persistDiagnosisArtifacts } from './synthesis'

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
