import { test, expect } from 'vitest'
import { validateSkill } from '../validator'

test('atom: align-performance-to-score conforms to spec', async () => {
  const r = await validateSkill('docs/harness/skills/atoms/align-performance-to-score.md')
  expect(r.errors).toEqual([])
  expect(r.valid).toBe(true)
})
