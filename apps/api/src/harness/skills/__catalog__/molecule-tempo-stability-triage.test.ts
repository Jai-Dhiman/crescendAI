import { test, expect } from 'vitest'
import { validateSkill } from '../validator'

test('molecule: tempo-stability-triage conforms to spec', async () => {
  const r = await validateSkill('docs/harness/skills/molecules/tempo-stability-triage.md')
  expect(r.errors).toEqual([])
  expect(r.valid).toBe(true)
})
