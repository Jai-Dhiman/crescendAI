import { test, expect } from 'vitest'
import { validateSkill } from '../validator'

test('molecule: pedal-triage conforms to spec', async () => {
  const r = await validateSkill('docs/harness/skills/molecules/pedal-triage.md')
  expect(r.errors).toEqual([])
  expect(r.valid).toBe(true)
})
