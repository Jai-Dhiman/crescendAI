import { test, expect } from 'vitest'
import { validateSkill } from '../validator'

test('atom: compute-velocity-curve conforms to spec', async () => {
  const r = await validateSkill('docs/harness/skills/atoms/compute-velocity-curve.md')
  expect(r.errors).toEqual([])
  expect(r.valid).toBe(true)
})
