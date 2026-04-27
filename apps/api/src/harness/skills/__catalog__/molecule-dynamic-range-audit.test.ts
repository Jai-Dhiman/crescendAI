import { test, expect } from 'vitest'
import { validateSkill } from '../validator'

test('molecule: dynamic-range-audit conforms to spec', async () => {
  const r = await validateSkill('docs/harness/skills/molecules/dynamic-range-audit.md')
  expect(r.errors).toEqual([])
  expect(r.valid).toBe(true)
})
