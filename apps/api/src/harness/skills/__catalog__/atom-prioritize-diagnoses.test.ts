import { test, expect } from 'vitest'
import { validateSkill } from '../validator'

test('atom: prioritize-diagnoses conforms to spec', async () => {
  const r = await validateSkill('docs/harness/skills/atoms/prioritize-diagnoses.md')
  expect(r.errors).toEqual([])
  expect(r.valid).toBe(true)
})
