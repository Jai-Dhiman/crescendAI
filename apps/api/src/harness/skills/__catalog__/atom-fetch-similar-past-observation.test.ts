import { test, expect } from 'vitest'
import { validateSkill } from '../validator'

test('atom: fetch-similar-past-observation conforms to spec', async () => {
  const r = await validateSkill('docs/harness/skills/atoms/fetch-similar-past-observation.md')
  expect(r.errors).toEqual([])
  expect(r.valid).toBe(true)
})
