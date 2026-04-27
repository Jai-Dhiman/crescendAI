import { test, expect } from 'vitest'
import { validateSkill } from '../validator'

test('atom: fetch-session-history conforms to spec', async () => {
  const r = await validateSkill('docs/harness/skills/atoms/fetch-session-history.md')
  expect(r.errors).toEqual([])
  expect(r.valid).toBe(true)
})
