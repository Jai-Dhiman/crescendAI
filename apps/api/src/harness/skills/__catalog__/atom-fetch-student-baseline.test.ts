import { test, expect } from 'vitest'
import { validateSkill } from '../validator'

test('atom: fetch-student-baseline conforms to spec', async () => {
  const r = await validateSkill('docs/harness/skills/atoms/fetch-student-baseline.md')
  expect(r.errors).toEqual([])
  expect(r.valid).toBe(true)
})
