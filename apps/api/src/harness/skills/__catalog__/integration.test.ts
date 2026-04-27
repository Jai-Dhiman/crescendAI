import { test, expect } from 'vitest'
import { validateCatalog } from '../validator'

test('full catalog: validateCatalog returns no errors', async () => {
  const r = await validateCatalog('docs/harness/skills')
  expect(r.errors).toEqual([])
  expect(r.valid).toBe(true)
})
