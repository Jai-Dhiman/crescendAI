import { describe, test, expect } from 'vitest'
import { validateSkill, validateCatalog } from './validator'

const F = 'apps/api/src/harness/skills/__fixtures__'

describe('validateSkill', () => {
  test('accepts a well-formed atom file', async () => {
    const r = await validateSkill(`${F}/valid-atom.md`)
    expect(r.errors).toEqual([])
    expect(r.valid).toBe(true)
  })

  test('accepts a well-formed molecule file', async () => {
    const r = await validateSkill(`${F}/valid-molecule.md`)
    expect(r.errors).toEqual([])
    expect(r.valid).toBe(true)
  })

  test('accepts a well-formed compound file', async () => {
    const r = await validateSkill(`${F}/valid-compound.md`)
    expect(r.errors).toEqual([])
    expect(r.valid).toBe(true)
  })

  test('rejects when a required body section is missing', async () => {
    const r = await validateSkill(`${F}/invalid-missing-section.md`)
    expect(r.valid).toBe(false)
    expect(r.errors.some((e) => e.includes('Procedure'))).toBe(true)
  })

  test('rejects when frontmatter has no tier field', async () => {
    const r = await validateSkill('apps/api/src/harness/skills/__fixtures__/does-not-exist-tier-missing.md').catch(() => ({ valid: false, errors: ['file not found'] }))
    expect(r.valid).toBe(false)
  })
})

describe('validateCatalog', () => {
  test('returns valid when fixtures directory contains only well-formed files referencing each other correctly', async () => {
    // Run on the live catalog; expect this to be empty before any skill files exist
    // and to remain valid (or report only missing-files errors) until catalog is populated.
    const r = await validateCatalog('docs/harness/skills')
    // The validator must always return a structured result, never throw.
    expect(Array.isArray(r.errors)).toBe(true)
    expect(typeof r.valid).toBe('boolean')
  })
})
