import { test, expect } from 'vitest'
import { ALL_MOLECULES } from './index'

test('ALL_MOLECULES contains 9 ToolDefinition objects with unique names', () => {
  expect(ALL_MOLECULES).toHaveLength(9)
  const names = ALL_MOLECULES.map(m => m.name)
  expect(new Set(names).size).toBe(9)
  for (const mol of ALL_MOLECULES) {
    expect(typeof mol.name).toBe('string')
    expect(mol.name.length).toBeGreaterThan(0)
    expect(typeof mol.description).toBe('string')
    expect(typeof mol.invoke).toBe('function')
    expect(typeof mol.input_schema).toBe('object')
  }
})

test('ALL_MOLECULES includes all 9 named molecules', () => {
  const names = new Set(ALL_MOLECULES.map(m => m.name))
  const expected = [
    'voicing-diagnosis', 'pedal-triage', 'rubato-coaching', 'phrasing-arc-analysis',
    'tempo-stability-triage', 'dynamic-range-audit', 'articulation-clarity-check',
    'cross-modal-contradiction-check', 'exercise-proposal',
  ]
  for (const name of expected) {
    expect(names.has(name), `missing molecule: ${name}`).toBe(true)
  }
})
