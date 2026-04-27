import { readFile, readdir } from 'node:fs/promises'
import { join } from 'node:path'
import { z } from 'zod'
import { ARTIFACT_NAMES } from '../artifacts'

export type ValidationResult = { valid: boolean; errors: string[] }
export type CatalogValidationResult = { valid: boolean; errors: string[] }

const TIERS = ['atom', 'molecule', 'compound'] as const
const REQUIRED_SECTIONS = [
  'When-to-fire',
  'When-NOT-to-fire',
  'Procedure',
  'Concrete example',
  'Post-conditions',
] as const

const ArtifactNameUnion = z.enum(ARTIFACT_NAMES as readonly [string, ...string[]])
const WritesField = z.union([ArtifactNameUnion, z.string().regex(/^scalar:/)])

const FrontmatterBase = z.object({
  name: z.string().min(1),
  tier: z.enum(TIERS),
  description: z.string().min(1),
  dimensions: z.array(z.string().min(1)).min(1),
  reads: z.object({
    signals: z.string().min(1),
    artifacts: z.array(ArtifactNameUnion),
  }),
  writes: WritesField,
  depends_on: z.array(z.string().min(1)),
})

const AtomFrontmatter = FrontmatterBase.extend({ tier: z.literal('atom') }).refine(
  (f) => f.depends_on.length === 0,
  { message: 'atoms must have empty depends_on', path: ['depends_on'] },
)

const MoleculeFrontmatter = FrontmatterBase.extend({ tier: z.literal('molecule') })

const CompoundFrontmatter = FrontmatterBase.extend({
  tier: z.literal('compound'),
  triggered_by: z.string().min(1),
})

function parseFrontmatter(source: string): { data: unknown; body: string } | null {
  const match = source.match(/^---\n([\s\S]*?)\n---\n([\s\S]*)$/)
  if (!match) return null
  const yamlText = match[1]
  const data = parseYaml(yamlText)
  return { data, body: match[2] }
}

function parseYaml(yamlText: string): unknown {
  const lines = yamlText.split('\n')
  const root: Record<string, unknown> = {}
  let i = 0
  while (i < lines.length) {
    const line = lines[i]
    if (line.trim() === '' || line.trim().startsWith('#')) { i++; continue }
    const m = line.match(/^([a-zA-Z_]+):\s*(.*)$/)
    if (!m) { i++; continue }
    const key = m[1]
    const rest = m[2]
    if (rest === '|') {
      const buf: string[] = []
      i++
      while (i < lines.length && (lines[i].startsWith('  ') || lines[i].trim() === '')) {
        buf.push(lines[i].replace(/^ {2}/, ''))
        i++
      }
      root[key] = buf.join('\n').trim()
      continue
    }
    if (rest === '') {
      // nested object
      const obj: Record<string, unknown> = {}
      i++
      while (i < lines.length && lines[i].startsWith('  ')) {
        const sub = lines[i].slice(2)
        const sm = sub.match(/^([a-zA-Z_]+):\s*(.*)$/)
        if (sm) obj[sm[1]] = parseScalar(sm[2])
        i++
      }
      root[key] = obj
      continue
    }
    root[key] = parseScalar(rest)
    i++
  }
  return root
}

function parseScalar(s: string): unknown {
  const t = s.trim()
  if (t.startsWith('[') && t.endsWith(']')) {
    const inner = t.slice(1, -1).trim()
    if (inner === '') return []
    return inner.split(',').map((x) => x.trim().replace(/^['"]|['"]$/g, ''))
  }
  if ((t.startsWith("'") && t.endsWith("'")) || (t.startsWith('"') && t.endsWith('"'))) {
    return t.slice(1, -1)
  }
  return t
}

export async function validateSkill(filePath: string): Promise<ValidationResult> {
  let source: string
  try {
    source = await readFile(filePath, 'utf8')
  } catch {
    return { valid: false, errors: [`file not found: ${filePath}`] }
  }
  const errors: string[] = []
  const parsed = parseFrontmatter(source)
  if (!parsed) {
    return { valid: false, errors: ['frontmatter not found or malformed'] }
  }
  const data = parsed.data as { tier?: string }
  const schema =
    data.tier === 'atom' ? AtomFrontmatter
      : data.tier === 'molecule' ? MoleculeFrontmatter
      : data.tier === 'compound' ? CompoundFrontmatter
      : null
  if (!schema) {
    errors.push(`tier must be one of [atom, molecule, compound]; got ${String(data.tier)}`)
  } else {
    const r = schema.safeParse(parsed.data)
    if (!r.success) {
      for (const issue of r.error.issues) {
        errors.push(`frontmatter: ${issue.path.join('.')}: ${issue.message}`)
      }
    }
  }
  for (const section of REQUIRED_SECTIONS) {
    const re = new RegExp(`^##\\s+${section.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')}\\b`, 'm')
    if (!re.test(parsed.body)) {
      errors.push(`missing required section: ${section}`)
    }
  }
  return { valid: errors.length === 0, errors }
}

export async function validateCatalog(rootDir: string): Promise<CatalogValidationResult> {
  const errors: string[] = []
  const tiers: Record<string, Set<string>> = { atom: new Set(), molecule: new Set(), compound: new Set() }

  async function readTier(tier: 'atom' | 'molecule' | 'compound', subdir: string) {
    const dir = join(rootDir, subdir)
    let entries: string[] = []
    try { entries = await readdir(dir) } catch { return }
    for (const entry of entries) {
      if (!entry.endsWith('.md') || entry === 'README.md') continue
      const filePath = join(dir, entry)
      const r = await validateSkill(filePath)
      if (!r.valid) {
        for (const e of r.errors) errors.push(`${filePath}: ${e}`)
      }
      tiers[tier].add(entry.replace(/\.md$/, ''))
    }
  }

  await readTier('atom', 'atoms')
  await readTier('molecule', 'molecules')
  await readTier('compound', 'compounds')

  // Cross-file: molecule depends_on must resolve to atom names; compound depends_on must resolve to molecule or atom names.
  for (const tier of ['molecule', 'compound'] as const) {
    const subdir = tier === 'molecule' ? 'molecules' : 'compounds'
    const allowed = tier === 'molecule' ? tiers.atom : new Set([...tiers.molecule, ...tiers.atom])
    let entries: string[] = []
    try { entries = await readdir(join(rootDir, subdir)) } catch { continue }
    for (const entry of entries) {
      if (!entry.endsWith('.md') || entry === 'README.md') continue
      const filePath = join(rootDir, subdir, entry)
      const source = await readFile(filePath, 'utf8').catch(() => '')
      const parsed = parseFrontmatter(source)
      if (!parsed) continue
      const dl = (parsed.data as { depends_on?: unknown }).depends_on
      if (!Array.isArray(dl)) continue
      for (const dep of dl as string[]) {
        if (!allowed.has(dep)) {
          errors.push(`${filePath}: depends_on entry "${dep}" does not resolve to an existing ${tier === 'molecule' ? 'atom' : 'molecule or atom'} file`)
        }
      }
    }
  }

  return { valid: errors.length === 0, errors }
}
