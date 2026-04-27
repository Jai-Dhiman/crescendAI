import type { ZodTypeAny } from 'zod'
import { DiagnosisArtifactSchema, type DiagnosisArtifact } from './diagnosis'
import { ExerciseArtifactSchema, type ExerciseArtifact } from './exercise'
import { SynthesisArtifactSchema, type SynthesisArtifact } from './synthesis'

export { DiagnosisArtifactSchema, type DiagnosisArtifact } from './diagnosis'
export { ExerciseArtifactSchema, type ExerciseArtifact } from './exercise'
export { SynthesisArtifactSchema, type SynthesisArtifact } from './synthesis'

export const ARTIFACT_NAMES = ['DiagnosisArtifact', 'ExerciseArtifact', 'SynthesisArtifact'] as const
export type ArtifactName = (typeof ARTIFACT_NAMES)[number]

export const artifactSchemas: Record<ArtifactName, ZodTypeAny> = {
  DiagnosisArtifact: DiagnosisArtifactSchema,
  ExerciseArtifact: ExerciseArtifactSchema,
  SynthesisArtifact: SynthesisArtifactSchema,
}
