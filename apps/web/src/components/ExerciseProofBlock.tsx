// apps/web/src/components/ExerciseProofBlock.tsx
import type { ProofCardManifest } from "../types/landing";
import { ProofCard } from "./ProofCard";

interface ExerciseProofBlockProps {
  manifests: [ProofCardManifest, ProofCardManifest, ProofCardManifest];
}

export function ExerciseProofBlock({ manifests }: ExerciseProofBlockProps) {
  return (
    <section aria-label="Exercise proof block" className="py-24 lg:py-32">
      <div className="max-w-5xl mx-auto px-6 lg:px-12">
        {/* Research callout */}
        <p className="text-body-md text-text-secondary text-center mb-12">
          Music AI that listens for expression, not just notes &mdash; trained on competitive performance data from international competitions.
          <sup aria-label="See footnote 1">1</sup>
        </p>

        {/* Three stacked ProofCards */}
        <div className="space-y-12">
          {manifests.map((manifest, i) => (
            <ProofCard key={manifest.pieceId} manifest={manifest} cardIndex={i} />
          ))}
        </div>
      </div>
    </section>
  );
}
