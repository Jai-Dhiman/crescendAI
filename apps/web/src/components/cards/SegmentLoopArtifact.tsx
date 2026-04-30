import { useState } from "react";
import { acceptSegmentLoop, declineSegmentLoop, dismissSegmentLoop } from "../../lib/api";
import type { SegmentLoopConfig } from "../../lib/types";

interface Props {
  config: SegmentLoopConfig;
}

export function SegmentLoopArtifactCard({ config }: Props) {
  const [status, setStatus] = useState(config.status);
  const [attempts] = useState(config.attemptsCompleted);
  const [error, setError] = useState<string | null>(null);

  if (status === "completed") {
    return (
      <div className="bg-surface-card border border-border rounded-xl p-4">
        <p className="font-medium text-accent">
          Loop complete — bars {config.barsStart}–{config.barsEnd}
        </p>
      </div>
    );
  }

  if (status === "dismissed" || status === "superseded") {
    return (
      <div className="bg-surface-card border border-border rounded-xl p-4 opacity-50">
        <p className="text-body-sm text-text-tertiary">
          Bars {config.barsStart}–{config.barsEnd} — {status}
        </p>
      </div>
    );
  }

  return (
    <div className="bg-surface-card border border-border rounded-xl p-4 space-y-3">
      <div>
        <p className="font-medium text-text-primary">
          Practice loop: bars {config.barsStart}–{config.barsEnd}
        </p>
        {config.dimension && (
          <p className="text-body-sm text-text-secondary">Focus: {config.dimension}</p>
        )}
      </div>

      {error && (
        <p className="text-body-xs text-red-400">{error}</p>
      )}

      {status === "active" && (
        <div className="flex items-center gap-2">
          <span className="text-body-sm text-text-primary">
            {attempts} / {config.requiredCorrect} attempts
          </span>
          <button
            type="button"
            onClick={async () => {
              setError(null);
              try {
                await dismissSegmentLoop(config.id);
                setStatus("dismissed");
              } catch (e) {
                setError(e instanceof Error ? e.message : "Failed to dismiss loop");
              }
            }}
            className="ml-auto text-body-xs text-text-secondary underline"
          >
            Dismiss
          </button>
        </div>
      )}

      {status === "pending" && (
        <div className="flex gap-2">
          <button
            type="button"
            onClick={async () => {
              setError(null);
              try {
                await acceptSegmentLoop(config.id);
                setStatus("active");
              } catch (e) {
                setError(e instanceof Error ? e.message : "Failed to accept loop");
              }
            }}
            className="rounded-lg bg-accent px-3 py-1 text-body-sm text-cream"
          >
            Accept
          </button>
          <button
            type="button"
            onClick={async () => {
              setError(null);
              try {
                await declineSegmentLoop(config.id);
                setStatus("dismissed");
              } catch (e) {
                setError(e instanceof Error ? e.message : "Failed to decline loop");
              }
            }}
            className="rounded-lg border border-border px-3 py-1 text-body-sm text-text-secondary"
          >
            Skip
          </button>
        </div>
      )}
    </div>
  );
}
