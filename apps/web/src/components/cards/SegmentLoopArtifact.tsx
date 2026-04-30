import { useState } from "react";
import { acceptSegmentLoop, declineSegmentLoop, dismissSegmentLoop } from "../../lib/api";

interface SegmentLoopConfig {
  id: string;
  pieceId: string;
  barsStart: number;
  barsEnd: number;
  requiredCorrect: number;
  attemptsCompleted: number;
  status: "pending" | "active" | "completed" | "dismissed" | "superseded";
  dimension: string | null;
}

interface Props {
  config: SegmentLoopConfig;
}

export function SegmentLoopArtifactCard({ config }: Props) {
  const [status, setStatus] = useState(config.status);
  const [attempts] = useState(config.attemptsCompleted);

  if (status === "completed") {
    return (
      <div className="rounded-lg border p-4">
        <p className="font-medium text-green-700">
          Loop complete — bars {config.barsStart}–{config.barsEnd}
        </p>
      </div>
    );
  }

  if (status === "dismissed" || status === "superseded") {
    return (
      <div className="rounded-lg border p-4 opacity-50">
        <p className="text-sm text-gray-500">
          Bars {config.barsStart}–{config.barsEnd} — {status}
        </p>
      </div>
    );
  }

  return (
    <div className="rounded-lg border p-4 space-y-3">
      <div>
        <p className="font-medium">
          Practice loop: bars {config.barsStart}–{config.barsEnd}
        </p>
        {config.dimension && (
          <p className="text-sm text-gray-600">Focus: {config.dimension}</p>
        )}
      </div>

      {status === "active" && (
        <div className="flex items-center gap-2">
          <span className="text-sm">
            {attempts} / {config.requiredCorrect} attempts
          </span>
          <button
            type="button"
            onClick={async () => {
              await dismissSegmentLoop(config.id);
              setStatus("dismissed");
            }}
            className="ml-auto text-xs text-gray-500 underline"
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
              await acceptSegmentLoop(config.id);
              setStatus("active");
            }}
            className="rounded bg-indigo-600 px-3 py-1 text-sm text-white"
          >
            Accept
          </button>
          <button
            type="button"
            onClick={async () => {
              await declineSegmentLoop(config.id);
              setStatus("dismissed");
            }}
            className="rounded border px-3 py-1 text-sm"
          >
            Skip
          </button>
        </div>
      )}
    </div>
  );
}
