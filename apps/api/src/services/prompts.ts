export const CHAT_SYSTEM = `You are a warm, encouraging piano teacher. You help students improve their playing through thoughtful conversation. You give specific, actionable advice grounded in the student's actual playing data when available.

Key principles:
- Celebrate strengths before suggesting improvements
- Frame observations, not absolute judgments
- Give actionable practice strategies
- Be specific about musical elements (dynamics, timing, pedaling, articulation, phrasing, interpretation)
- Adapt to the student's level and goals`;

export function buildChatUserContext(student: {
  inferredLevel?: string | null;
  explicitGoals?: string | null;
  baselines?: Record<string, number | null>;
}): string {
  const parts: string[] = [];
  if (student.inferredLevel) {
    parts.push(`Student level: ${student.inferredLevel}`);
  }
  if (student.explicitGoals) {
    parts.push(`Student goals: ${student.explicitGoals}`);
  }
  if (student.baselines) {
    const dims = Object.entries(student.baselines)
      .filter(([, v]) => v != null)
      .map(([k, v]) => `${k}: ${(v as number).toFixed(2)}`)
      .join(", ");
    if (dims) parts.push(`Current baselines: ${dims}`);
  }
  return parts.join("\n");
}

export function buildTitlePrompt(firstMessage: string): string {
  return `Generate a concise title (3-6 words, no quotes) for a piano lesson conversation that starts with: "${firstMessage}"`;
}
