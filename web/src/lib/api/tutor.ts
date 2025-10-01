import { PUBLIC_API_URL } from '$env/static/public';

export type AnalysisData = {
  rhythm: number;
  pitch: number;
  dynamics: number;
  tempo: number;
  articulation: number;
  expression: number;
  technique: number;
  timing: number;
  phrasing: number;
  voicing: number;
  pedaling: number;
  hand_coordination: number;
  musical_understanding: number;
  stylistic_accuracy: number;
  creativity: number;
  listening: number;
  overall_performance: number;
  stage_presence: number;
  repertoire_difficulty: number;
};

export type RepertoireInfo = { composer?: string; piece?: string; difficulty?: number };
export type UserContext = {
  goals: string[];
  practice_time_per_day_minutes: number;
  constraints: string[];
  repertoire_info?: RepertoireInfo;
};

export type TutorRecommendation = {
  title: string;
  detail: string;
  applies_to: string[];
  practice_plan: string[];
  estimated_time_minutes: number;
  citations: string[];
};

export type TutorCitation = { id: string; title: string; source: string; url?: string; sections: string[] };
export type TutorFeedback = { recommendations: TutorRecommendation[]; citations: TutorCitation[] };

function apiKeyHeader() {
  // Read API key from localStorage to avoid hardcoding secrets in code
  const key = typeof localStorage !== 'undefined' ? localStorage.getItem('API_KEY') : null;
  return key ? { 'X-API-Key': key } : {};
}

const API_BASE = (typeof PUBLIC_API_URL === 'string' ? PUBLIC_API_URL : '').trim();

if (!API_BASE) {
  throw new Error('PUBLIC_API_URL is not defined. Set it in your .env files (e.g., .env.local or .env.production).');
}

export async function fetchAnalysisById(id: string): Promise<{ analysis: AnalysisData } | null> {
  const res = await fetch(`${API_BASE}/api/v1/result/${encodeURIComponent(id)}`, {
    headers: {
      'Content-Type': 'application/json',
      ...apiKeyHeader(),
    },
  });
  if (res.status === 404 || res.status === 400) return null;
  if (!res.ok) throw new Error(`Failed to fetch analysis: ${res.status}`);
  const json = await res.json();
  return json;
}

export async function getTutorFeedback(
  analysis: AnalysisData,
  userContext: UserContext,
  options?: { top_k?: number; model?: string }
): Promise<TutorFeedback> {
  const res = await fetch(`${API_BASE}/api/v1/tutor`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      ...apiKeyHeader(),
    },
    body: JSON.stringify({ analysis, user_context: userContext, options }),
  });
  if (!res.ok) throw new Error(`Tutor API error ${res.status}`);
  return await res.json();
}
