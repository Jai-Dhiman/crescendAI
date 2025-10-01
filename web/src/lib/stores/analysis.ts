import { writable } from 'svelte/store';
import type { AnalysisResult } from '$lib/types/analysis';

const STORAGE_KEY = 'analysis:last';

function loadInitial(): AnalysisResult | null {
  if (typeof localStorage === 'undefined') return null;
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    return raw ? (JSON.parse(raw) as AnalysisResult) : null;
  } catch {
    return null;
  }
}

export const analysisStore = writable<AnalysisResult | null>(loadInitial());

if (typeof localStorage !== 'undefined') {
  analysisStore.subscribe((val) => {
    try {
      if (val) {
        localStorage.setItem(STORAGE_KEY, JSON.stringify(val));
      } else {
        localStorage.removeItem(STORAGE_KEY);
      }
    } catch {
      // ignore persistence errors
    }
  });
}

