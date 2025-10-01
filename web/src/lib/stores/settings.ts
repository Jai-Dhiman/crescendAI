import { writable } from 'svelte/store';

// Whether to request real GPU inference on the next analysis/compare call
// Defaults to false for portfolio demos to control costs.
export const runOnGPU = writable<boolean>(false);

// Persist to localStorage
runOnGPU.subscribe((val) => {
  try {
    localStorage.setItem('crescendai.runOnGPU', JSON.stringify(val));
  } catch {}
});

export function loadRunOnGPUFromStorage() {
  try {
    const v = localStorage.getItem('crescendai.runOnGPU');
    if (v != null) runOnGPU.set(JSON.parse(v));
  } catch {}
}
