export const DIMS_6 = [
  "dynamics",
  "timing",
  "pedaling",
  "articulation",
  "phrasing",
  "interpretation",
] as const;

export type Dimension = (typeof DIMS_6)[number];

export const DIM_INDEX: Record<Dimension, number> = {
  dynamics: 0,
  timing: 1,
  pedaling: 2,
  articulation: 3,
  phrasing: 4,
  interpretation: 5,
};
