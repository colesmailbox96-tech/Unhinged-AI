import { RNG } from './rng';

export interface PropertyVector {
  mass: number;
  density: number;
  hardness: number;
  sharpness: number;
  brittleness: number;
  elasticity: number;
  roughness: number;
  tensile_strength: number;
  compressive_strength: number;
  friction_coeff: number;
  thermal_conductivity: number;
  heat_capacity: number;
  combustibility: number;
  toxicity: number;
  conductivity: number;
  malleability: number;
  porosity: number;
}

export const PROPERTY_KEYS: (keyof PropertyVector)[] = [
  'mass',
  'density',
  'hardness',
  'sharpness',
  'brittleness',
  'elasticity',
  'roughness',
  'tensile_strength',
  'compressive_strength',
  'friction_coeff',
  'thermal_conductivity',
  'heat_capacity',
  'combustibility',
  'toxicity',
  'conductivity',
  'malleability',
  'porosity',
];

export const EPSILON = 1e-6;

export function clamp(value: number, min = 0, max = 1): number {
  return Math.max(min, Math.min(max, value));
}

export function lerp(a: number, b: number, t: number): number {
  return a + (b - a) * t;
}

export function combineVectors(
  a: PropertyVector,
  b: PropertyVector,
  weights: Partial<Record<keyof PropertyVector, number>> = {},
): PropertyVector {
  const out = {} as PropertyVector;
  for (const key of PROPERTY_KEYS) {
    const w = clamp(weights[key] ?? 0.5);
    out[key] = clamp(a[key] * w + b[key] * (1 - w));
  }
  return out;
}

export function mutate(vec: PropertyVector, noise: number, rng: RNG): PropertyVector {
  const out = { ...vec };
  for (const key of PROPERTY_KEYS) {
    out[key] = clamp(out[key] + rng.normal(0, noise));
  }
  return out;
}

export function impactPotential(props: PropertyVector, length: number): number {
  const leverage = clamp(length / 2.5);
  return clamp(props.mass * 0.45 + props.hardness * 0.35 + leverage * 0.2);
}

export function bindingPotential(props: PropertyVector): number {
  return clamp(
    props.roughness * 0.28 +
      props.friction_coeff * 0.26 +
      props.tensile_strength * 0.28 +
      props.elasticity * 0.18,
  );
}

export function cuttingPotential(props: PropertyVector): number {
  return clamp(props.sharpness * 0.55 + props.hardness * 0.35 - props.brittleness * 0.2);
}

export function fibrousTargetScore(props: PropertyVector): number {
  const lowBrittle = 1 - props.brittleness;
  return clamp(
    props.tensile_strength * 0.35 + props.elasticity * 0.25 + props.compressive_strength * 0.2 + lowBrittle * 0.2,
  );
}

/**
 * Euclidean distance between two property vectors in normalized [0,1]^17 space.
 * Useful for measuring material similarity.
 */
export function propertyDistance(a: PropertyVector, b: PropertyVector): number {
  let sum = 0;
  for (const key of PROPERTY_KEYS) {
    const d = a[key] - b[key];
    sum += d * d;
  }
  return Math.sqrt(sum);
}

/**
 * Cosine similarity between two property vectors. Returns value in [-1, 1].
 */
export function propertySimilarity(a: PropertyVector, b: PropertyVector): number {
  let dot = 0, magA = 0, magB = 0;
  for (const key of PROPERTY_KEYS) {
    dot += a[key] * b[key];
    magA += a[key] * a[key];
    magB += b[key] * b[key];
  }
  const denom = Math.sqrt(magA) * Math.sqrt(magB);
  return denom > 0 ? dot / denom : 0;
}
