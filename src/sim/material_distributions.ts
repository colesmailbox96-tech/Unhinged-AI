import type { PropertyVector } from './properties';
import { clamp } from './properties';
import { RNG } from './rng';

export interface MaterialDistribution {
  debugFamily: string;
  mean: PropertyVector;
  stddev: Partial<Record<keyof PropertyVector, number>>;
}

const std = 0.05;

export const materialDistributions: MaterialDistribution[] = [
  {
    debugFamily: 'stone-like',
    mean: {
      mass: 0.86,
      density: 0.9,
      hardness: 0.85,
      sharpness: 0.25,
      brittleness: 0.72,
      elasticity: 0.2,
      roughness: 0.7,
      tensile_strength: 0.35,
      compressive_strength: 0.9,
      friction_coeff: 0.7,
      thermal_conductivity: 0.35,
      heat_capacity: 0.5,
      combustibility: 0.02,
      toxicity: 0.1,
    },
    stddev: { sharpness: 0.08, brittleness: 0.07, thermal_conductivity: 0.1 },
  },
  {
    debugFamily: 'wood-like',
    mean: {
      mass: 0.45,
      density: 0.4,
      hardness: 0.45,
      sharpness: 0.12,
      brittleness: 0.25,
      elasticity: 0.62,
      roughness: 0.6,
      tensile_strength: 0.62,
      compressive_strength: 0.5,
      friction_coeff: 0.64,
      thermal_conductivity: 0.2,
      heat_capacity: 0.66,
      combustibility: 0.7,
      toxicity: 0.05,
    },
    stddev: { elasticity: 0.08, tensile_strength: 0.08, combustibility: 0.09 },
  },
  {
    debugFamily: 'plant-like',
    mean: {
      mass: 0.26,
      density: 0.22,
      hardness: 0.2,
      sharpness: 0.1,
      brittleness: 0.18,
      elasticity: 0.75,
      roughness: 0.42,
      tensile_strength: 0.5,
      compressive_strength: 0.22,
      friction_coeff: 0.48,
      thermal_conductivity: 0.15,
      heat_capacity: 0.8,
      combustibility: 0.74,
      toxicity: 0.2,
    },
    stddev: { elasticity: 0.1, toxicity: 0.15, combustibility: 0.1 },
  },
  {
    debugFamily: 'metal-like',
    mean: {
      mass: 0.94,
      density: 0.95,
      hardness: 0.92,
      sharpness: 0.4,
      brittleness: 0.4,
      elasticity: 0.42,
      roughness: 0.3,
      tensile_strength: 0.82,
      compressive_strength: 0.88,
      friction_coeff: 0.38,
      thermal_conductivity: 0.82,
      heat_capacity: 0.3,
      combustibility: 0,
      toxicity: 0.18,
    },
    stddev: { sharpness: 0.12, roughness: 0.09, hardness: 0.06 },
  },
];

export function sampleProperties(distribution: MaterialDistribution, rng: RNG): PropertyVector {
  const out = {} as PropertyVector;
  for (const key of Object.keys(distribution.mean) as (keyof PropertyVector)[]) {
    const sigma = distribution.stddev[key] ?? std;
    out[key] = clamp(distribution.mean[key] + rng.normal(0, sigma));
  }
  return out;
}

export function fibrousTargetProps(rng: RNG): PropertyVector {
  const wood = sampleProperties(materialDistributions[1], rng);
  const plant = sampleProperties(materialDistributions[2], rng);
  return {
    ...wood,
    hardness: clamp((wood.hardness + plant.hardness) * 0.55 + 0.15),
    brittleness: clamp((wood.brittleness + plant.brittleness) * 0.4),
    tensile_strength: clamp((wood.tensile_strength + plant.tensile_strength) * 0.65 + 0.25),
    compressive_strength: clamp((wood.compressive_strength + plant.compressive_strength) * 0.6 + 0.1),
    elasticity: clamp((wood.elasticity + plant.elasticity) * 0.55 + 0.2),
  };
}
