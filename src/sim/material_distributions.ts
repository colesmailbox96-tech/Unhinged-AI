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
      conductivity: 0.15,
      malleability: 0.1,
      porosity: 0.12,
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
      conductivity: 0.18,
      malleability: 0.35,
      porosity: 0.45,
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
      conductivity: 0.12,
      malleability: 0.5,
      porosity: 0.55,
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
      conductivity: 0.8,
      malleability: 0.7,
      porosity: 0.05,
    },
    stddev: { sharpness: 0.12, roughness: 0.09, hardness: 0.06 },
  },
  {
    debugFamily: 'ite-like',
    mean: {
      mass: 0.72,
      density: 0.78,
      hardness: 0.65,
      sharpness: 0.15,
      brittleness: 0.55,
      elasticity: 0.3,
      roughness: 0.55,
      tensile_strength: 0.42,
      compressive_strength: 0.7,
      friction_coeff: 0.52,
      thermal_conductivity: 0.28,
      heat_capacity: 0.62,
      combustibility: 0.04,
      toxicity: 0.05,
      conductivity: 0.12,
      malleability: 0.18,
      porosity: 0.3,
    },
    stddev: { brittleness: 0.08, porosity: 0.1, hardness: 0.08 },
  },
  {
    debugFamily: 'paste-like',
    mean: {
      mass: 0.58,
      density: 0.62,
      hardness: 0.32,
      sharpness: 0.05,
      brittleness: 0.12,
      elasticity: 0.45,
      roughness: 0.75,
      tensile_strength: 0.38,
      compressive_strength: 0.48,
      friction_coeff: 0.72,
      thermal_conductivity: 0.22,
      heat_capacity: 0.72,
      combustibility: 0.15,
      toxicity: 0.08,
      conductivity: 0.1,
      malleability: 0.82,
      porosity: 0.42,
    },
    stddev: { malleability: 0.1, roughness: 0.08, friction_coeff: 0.08 },
  },
  {
    debugFamily: 'resin-like',
    mean: {
      mass: 0.35,
      density: 0.42,
      hardness: 0.38,
      sharpness: 0.08,
      brittleness: 0.35,
      elasticity: 0.55,
      roughness: 0.25,
      tensile_strength: 0.52,
      compressive_strength: 0.35,
      friction_coeff: 0.45,
      thermal_conductivity: 0.12,
      heat_capacity: 0.48,
      combustibility: 0.55,
      toxicity: 0.22,
      conductivity: 0.08,
      malleability: 0.6,
      porosity: 0.08,
    },
    stddev: { combustibility: 0.1, elasticity: 0.08, toxicity: 0.1 },
  },
  {
    debugFamily: 'ite-glass',
    mean: {
      mass: 0.68,
      density: 0.82,
      hardness: 0.78,
      sharpness: 0.72,
      brittleness: 0.88,
      elasticity: 0.12,
      roughness: 0.15,
      tensile_strength: 0.22,
      compressive_strength: 0.62,
      friction_coeff: 0.2,
      thermal_conductivity: 0.55,
      heat_capacity: 0.35,
      combustibility: 0.0,
      toxicity: 0.3,
      conductivity: 0.15,
      malleability: 0.05,
      porosity: 0.02,
    },
    stddev: { sharpness: 0.1, brittleness: 0.06, toxicity: 0.12 },
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
