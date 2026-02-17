import type { Vec2, WorldObject } from './object_model';
import { clamp } from './properties';

export interface StationQualityMetrics {
  stiffness: number;
  precisionSurface: number;
  geometricAlignment: number;
  thermalStability: number;
}

export interface StationBonuses {
  processNoiseScale: number;
  repeatabilityGain: number;
  measurementPrecisionGain: number;
  maxPlanarityGain: number;
}

export type StationFunction = 'storage' | 'workshop' | 'purifier' | 'beacon';

export interface AnchoredStation {
  objectId: number;
  worldPos: Vec2;
  quality: number;
  functionType: StationFunction;
  metrics: StationQualityMetrics;
  bonuses: StationBonuses;
}

export function computeStationQualityMetrics(obj: WorldObject): StationQualityMetrics {
  const stiffness = clamp((obj.props.tensile_strength * 0.55 + obj.props.compressive_strength * 0.45) * (0.7 + obj.integrity * 0.3));
  const precisionSurface = clamp(obj.latentPrecision.surface_planarity * 0.55 + obj.latentPrecision.microstructure_order * 0.45);
  const geometricAlignment = clamp(1 - Math.hypot(obj.center_of_mass_offset.x, obj.center_of_mass_offset.y) / Math.max(0.2, obj.length * 0.5));
  const thermalStability = clamp((obj.props.heat_capacity * 0.6 + obj.props.mass * 0.4) * (0.65 + obj.latentPrecision.microstructure_order * 0.35));
  return { stiffness, precisionSurface, geometricAlignment, thermalStability };
}

export function stationFromAnchoredObject(obj: WorldObject): AnchoredStation {
  const metrics = computeStationQualityMetrics(obj);
  const quality = clamp(metrics.stiffness * 0.3 + metrics.precisionSurface * 0.3 + metrics.geometricAlignment * 0.2 + metrics.thermalStability * 0.2);
  const bonuses: StationBonuses = {
    processNoiseScale: clamp(1 - quality * 0.45, 0.4, 1),
    repeatabilityGain: clamp(quality * 0.55),
    measurementPrecisionGain: clamp(quality * 0.6),
    maxPlanarityGain: clamp(quality * 0.35),
  };
  return {
    objectId: obj.id,
    worldPos: { ...obj.pos },
    quality,
    functionType: 'workshop',
    metrics,
    bonuses,
  };
}

export function anchorObject(obj: WorldObject, worldPos: Vec2): WorldObject {
  return {
    ...obj,
    pos: { ...worldPos },
    vel: { x: 0, y: 0 },
    anchored: true,
  };
}
