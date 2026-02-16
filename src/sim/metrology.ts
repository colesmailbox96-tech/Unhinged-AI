import type { WorldObject } from './object_model';
import { clamp } from './properties';
import type { RNG } from './rng';
import type { AnchoredStation } from './stations';

export type MeasurementKind = 'mass' | 'geometry' | 'hardness' | 'conductivity' | 'optical';

export interface MeasurementResult {
  kind: MeasurementKind;
  value: number | { length: number; thickness: number; flatness: number };
  sigma: number;
  ciLow: number;
  ciHigh: number;
  sampleCount: number;
  instrumentId?: number;
}

interface RunningStats {
  count: number;
  mean: number;
  m2: number;
}

const measurementStats = new Map<string, RunningStats>();

export function resetMetrologyState(): void {
  measurementStats.clear();
}

function keyFor(kind: MeasurementKind, objId: number, instrumentId?: number): string {
  return `${kind}:${objId}:${instrumentId ?? 0}`;
}

function instrumentQuality(instrument?: WorldObject, station?: AnchoredStation): number {
  const base = instrument
    ? clamp(
        instrument.latentPrecision.surface_planarity * 0.35 +
          instrument.latentPrecision.microstructure_order * 0.35 +
          instrument.props.hardness * 0.15 +
          instrument.integrity * 0.15,
      )
    : 0.18;
  const stationBoost = station ? station.bonuses.measurementPrecisionGain * 0.7 + station.quality * 0.1 : 0;
  return clamp(base + stationBoost);
}

function updateStats(stats: RunningStats, sample: number): RunningStats {
  const count = stats.count + 1;
  const delta = sample - stats.mean;
  const mean = stats.mean + delta / count;
  const delta2 = sample - mean;
  return { count, mean, m2: stats.m2 + delta * delta2 };
}

function summarize(kind: MeasurementKind, raw: number, objId: number, sigmaBase: number, instrumentId?: number): MeasurementResult {
  const key = keyFor(kind, objId, instrumentId);
  const next = updateStats(measurementStats.get(key) ?? { count: 0, mean: 0, m2: 0 }, raw);
  measurementStats.set(key, next);
  const sampleVariance = next.count > 1 ? next.m2 / (next.count - 1) : sigmaBase ** 2;
  const standardError = Math.sqrt(sampleVariance / Math.max(1, next.count));
  const ci = 1.96 * standardError;
  return {
    kind,
    value: clamp(next.mean),
    sigma: standardError,
    ciLow: clamp(next.mean - ci),
    ciHigh: clamp(next.mean + ci),
    sampleCount: next.count,
    instrumentId,
  };
}

function measuredScalar(
  kind: MeasurementKind,
  truth: number,
  obj: WorldObject,
  rng: RNG,
  sigmaBase: number,
  instrument?: WorldObject,
  station?: AnchoredStation,
): MeasurementResult {
  const quality = instrumentQuality(instrument, station);
  const noiseScale = clamp(1.08 - quality * 0.72, 0.25, 1.1) * (station?.bonuses.processNoiseScale ?? 1);
  const bias = (1 - quality) * 0.18 * sigmaBase;
  const noisy = truth + bias + rng.normal(0, sigmaBase * noiseScale);
  const result = summarize(kind, noisy, obj.id, sigmaBase, instrument?.id);
  return result;
}

export function measureMass(obj: WorldObject, rng: RNG, instrument?: WorldObject, station?: AnchoredStation): MeasurementResult {
  return measuredScalar('mass', obj.props.mass, obj, rng, 0.06, instrument, station);
}

export function measureGeometry(
  obj: WorldObject,
  rng: RNG,
  instrument?: WorldObject,
  station?: AnchoredStation,
): MeasurementResult {
  const quality = instrumentQuality(instrument, station);
  const noiseScale = clamp(1.05 - quality * 0.7, 0.3, 1.05) * (station?.bonuses.processNoiseScale ?? 1);
  const bias = (1 - quality) * 0.02;
  const length = clamp(obj.length / 2.5 + bias + rng.normal(0, 0.05 * noiseScale));
  const thickness = clamp(obj.thickness / 1.2 + bias + rng.normal(0, 0.04 * noiseScale));
  const flatness = clamp(obj.latentPrecision.surface_planarity + bias + rng.normal(0, 0.05 * noiseScale));
  const flatnessSummary = summarize('geometry', flatness, obj.id, 0.05, instrument?.id);
  return {
    ...flatnessSummary,
    value: { length, thickness, flatness: Number(flatnessSummary.value) },
  };
}

export function measureHardness(obj: WorldObject, rng: RNG, instrument?: WorldObject, station?: AnchoredStation): MeasurementResult {
  const truth = clamp(obj.props.hardness * (0.8 + obj.latentPrecision.microstructure_order * 0.2) - obj.latentPrecision.internal_stress * 0.12);
  return measuredScalar('hardness', truth, obj, rng, 0.08, instrument, station);
}

export function measureConductivity(obj: WorldObject, rng: RNG, instrument?: WorldObject, station?: AnchoredStation): MeasurementResult {
  const truth = clamp(
    obj.props.thermal_conductivity *
      (0.55 + obj.latentPrecision.microstructure_order * 0.45) *
      (1 - obj.latentPrecision.impurity_level * 0.55),
  );
  return measuredScalar('conductivity', truth, obj, rng, 0.07, instrument, station);
}

export function measureOptical(obj: WorldObject, rng: RNG, instrument?: WorldObject, station?: AnchoredStation): MeasurementResult {
  const truth = clamp(obj.latentPrecision.microstructure_order * 0.6 + obj.latentPrecision.surface_planarity * 0.3 + (1 - obj.latentPrecision.impurity_level) * 0.1);
  return measuredScalar('optical', truth, obj, rng, 0.07, instrument, station);
}
