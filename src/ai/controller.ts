import { type MeasurementResult, measureGeometry, measureHardness, measureOptical } from '../sim/metrology';
import type { WorldObject } from '../sim/object_model';
import type { World } from '../sim/world';

export type ControllerTargetMetric = 'surface_planarity' | 'microstructure_order' | 'impurity_level' | 'repeatability_strike_damage';
export type ControllerVerb = 'GRIND' | 'HEAT' | 'SOAK' | 'COOL';

export interface ControllerTarget {
  metric: ControllerTargetMetric;
  target: number;
  minSteps?: number;
}

export interface ControllerStep {
  verb: ControllerVerb;
  intensity: number;
  measured: MeasurementResult;
  achieved: number;
  error: number;
}

interface ControllerState {
  param: number;
  direction: 1 | -1;
  bestError: number;
  lastError?: number;
}

function clamp01(v: number): number {
  return Math.max(0, Math.min(1, v));
}

export class ClosedLoopController {
  private readonly states = new Map<string, ControllerState>();

  private keyFor(objectId: number, target: ControllerTarget): string {
    return `${objectId}:${target.metric}:${target.target.toFixed(3)}`;
  }

  private chooseVerb(target: ControllerTargetMetric): ControllerVerb {
    if (target === 'surface_planarity') return 'GRIND';
    if (target === 'microstructure_order') return 'HEAT';
    if (target === 'impurity_level') return 'SOAK';
    return 'GRIND';
  }

  private measureFor(held: WorldObject, world: World, target: ControllerTargetMetric): MeasurementResult {
    if (target === 'surface_planarity') return measureGeometry(held, world.rng);
    if (target === 'microstructure_order') return measureOptical(held, world.rng);
    if (target === 'impurity_level') return measureHardness(held, world.rng);
    return measureHardness(held, world.rng);
  }

  step(world: World, held: WorldObject, target: ControllerTarget): ControllerStep {
    const key = this.keyFor(held.id, target);
    const state = this.states.get(key) ?? { param: 0.5, direction: 1 as const, bestError: Number.POSITIVE_INFINITY };
    const verb = this.chooseVerb(target.metric);
    const measuredBefore = this.measureFor(held, world, target.metric);
    const achievedBefore =
      target.metric === 'surface_planarity'
        ? Number((measuredBefore.value as { flatness: number }).flatness ?? measuredBefore.value)
        : Number(measuredBefore.value);
    const errorBefore = Math.abs(target.target - achievedBefore);
    if (state.lastError !== undefined && errorBefore > state.lastError) state.direction = state.direction === 1 ? -1 : 1;
    const stepSize = errorBefore > 0.2 ? 0.15 : 0.06;
    state.param = clamp01(state.param + state.direction * stepSize);
    if (verb === 'GRIND') {
      const abrasiveId = world
        .getNearbyObjectIds()
        .find((id) => id !== held.id);
      if (abrasiveId) world.apply({ type: 'GRIND', abrasiveId, intensity: state.param });
    }
    if (verb === 'HEAT') world.apply({ type: 'HEAT', intensity: state.param });
    if (verb === 'SOAK') world.apply({ type: 'SOAK', intensity: state.param });
    if (verb === 'COOL') world.apply({ type: 'COOL', intensity: state.param });
    const refreshed = world.objects.get(held.id) ?? held;
    const measured = this.measureFor(refreshed, world, target.metric);
    const achieved =
      target.metric === 'surface_planarity'
        ? Number((measured.value as { flatness: number }).flatness ?? measured.value)
        : Number(measured.value);
    const error = Math.abs(target.target - achieved);
    state.lastError = error;
    state.bestError = Math.min(state.bestError, error);
    this.states.set(key, state);
    return {
      verb,
      intensity: state.param,
      measured,
      achieved,
      error,
    };
  }
}
