/**
 * Environment — environmental hazards, day/night cycles, seasonal variation,
 * and hazard zones that create evolutionary pressure (urgency to adapt).
 *
 * The simulation agents must discover the effects of these systems through
 * interaction; nothing is labelled or named for the agent.
 */

import { clamp } from './properties';
import type { RNG } from './rng';

// ─── Day/Night cycle ────────────────────────────────────────────────

/**
 * A continuous cycle producing a value in [0,1] representing time of day.
 * 0 = midnight, 0.5 = noon.  Derived from tick count and cycle length.
 */
export function dayPhase(tick: number, cycleLengthTicks: number): number {
  return (tick % cycleLengthTicks) / cycleLengthTicks;
}

/**
 * Ambient temperature modifier from day/night cycle.
 * Peaks at noon (phase=0.5), coldest at midnight (phase=0).
 * Returns a deviation in [-maxSwing, +maxSwing].
 */
export function dayNightTemperatureShift(phase: number, maxSwing = 0.15): number {
  // Sinusoidal with peak at phase=0.5
  return Math.sin(phase * Math.PI * 2 - Math.PI / 2) * maxSwing;
}

// ─── Seasonal variation ─────────────────────────────────────────────

/**
 * Season phase in [0,1) where 0=spring, 0.25=summer, 0.5=autumn, 0.75=winter.
 */
export function seasonPhase(tick: number, seasonLengthTicks: number): number {
  return (tick % seasonLengthTicks) / seasonLengthTicks;
}

const SEASONAL_BASE = 0.7;
const SEASONAL_AMPLITUDE = 0.3;
const SEASONAL_MIN = 0.6;
const SEASONAL_SUMMER_PHASE = 0.25;

/**
 * Biomass growth multiplier from season.
 * Higher in spring/summer, lower in autumn/winter.
 * Range is [SEASONAL_MIN, 1.0] — subtle enough to not break existing progression.
 */
export function seasonalBiomassMultiplier(season: number): number {
  return clamp(
    SEASONAL_BASE + SEASONAL_AMPLITUDE * Math.sin((season - SEASONAL_SUMMER_PHASE) * Math.PI * 2 + Math.PI / 2),
    SEASONAL_MIN,
    1,
  );
}

/**
 * Seasonal temperature bias.
 */
export function seasonalTemperatureBias(season: number): number {
  return Math.sin((season - 0.25) * Math.PI * 2 + Math.PI / 2) * 0.12;
}

// ─── Hazard zones ───────────────────────────────────────────────────

export interface HazardZone {
  /** World position (center). */
  x: number;
  y: number;
  /** Effective radius. */
  radius: number;
  /** Intensity in [0,1]. Determines damage/effect strength. */
  intensity: number;
  /** The kind of hazard — but agents cannot see this label. */
  kind: 'thermal' | 'toxic' | 'erosion';
  /** How many ticks this zone persists (0 = permanent). */
  remainingTicks: number;
}

/**
 * Spawn initial hazard zones for a world of given dimensions.
 */
export function spawnInitialHazards(
  worldWidth: number,
  worldHeight: number,
  rng: RNG,
  count = 3,
): HazardZone[] {
  const zones: HazardZone[] = [];
  const kinds: HazardZone['kind'][] = ['thermal', 'toxic', 'erosion'];
  for (let i = 0; i < count; i++) {
    zones.push({
      x: rng.range(0.5, worldWidth - 0.5),
      y: rng.range(0.5, worldHeight - 0.5),
      radius: rng.range(1.0, 2.5),
      intensity: rng.range(0.3, 0.8),
      kind: kinds[i % kinds.length],
      remainingTicks: 0, // permanent
    });
  }
  return zones;
}

/**
 * Compute hazard exposure at a given point. Returns total hazard intensity
 * from all zones that overlap. The agent sees this as an opaque number —
 * it must learn what causes it and how to avoid/shelter from it.
 */
export function hazardExposure(
  x: number,
  y: number,
  zones: HazardZone[],
): number {
  let total = 0;
  for (const z of zones) {
    const dist = Math.hypot(z.x - x, z.y - y);
    if (dist < z.radius) {
      const falloff = 1 - dist / z.radius;
      total += z.intensity * falloff;
    }
  }
  return total;
}

/**
 * Per-kind hazard exposure breakdown. Returns intensity for each kind.
 */
export function hazardExposureByKind(
  x: number,
  y: number,
  zones: HazardZone[],
): Record<HazardZone['kind'], number> {
  const out: Record<HazardZone['kind'], number> = { thermal: 0, toxic: 0, erosion: 0 };
  for (const z of zones) {
    const dist = Math.hypot(z.x - x, z.y - y);
    if (dist < z.radius) {
      const falloff = 1 - dist / z.radius;
      out[z.kind] += z.intensity * falloff;
    }
  }
  return out;
}

// ─── Environmental events (storms, droughts) ────────────────────────

export type EnvironmentalEventKind = 'storm' | 'drought' | 'cold-snap';

export interface EnvironmentalEvent {
  kind: EnvironmentalEventKind;
  /** Remaining ticks for this event. */
  remainingTicks: number;
  /** Intensity in [0,1]. */
  intensity: number;
}

/**
 * Possibly trigger a new environmental event based on probability.
 * Returns the event if triggered, otherwise undefined.
 */
export function maybeTriggerEvent(
  rng: RNG,
  probabilityPerTick: number,
  durationTicks: number,
): EnvironmentalEvent | undefined {
  if (rng.float() > probabilityPerTick) return undefined;
  const kinds: EnvironmentalEventKind[] = ['storm', 'drought', 'cold-snap'];
  return {
    kind: rng.pick(kinds) ?? 'storm',
    remainingTicks: durationTicks,
    intensity: rng.range(0.3, 0.9),
  };
}

/**
 * Tick down all active events, removing expired ones.
 */
export function tickEvents(events: EnvironmentalEvent[]): EnvironmentalEvent[] {
  return events
    .map(e => ({ ...e, remainingTicks: e.remainingTicks - 1 }))
    .filter(e => e.remainingTicks > 0);
}

/**
 * Get the aggregate environmental pressure from active events.
 * Returns modifiers that the world/needs systems can apply.
 */
export function eventPressure(events: EnvironmentalEvent[]): {
  temperatureShift: number;
  biomassMultiplier: number;
  hazardIntensityBoost: number;
  hydrationDrain: number;
} {
  let temperatureShift = 0;
  let biomassMultiplier = 1;
  let hazardIntensityBoost = 0;
  let hydrationDrain = 0;

  for (const e of events) {
    switch (e.kind) {
      case 'storm':
        hazardIntensityBoost += e.intensity * 0.3;
        biomassMultiplier *= clamp(1 - e.intensity * 0.2, 0.5, 1);
        break;
      case 'drought':
        hydrationDrain += e.intensity * 0.004;
        biomassMultiplier *= clamp(1 - e.intensity * 0.35, 0.3, 1);
        break;
      case 'cold-snap':
        temperatureShift -= e.intensity * 0.2;
        biomassMultiplier *= clamp(1 - e.intensity * 0.15, 0.5, 1);
        break;
    }
  }

  return { temperatureShift, biomassMultiplier, hazardIntensityBoost, hydrationDrain };
}

// ─── Scarcity pressure (increasing urgency over time) ───────────────

/**
 * Compute energy drain scaling that increases with world age.
 * Simulates increasing environmental harshness — early life must
 * evolve or face extinction as conditions become harder.
 *
 * Returns a multiplier >= 1 that grows logarithmically.
 */
export function scarcityPressure(tick: number, scaleTicks = 6000): number {
  return 1 + Math.log1p(tick / scaleTicks) * 0.15;
}
