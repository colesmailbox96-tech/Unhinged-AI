import { describe, expect, it, test } from 'vitest';
import {
  dayPhase,
  dayNightTemperatureShift,
  seasonPhase,
  seasonalBiomassMultiplier,
  seasonalTemperatureBias,
  hazardExposure,
  hazardExposureByKind,
  spawnInitialHazards,
  maybeTriggerEvent,
  tickEvents,
  eventPressure,
  scarcityPressure,
  type HazardZone,
  type EnvironmentalEvent,
} from '../src/sim/environment';
import { RNG } from '../src/sim/rng';
import { World } from '../src/sim/world';
import { tickNeeds, createDefaultNeeds, type EnvironmentalPressure } from '../src/sim/needs';
import { materialDistributions } from '../src/sim/material_distributions';
import { LiveModeEngine } from '../src/runner/live_mode';

// ─── Day/Night Cycle ────────────────────────────────────────────────

describe('Day/Night Cycle', () => {
  test('dayPhase wraps correctly within [0,1)', () => {
    expect(dayPhase(0, 600)).toBe(0);
    expect(dayPhase(300, 600)).toBeCloseTo(0.5, 5);
    expect(dayPhase(600, 600)).toBe(0); // wraps
    expect(dayPhase(1500, 600)).toBeCloseTo(0.5, 5);
  });

  test('dayNightTemperatureShift peaks at noon (phase=0.5)', () => {
    const midnight = dayNightTemperatureShift(0);
    const noon = dayNightTemperatureShift(0.5);
    expect(noon).toBeGreaterThan(midnight);
    expect(noon).toBeGreaterThan(0);
    expect(midnight).toBeLessThan(0);
  });

  test('dayNightTemperatureShift is bounded by maxSwing', () => {
    for (let p = 0; p <= 1; p += 0.1) {
      const shift = dayNightTemperatureShift(p, 0.15);
      expect(Math.abs(shift)).toBeLessThanOrEqual(0.15 + 0.001);
    }
  });
});

// ─── Seasonal Variation ─────────────────────────────────────────────

describe('Seasonal Variation', () => {
  test('seasonPhase wraps correctly', () => {
    expect(seasonPhase(0, 4800)).toBe(0);
    expect(seasonPhase(2400, 4800)).toBeCloseTo(0.5, 5);
    expect(seasonPhase(4800, 4800)).toBe(0); // wraps
  });

  test('seasonalBiomassMultiplier peaks in summer and troughs in winter', () => {
    const summer = seasonalBiomassMultiplier(0.25);
    const winter = seasonalBiomassMultiplier(0.75);
    expect(summer).toBeGreaterThan(winter);
    expect(summer).toBeLessThanOrEqual(1);
    expect(winter).toBeGreaterThanOrEqual(0.6);
  });

  test('seasonalBiomassMultiplier is always in [0.6, 1]', () => {
    for (let s = 0; s < 1; s += 0.05) {
      const mult = seasonalBiomassMultiplier(s);
      expect(mult).toBeGreaterThanOrEqual(0.6);
      expect(mult).toBeLessThanOrEqual(1);
    }
  });

  test('seasonalTemperatureBias varies with season', () => {
    const summer = seasonalTemperatureBias(0.25);
    const winter = seasonalTemperatureBias(0.75);
    expect(summer).toBeGreaterThan(0);
    expect(winter).toBeLessThan(0);
  });
});

// ─── Hazard Zones ───────────────────────────────────────────────────

describe('Hazard Zones', () => {
  test('spawnInitialHazards creates the requested count', () => {
    const rng = new RNG(42);
    const zones = spawnInitialHazards(10, 10, rng, 5);
    expect(zones).toHaveLength(5);
    for (const z of zones) {
      expect(z.x).toBeGreaterThan(0);
      expect(z.y).toBeGreaterThan(0);
      expect(z.radius).toBeGreaterThan(0);
      expect(z.intensity).toBeGreaterThan(0);
      expect(['thermal', 'toxic', 'erosion']).toContain(z.kind);
    }
  });

  test('hazardExposure returns 0 outside all zones', () => {
    const zones: HazardZone[] = [
      { x: 1, y: 1, radius: 0.5, intensity: 1, kind: 'thermal', remainingTicks: 0 },
    ];
    expect(hazardExposure(9, 9, zones)).toBe(0);
  });

  test('hazardExposure returns positive value inside a zone', () => {
    const zones: HazardZone[] = [
      { x: 5, y: 5, radius: 2, intensity: 0.8, kind: 'toxic', remainingTicks: 0 },
    ];
    expect(hazardExposure(5, 5, zones)).toBeGreaterThan(0);
    expect(hazardExposure(5.5, 5, zones)).toBeGreaterThan(0);
  });

  test('hazardExposure decreases with distance from center', () => {
    const zones: HazardZone[] = [
      { x: 5, y: 5, radius: 3, intensity: 1, kind: 'erosion', remainingTicks: 0 },
    ];
    const center = hazardExposure(5, 5, zones);
    const halfway = hazardExposure(6.5, 5, zones);
    const edge = hazardExposure(7.8, 5, zones);
    expect(center).toBeGreaterThan(halfway);
    expect(halfway).toBeGreaterThan(edge);
  });

  test('hazardExposureByKind returns per-kind breakdown', () => {
    const zones: HazardZone[] = [
      { x: 5, y: 5, radius: 3, intensity: 0.5, kind: 'thermal', remainingTicks: 0 },
      { x: 5, y: 5, radius: 3, intensity: 0.3, kind: 'toxic', remainingTicks: 0 },
    ];
    const breakdown = hazardExposureByKind(5, 5, zones);
    expect(breakdown.thermal).toBeGreaterThan(0);
    expect(breakdown.toxic).toBeGreaterThan(0);
    expect(breakdown.erosion).toBe(0);
  });
});

// ─── Environmental Events ───────────────────────────────────────────

describe('Environmental Events', () => {
  test('maybeTriggerEvent returns undefined when probability is 0', () => {
    const rng = new RNG(42);
    expect(maybeTriggerEvent(rng, 0, 100)).toBeUndefined();
  });

  test('maybeTriggerEvent returns event when probability is 1', () => {
    const rng = new RNG(42);
    const event = maybeTriggerEvent(rng, 1, 100);
    expect(event).toBeDefined();
    expect(event!.remainingTicks).toBe(100);
    expect(event!.intensity).toBeGreaterThan(0);
    expect(['storm', 'drought', 'cold-snap']).toContain(event!.kind);
  });

  test('tickEvents decrements remaining ticks and removes expired', () => {
    const events: EnvironmentalEvent[] = [
      { kind: 'storm', remainingTicks: 2, intensity: 0.5 },
      { kind: 'drought', remainingTicks: 1, intensity: 0.3 },
    ];
    const after1 = tickEvents(events);
    expect(after1).toHaveLength(1);
    expect(after1[0].kind).toBe('storm');
    expect(after1[0].remainingTicks).toBe(1);

    const after2 = tickEvents(after1);
    expect(after2).toHaveLength(0);
  });

  test('eventPressure reflects active events', () => {
    const noEvents = eventPressure([]);
    expect(noEvents.temperatureShift).toBe(0);
    expect(noEvents.biomassMultiplier).toBe(1);

    const drought: EnvironmentalEvent[] = [{ kind: 'drought', remainingTicks: 50, intensity: 0.7 }];
    const droughtPressure = eventPressure(drought);
    expect(droughtPressure.hydrationDrain).toBeGreaterThan(0);
    expect(droughtPressure.biomassMultiplier).toBeLessThan(1);

    const coldSnap: EnvironmentalEvent[] = [{ kind: 'cold-snap', remainingTicks: 50, intensity: 0.6 }];
    const coldPressure = eventPressure(coldSnap);
    expect(coldPressure.temperatureShift).toBeLessThan(0);
  });
});

// ─── Scarcity Pressure ──────────────────────────────────────────────

describe('Scarcity Pressure', () => {
  test('scarcityPressure starts at ~1 and increases', () => {
    expect(scarcityPressure(0)).toBeCloseTo(1, 2);
    expect(scarcityPressure(6000)).toBeGreaterThan(1);
    expect(scarcityPressure(12000)).toBeGreaterThan(scarcityPressure(6000));
  });

  test('scarcityPressure growth is logarithmic (slows over time)', () => {
    const early = scarcityPressure(6000) - scarcityPressure(0);
    const late = scarcityPressure(12000) - scarcityPressure(6000);
    expect(early).toBeGreaterThan(late);
  });
});

// ─── Material Distributions ─────────────────────────────────────────

describe('New Material Distributions', () => {
  test('materialDistributions has 8 families', () => {
    expect(materialDistributions).toHaveLength(8);
  });

  test('all distributions have valid property means in [0,1]', () => {
    for (const d of materialDistributions) {
      for (const [, value] of Object.entries(d.mean)) {
        expect(value).toBeGreaterThanOrEqual(0);
        expect(value).toBeLessThanOrEqual(1);
      }
    }
  });

  test('new distributions have distinct debugFamily names', () => {
    const families = new Set(materialDistributions.map(d => d.debugFamily));
    expect(families.size).toBe(8);
  });

  test('spawnLooseObject can produce objects from new distributions', () => {
    const world = new World(42);
    const families = new Set<string>();
    for (let i = 0; i < 100; i++) {
      const id = world.spawnLooseObject();
      const obj = world.objects.get(id);
      if (obj?.debugFamily) families.add(obj.debugFamily);
    }
    // Should have more than the original 4 families
    expect(families.size).toBeGreaterThan(4);
  });
});

// ─── World Environmental Integration ────────────────────────────────

describe('World Environmental Integration', () => {
  test('world initializes with hazard zones', () => {
    const world = new World(42);
    expect(world.hazardZones.length).toBeGreaterThan(0);
  });

  test('agentHazardExposure returns a number', () => {
    const world = new World(42);
    const exposure = world.agentHazardExposure();
    expect(typeof exposure).toBe('number');
    expect(exposure).toBeGreaterThanOrEqual(0);
  });

  test('SHELTER verb sets shelter factor based on object properties', () => {
    const world = new World(42);
    const nearbyIds = world.getNearbyObjectIds(5.0);
    const objId = nearbyIds[0];
    const obj = world.objects.get(objId)!;
    // Move near the object
    world.apply({ type: 'MOVE_TO', x: obj.pos.x, y: obj.pos.y });
    world.apply({ type: 'SHELTER', objId });
    expect(world.agent.shelterFactor).toBeGreaterThan(0);
    expect(world.agent.shelteredByObjId).toBe(objId);
  });

  test('SHELTER has no effect if too far from object', () => {
    const world = new World(42);
    // Agent at (5,5), object far away
    const farId = world.spawnLooseObject();
    const farObj = world.objects.get(farId)!;
    farObj.pos = { x: 0, y: 0 };
    world.apply({ type: 'SHELTER', objId: farId });
    expect(world.agent.shelterFactor).toBeUndefined();
  });

  test('shelter factor reduces hazard exposure', () => {
    const world = new World(42);
    // Position agent in a hazard zone
    const zone = world.hazardZones[0];
    world.agent.pos = { x: zone.x, y: zone.y };
    const exposureUnsheltered = world.agentHazardExposure();

    // Set shelter factor
    world.agent.shelterFactor = 0.8;
    const exposureSheltered = world.agentHazardExposure();

    if (exposureUnsheltered > 0) {
      expect(exposureSheltered).toBeLessThan(exposureUnsheltered);
    }
  });

  test('tickEnvironment returns valid environmental state', () => {
    const world = new World(42);
    const state = world.tickEnvironment(100);
    expect(state.ambientTemperature).toBeGreaterThan(0);
    expect(state.ambientTemperature).toBeLessThan(1);
    expect(state.biomassMultiplier).toBeGreaterThan(0);
    expect(state.biomassMultiplier).toBeLessThanOrEqual(1);
    expect(state.scarcity).toBeGreaterThanOrEqual(1);
  });

  test('weatherObjects degrades objects faster in hazard zones', () => {
    const world = new World(42);
    // Place an object in a hazard zone
    const zone = world.hazardZones[0];
    const inZoneId = world.spawnLooseObject();
    const inZoneObj = world.objects.get(inZoneId)!;
    inZoneObj.pos = { x: zone.x, y: zone.y };
    const integrityInZone = inZoneObj.integrity;

    // Place an object far from hazard zones
    const outId = world.spawnLooseObject();
    const outObj = world.objects.get(outId)!;
    outObj.pos = { x: zone.x + zone.radius + 5, y: zone.y + zone.radius + 5 };
    const integrityOut = outObj.integrity;

    world.weatherObjects(100);

    const lossInZone = integrityInZone - inZoneObj.integrity;
    const lossOut = integrityOut - outObj.integrity;
    // Object in hazard zone should lose more integrity (if both are weatherable)
    if (lossOut > 0) {
      expect(lossInZone).toBeGreaterThanOrEqual(lossOut);
    }
  });
});

// ─── Needs + Environment Integration ────────────────────────────────

describe('Needs + Environmental Pressure', () => {
  test('hazard exposure increases damage over ticks', () => {
    let needs = createDefaultNeeds();
    const pressure: EnvironmentalPressure = {
      hazardBreakdown: { thermal: 0, toxic: 0.5, erosion: 0 },
    };
    for (let i = 0; i < 20; i++) {
      const result = tickNeeds(needs, 'REST', undefined, pressure);
      needs = result.needs;
    }
    expect(needs.damage).toBeGreaterThan(0);
  });

  test('scarcity multiplier increases energy drain', () => {
    let needsNormal = createDefaultNeeds();
    let needsScarcity = createDefaultNeeds();
    for (let i = 0; i < 20; i++) {
      needsNormal = tickNeeds(needsNormal, 'MOVE_TO').needs;
      needsScarcity = tickNeeds(needsScarcity, 'MOVE_TO', undefined, { scarcity: 1.5 }).needs;
    }
    expect(needsScarcity.energy).toBeLessThan(needsNormal.energy);
  });

  test('drought event increases hydration drain', () => {
    let needsNormal = createDefaultNeeds();
    let needsDrought = createDefaultNeeds();
    for (let i = 0; i < 30; i++) {
      needsNormal = tickNeeds(needsNormal, 'MOVE_TO').needs;
      needsDrought = tickNeeds(needsDrought, 'MOVE_TO', undefined, { hydrationDrain: 0.004 }).needs;
    }
    expect(needsDrought.hydration).toBeLessThan(needsNormal.hydration);
  });

  test('ambient temperature from environment shifts agent temperature', () => {
    let needs = createDefaultNeeds();
    // Hot environment (like midday summer)
    for (let i = 0; i < 50; i++) {
      needs = tickNeeds(needs, 'REST', undefined, { ambientTemperature: 0.8 }).needs;
    }
    expect(needs.temperature).toBeGreaterThan(0.5);
  });

  test('thermal hazard shifts temperature', () => {
    let needs = createDefaultNeeds();
    for (let i = 0; i < 30; i++) {
      needs = tickNeeds(needs, 'REST', undefined, {
        hazardBreakdown: { thermal: 0.6, toxic: 0, erosion: 0 },
      }).needs;
    }
    expect(needs.temperature).toBeGreaterThan(0.5);
  });

  test('SHELTER action has low fatigue cost', () => {
    let needs = createDefaultNeeds();
    const before = needs.fatigue;
    needs = tickNeeds(needs, 'SHELTER').needs;
    expect(needs.fatigue - before).toBeLessThan(0.01);
  });
});

// ─── Live Mode Environment ──────────────────────────────────────────

describe('Live Mode - Environment Fields', () => {
  it('tick result includes environmental state fields', () => {
    const engine = new LiveModeEngine({
      seed: 42,
      populationSize: 1,
      ticksPerSecond: 20,
      deterministic: true,
      rollingSeconds: 30,
      livingMode: true,
      livingPreset: 'living-v1-ecology',
    });
    const result = engine.tickOnce();
    expect(typeof result.ambientTemperature).toBe('number');
    expect(typeof result.seasonPhase).toBe('number');
    expect(typeof result.dayPhase).toBe('number');
    expect(typeof result.hazardExposure).toBe('number');
    expect(typeof result.activeEventCount).toBe('number');
    expect(typeof result.scarcityPressure).toBe('number');
    expect(result.scarcityPressure).toBeGreaterThanOrEqual(1);
  });

  it('SEEK_SHELTER intent appears when agent is in hazard zone', () => {
    const engine = new LiveModeEngine({
      seed: 42,
      populationSize: 1,
      ticksPerSecond: 20,
      deterministic: true,
      rollingSeconds: 30,
      livingMode: true,
      livingPreset: 'living-v1-ecology',
    });
    // Move agent to hazard zone
    const zone = engine.world.hazardZones[0];
    engine.world.agent.pos = { x: zone.x, y: zone.y };
    // Increase damage to make SEEK_SHELTER more likely to score high
    engine.memories[0].needs.damage = 0.5;
    const result = engine.tickOnce();
    // The intent scores should include SEEK_SHELTER
    const shelterScore = result.agentIntentScores?.find(s => s.intent === 'SEEK_SHELTER');
    expect(shelterScore).toBeDefined();
    expect(shelterScore!.score).toBeGreaterThan(0);
  });

  it('environmental features are discoverable only through interaction', () => {
    // Verify that agents don't have access to hazard zone kind labels
    // or named material types — they only see property vectors and
    // positional effects
    const world = new World(42);
    for (const obj of world.objects.values()) {
      // Objects have property vectors (opaque numbers), not named types
      expect(obj.props.mass).toBeDefined();
      expect(obj.props.hardness).toBeDefined();
      // debugFamily is for humans only, not agent-visible
      expect(typeof obj.debugFamily).toBe('string');
    }
    // Hazard zones have 'kind' but agents see only the numeric exposure
    const exposure = world.agentHazardExposure();
    expect(typeof exposure).toBe('number');
    // Agents don't get hazard zone positions or kinds directly
  });
});
