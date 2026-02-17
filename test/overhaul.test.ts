import { describe, expect, test } from 'vitest';
import { RNG } from '../src/sim/rng';
import { SpatialGrid } from '../src/sim/spatial_grid';
import { EventBus, type SimEvent } from '../src/sim/events';
import { ReplayBuffer } from '../src/ai/replay_buffer';
import { propertyDistance, propertySimilarity, type PropertyVector } from '../src/sim/properties';
import { World } from '../src/sim/world';
import { resetMeasurementStats } from '../src/sim/metrology';

function baseProps(overrides: Partial<PropertyVector> = {}): PropertyVector {
  return {
    mass: 0.5, density: 0.5, hardness: 0.5, sharpness: 0.5,
    brittleness: 0.5, elasticity: 0.5, roughness: 0.5,
    tensile_strength: 0.5, compressive_strength: 0.5, friction_coeff: 0.5,
    thermal_conductivity: 0.5, heat_capacity: 0.5, combustibility: 0.5,
    toxicity: 0.5, conductivity: 0.5, malleability: 0.5, porosity: 0.5,
    ...overrides,
  };
}

// ---------- RNG ----------
describe('RNG enhancements', () => {
  test('clone preserves exact state', () => {
    const a = new RNG(42);
    // Advance state
    for (let i = 0; i < 10; i++) a.float();
    const b = a.clone();
    // Both should produce identical sequences
    const seqA = Array.from({ length: 20 }, () => a.float());
    const seqB = Array.from({ length: 20 }, () => b.float());
    expect(seqA).toEqual(seqB);
  });

  test('clone preserves spare normal', () => {
    const a = new RNG(99);
    a.normal(); // generates spare
    const b = a.clone();
    const nA = a.normal(); // should use spare
    const nB = b.normal(); // should also use spare
    expect(nA).toBe(nB);
  });

  test('pick returns element from array', () => {
    const rng = new RNG(7);
    const arr = [10, 20, 30, 40, 50];
    const result = rng.pick(arr);
    expect(arr).toContain(result);
  });

  test('pick returns undefined for empty array', () => {
    const rng = new RNG(7);
    expect(rng.pick([])).toBeUndefined();
  });

  test('shuffle returns permutation of same elements', () => {
    const rng = new RNG(100);
    const arr = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
    const shuffled = rng.shuffle(arr);
    expect(shuffled).toHaveLength(arr.length);
    expect(shuffled.sort((a, b) => a - b)).toEqual(arr);
  });

  test('shuffle does not mutate original', () => {
    const rng = new RNG(100);
    const arr = [1, 2, 3];
    rng.shuffle(arr);
    expect(arr).toEqual([1, 2, 3]);
  });
});

// ---------- SpatialGrid ----------
describe('SpatialGrid', () => {
  test('inserts and queries by radius', () => {
    const grid = new SpatialGrid<{ id: number; pos: { x: number; y: number } }>(2.0);
    grid.upsert({ id: 1, pos: { x: 1, y: 1 } });
    grid.upsert({ id: 2, pos: { x: 1.5, y: 1.5 } });
    grid.upsert({ id: 3, pos: { x: 10, y: 10 } });

    const near = grid.queryRadius(1, 1, 1.0);
    expect(near.map(i => i.id).sort()).toEqual([1, 2]);

    const far = grid.queryRadius(10, 10, 0.5);
    expect(far.map(i => i.id)).toEqual([3]);
  });

  test('remove correctly excludes item', () => {
    const grid = new SpatialGrid<{ id: number; pos: { x: number; y: number } }>(2.0);
    grid.upsert({ id: 1, pos: { x: 1, y: 1 } });
    grid.upsert({ id: 2, pos: { x: 1.2, y: 1.2 } });
    grid.remove(1);
    const near = grid.queryRadius(1, 1, 2.0);
    expect(near.map(i => i.id)).toEqual([2]);
    expect(grid.size).toBe(1);
  });

  test('upsert updates position when item moves', () => {
    const grid = new SpatialGrid<{ id: number; pos: { x: number; y: number } }>(2.0);
    const item = { id: 1, pos: { x: 0, y: 0 } };
    grid.upsert(item);
    expect(grid.queryRadius(0, 0, 0.5).length).toBe(1);

    // Move to a different cell
    item.pos = { x: 10, y: 10 };
    grid.upsert(item);
    expect(grid.queryRadius(0, 0, 0.5).length).toBe(0);
    expect(grid.queryRadius(10, 10, 0.5).length).toBe(1);
  });

  test('clear removes everything', () => {
    const grid = new SpatialGrid<{ id: number; pos: { x: number; y: number } }>(2.0);
    for (let i = 0; i < 20; i++) grid.upsert({ id: i, pos: { x: i, y: i } });
    expect(grid.size).toBe(20);
    grid.clear();
    expect(grid.size).toBe(0);
  });
});

// ---------- EventBus ----------
describe('EventBus', () => {
  test('listeners receive matching events', () => {
    const bus = new EventBus();
    const received: SimEvent[] = [];
    bus.on('object-created', (e) => received.push(e));

    bus.emit({ kind: 'object-created', tick: 1, ids: [10], data: {} });
    bus.emit({ kind: 'object-destroyed', tick: 2, ids: [10], data: {} });

    expect(received).toHaveLength(1);
    expect(received[0].tick).toBe(1);
  });

  test('wildcard listener receives all events', () => {
    const bus = new EventBus();
    const received: SimEvent[] = [];
    bus.on('*', (e) => received.push(e));

    bus.emit({ kind: 'object-created', tick: 1, ids: [], data: {} });
    bus.emit({ kind: 'skill-learned', tick: 2, ids: [], data: {} });

    expect(received).toHaveLength(2);
  });

  test('unsubscribe stops receiving', () => {
    const bus = new EventBus();
    const received: SimEvent[] = [];
    const unsub = bus.on('stall-detected', (e) => received.push(e));

    bus.emit({ kind: 'stall-detected', tick: 1, ids: [], data: {} });
    unsub();
    bus.emit({ kind: 'stall-detected', tick: 2, ids: [], data: {} });

    expect(received).toHaveLength(1);
  });

  test('history respects max capacity', () => {
    const bus = new EventBus(5);
    for (let i = 0; i < 10; i++) {
      bus.emit({ kind: 'agent-action', tick: i, ids: [], data: {} });
    }
    expect(bus.recent().length).toBeLessThanOrEqual(5);
  });

  test('countSince filters by tick', () => {
    const bus = new EventBus();
    bus.emit({ kind: 'object-created', tick: 10, ids: [], data: {} });
    bus.emit({ kind: 'object-created', tick: 20, ids: [], data: {} });
    bus.emit({ kind: 'object-created', tick: 30, ids: [], data: {} });

    expect(bus.countSince('object-created', 15)).toBe(2);
    expect(bus.countSince('object-created', 25)).toBe(1);
  });
});

// ---------- Prioritized ReplayBuffer ----------
describe('Prioritized ReplayBuffer', () => {
  test('samplePrioritized returns requested count', () => {
    const buf = new ReplayBuffer<string, string>(100);
    for (let i = 0; i < 50; i++) {
      buf.push({ state: `s${i}`, action: `a${i}`, reward: i * 0.01, priority: i * 0.1 });
    }
    const sample = buf.samplePrioritized(10);
    expect(sample).toHaveLength(10);
  });

  test('samplePrioritized favors high priority items', () => {
    const buf = new ReplayBuffer<number, number>(100);
    // Add many low-priority items and one high-priority item
    for (let i = 0; i < 50; i++) {
      buf.push({ state: i, action: 0, reward: 0.01, priority: 0.001 });
    }
    buf.push({ state: 999, action: 0, reward: 1.0, priority: 10.0 });

    // Sample many times and count how often the high-priority item appears
    let highPriorityCount = 0;
    for (let trial = 0; trial < 50; trial++) {
      const sample = buf.samplePrioritized(5, 1.0);
      if (sample.some(s => s.state === 999)) highPriorityCount++;
    }
    // High priority item should appear very frequently
    expect(highPriorityCount).toBeGreaterThan(25);
  });

  test('updateLastPriority modifies the last item', () => {
    const buf = new ReplayBuffer<string, string>(100);
    buf.push({ state: 'a', action: 'x', reward: 0.5 });
    buf.updateLastPriority(5.0);
    const last = buf.sampleLast(1)[0];
    expect(last.priority).toBe(5.0);
  });

  test('samplePrioritized handles empty buffer', () => {
    const buf = new ReplayBuffer<string, string>(10);
    expect(buf.samplePrioritized(5)).toEqual([]);
  });
});

// ---------- Property Distance / Similarity ----------
describe('Property vector utilities', () => {
  test('distance between identical vectors is 0', () => {
    const a = baseProps();
    expect(propertyDistance(a, a)).toBe(0);
  });

  test('distance increases with property differences', () => {
    const a = baseProps();
    const b = baseProps({ hardness: 1.0, mass: 0.0 });
    const c = baseProps({ hardness: 1.0 });
    expect(propertyDistance(a, b)).toBeGreaterThan(propertyDistance(a, c));
  });

  test('similarity of identical vectors is 1', () => {
    const a = baseProps();
    expect(propertySimilarity(a, a)).toBeCloseTo(1.0, 5);
  });

  test('similarity decreases with difference', () => {
    const a = baseProps({ hardness: 0.9, mass: 0.9 });
    const b = baseProps({ hardness: 0.85, mass: 0.85 });
    const c = baseProps({ hardness: 0.1, mass: 0.1 });
    expect(propertySimilarity(a, b)).toBeGreaterThan(propertySimilarity(a, c));
  });
});

// ---------- World input validation ----------
describe('World input validation', () => {
  test('MOVE_TO with NaN coordinates is no-op', () => {
    const world = new World(42);
    const posBefore = { ...world.agent.pos };
    world.apply({ type: 'MOVE_TO', x: NaN, y: 5 });
    expect(world.agent.pos).toEqual(posBefore);
  });

  test('MOVE_TO with Infinity coordinates is no-op', () => {
    const world = new World(42);
    const posBefore = { ...world.agent.pos };
    world.apply({ type: 'MOVE_TO', x: Infinity, y: 5 });
    expect(world.agent.pos).toEqual(posBefore);
  });

  test('HEAT with NaN intensity is no-op', () => {
    const world = new World(42);
    // Pick up an object first
    const objId = [...world.objects.keys()][0];
    world.apply({ type: 'MOVE_TO', x: world.objects.get(objId)!.pos.x, y: world.objects.get(objId)!.pos.y });
    world.apply({ type: 'PICK_UP', objId });
    const before = world.objects.get(world.agent.heldObjectId!)!.props.brittleness;
    world.apply({ type: 'HEAT', intensity: NaN });
    const after = world.objects.get(world.agent.heldObjectId!)!.props.brittleness;
    expect(after).toBe(before);
  });
});

// ---------- Object weathering ----------
describe('Object weathering', () => {
  test('weatherObjects decreases integrity of loose objects', () => {
    const world = new World(42);
    const looseObj = [...world.objects.values()].find(o => !o.anchored && o.heldBy === undefined)!;
    const before = looseObj.integrity;
    world.weatherObjects(100);
    expect(looseObj.integrity).toBeLessThan(before);
  });

  test('weatherObjects increments ageTicks', () => {
    const world = new World(42);
    const obj = [...world.objects.values()][0];
    world.weatherObjects(50);
    expect(obj.ageTicks).toBe(50);
  });

  test('held objects are not weathered', () => {
    const world = new World(42);
    const objId = [...world.objects.keys()][0];
    const obj = world.objects.get(objId)!;
    // Move near and pick up
    world.apply({ type: 'MOVE_TO', x: obj.pos.x, y: obj.pos.y });
    world.apply({ type: 'PICK_UP', objId });
    const before = obj.integrity;
    world.weatherObjects(100);
    expect(obj.integrity).toBe(before);
  });
});

// ---------- Spatial grid integration in World ----------
describe('World spatial grid integration', () => {
  test('getNearbyObjectIds returns correct objects', () => {
    const world = new World(42);
    // Agent is at (5, 5); nearby objects should be within radius
    const nearby = world.getNearbyObjectIds(2.5);
    // Verify all returned IDs exist in world
    for (const id of nearby) {
      expect(world.objects.has(id)).toBe(true);
    }
    // Verify no objects within radius are missed
    for (const obj of world.objects.values()) {
      const dist = Math.hypot(obj.pos.x - world.agent.pos.x, obj.pos.y - world.agent.pos.y);
      if (dist <= 2.5) {
        expect(nearby).toContain(obj.id);
      }
    }
  });

  test('spatial grid stays in sync after BIND', () => {
    const world = new World(42);
    const nearby = world.getNearbyObjectIds(5.0);
    // Pick up first object
    const pickId = nearby[0];
    world.apply({ type: 'PICK_UP', objId: pickId });
    // Bind to second
    if (nearby.length > 1) {
      const bindId = nearby[1];
      world.apply({ type: 'BIND_TO', objId: bindId });
      // Original IDs should no longer appear; composite should
      const afterBind = world.getNearbyObjectIds(5.0);
      expect(afterBind).not.toContain(pickId);
      expect(afterBind).not.toContain(bindId);
    }
  });
});

// ---------- Metrology memory management ----------
describe('Metrology memory management', () => {
  test('resetMeasurementStats clears accumulated stats', () => {
    const world = new World(42);
    const objId = [...world.objects.keys()][0];
    // Take measurements to accumulate stats
    world.measureObject(objId);
    world.measureObject(objId);
    resetMeasurementStats();
    // After reset, first measurement should have sampleCount=1
    const results = world.measureObject(objId);
    expect(results.length).toBeGreaterThan(0);
    for (const r of results) {
      expect(r.sampleCount).toBe(1);
    }
  });
});
