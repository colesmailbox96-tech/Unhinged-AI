import { describe, expect, test } from 'vitest';
import { readFileSync, readdirSync, statSync } from 'node:fs';
import { join } from 'node:path';
import { strike, grind } from '../src/sim/interactions';
import { RNG } from '../src/sim/rng';
import { PropertyVector } from '../src/sim/properties';
import { WorldObject } from '../src/sim/object_model';
import { PerceptionHead } from '../src/ai/perception';
import { trainPolicy } from '../src/ai/rl';

function baseProps(overrides: Partial<PropertyVector> = {}): PropertyVector {
  return {
    mass: 0.6,
    density: 0.6,
    hardness: 0.6,
    sharpness: 0.5,
    brittleness: 0.4,
    elasticity: 0.5,
    roughness: 0.5,
    tensile_strength: 0.6,
    compressive_strength: 0.6,
    friction_coeff: 0.5,
    thermal_conductivity: 0.5,
    heat_capacity: 0.5,
    combustibility: 0.3,
    toxicity: 0.1,
    ...overrides,
  };
}

function obj(id: number, props: PropertyVector): WorldObject {
  return {
    id,
    pos: { x: 0, y: 0 },
    vel: { x: 0, y: 0 },
    radius: 0.3,
    length: 1.2,
    props,
    integrity: 1,
  };
}

describe('guardrail keywords', () => {
  test('forbidden recipe/type keywords are absent', () => {
    const collect = (dir: string): string[] =>
      readdirSync(dir)
        .flatMap((entry) => {
          const full = join(dir, entry);
          if (statSync(full).isDirectory()) return collect(full);
          return full.endsWith('.ts') ? [full] : [];
        });
    const files = collect('src');
    const content = files.map((f) => readFileSync(f, 'utf8')).join('\n');
    for (const keyword of ['recipe', 'Axe', 'Pickaxe', 'IronOre', 'CraftTable']) {
      expect(content).not.toContain(keyword);
    }
  });
});

describe('property-only outcomes', () => {
  test('strike damage changes smoothly with target hardness', () => {
    const rng = new RNG(10);
    const tool = obj(1, baseProps({ hardness: 0.85, sharpness: 0.75, mass: 0.8 }));
    const targetSoft = obj(2, baseProps({ hardness: 0.2 }));
    const targetHard = obj(3, baseProps({ hardness: 0.85 }));

    const soft = strike({ ...tool }, { ...targetSoft }, rng, (() => { let i = 100; return () => i++; })()).damage;
    const hard = strike({ ...tool }, { ...targetHard }, rng, (() => { let i = 200; return () => i++; })()).damage;

    expect(soft).toBeGreaterThan(hard);
    expect(soft - hard).toBeGreaterThan(0.1);
  });

  test('grind wear increases with brittleness', () => {
    const abrasive = obj(9, baseProps({ roughness: 0.9, friction_coeff: 0.9 }));
    const lowBrittle = grind(obj(1, baseProps({ brittleness: 0.1 })), abrasive).wear;
    const highBrittle = grind(obj(2, baseProps({ brittleness: 0.9 })), abrasive).wear;
    expect(highBrittle).toBeGreaterThan(lowBrittle);
  });
});

describe('perception learning', () => {
  test('hardness prediction error decreases with interaction experience', () => {
    const rng = new RNG(123);
    const head = new PerceptionHead(999);
    const data = Array.from({ length: 50 }, (_, i) =>
      obj(i + 1, baseProps({ hardness: 0.2 + (i % 10) * 0.07, mass: 0.2 + (i % 7) * 0.1, roughness: 0.1 + (i % 5) * 0.15 })),
    );

    const before = head.hardnessError(data, rng.clone());
    for (let e = 0; e < 6; e++) {
      for (const item of data) {
        const obs = head.observe(item, rng);
        head.train(obs, item.props, 1);
      }
    }
    const after = head.hardnessError(data, rng.clone());

    expect(after).toBeLessThan(before);
  });
});

describe('emergent tool discovery', () => {
  test('trained policy improves wood/minute over random baseline', () => {
    const summary = trainPolicy(1337, 80);
    expect(summary.improvementPct).toBeGreaterThan(10);
  });
});
