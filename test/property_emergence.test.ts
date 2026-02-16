import { describe, expect, test } from 'vitest';
import { readFileSync, readdirSync, statSync } from 'node:fs';
import { join } from 'node:path';
import { bindObjects, strike, grind } from '../src/sim/interactions';
import { RNG } from '../src/sim/rng';
import { PropertyVector } from '../src/sim/properties';
import { WorldObject } from '../src/sim/object_model';
import { PerceptionHead } from '../src/ai/perception';
import { trainPolicy } from '../src/ai/rl';
import { runEpisode } from '../src/ai/agent';
import { WorldModel } from '../src/ai/world_model';
import { ToolEmbedding } from '../src/ai/tool_embedding';

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
    shapeType: 'rod',
    radius: 0.3,
    length: 1.2,
    thickness: 0.24,
    center_of_mass_offset: { x: 0.06, y: 0 },
    grip_score: 0.6,
    props,
    integrity: 1,
  };
}

function createIdGenerator(startId: number): () => number {
  let id = startId;
  return () => id++;
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
  test('longer tools accumulate more angular strike damage', () => {
    const rng = new RNG(10);
    const shortTool = { ...obj(1, baseProps({ hardness: 0.85, sharpness: 0.75, mass: 0.8 })), length: 0.5, center_of_mass_offset: { x: 0.04, y: 0 } };
    const longTool = { ...obj(2, baseProps({ hardness: 0.85, sharpness: 0.75, mass: 0.8 })), length: 2.1, center_of_mass_offset: { x: 0.2, y: 0 } };
    const target = { ...obj(3, baseProps({ hardness: 0.6 })), shapeType: 'plate' as const };

    const shortDamage = strike({ ...shortTool }, { ...target }, rng, createIdGenerator(100)).damage;
    const longDamage = strike({ ...longTool }, { ...target }, rng, createIdGenerator(200)).damage;

    expect(longDamage).toBeGreaterThan(shortDamage);
    expect(longDamage - shortDamage).toBeGreaterThan(0.02);
  });

  test('binding heavy tip increases composite strike output via lever arm', () => {
    const rng = new RNG(44);
    const handle = { ...obj(1, baseProps({ mass: 0.35, roughness: 0.8 })), length: 2.0, thickness: 0.16, center_of_mass_offset: { x: 0.03, y: 0 } };
    const heavyHead = { ...obj(2, baseProps({ mass: 0.95, hardness: 0.9 })), shapeType: 'sphere' as const, length: 0.45, thickness: 0.45, center_of_mass_offset: { x: 0.28, y: 0 } };
    const lightHead = { ...obj(3, baseProps({ mass: 0.2, hardness: 0.45 })), shapeType: 'sphere' as const, length: 0.45, thickness: 0.45, center_of_mass_offset: { x: 0.02, y: 0 } };
    const target = { ...obj(9, baseProps({ hardness: 0.6 })), shapeType: 'plate' as const };

    const heavyComposite = bindObjects(handle, heavyHead, 100, rng).composite;
    const lightComposite = bindObjects(handle, lightHead, 101, rng).composite;

    const heavyDamage = strike({ ...heavyComposite }, { ...target }, rng, createIdGenerator(300)).damage;
    const lightDamage = strike({ ...lightComposite }, { ...target }, rng, createIdGenerator(400)).damage;

    expect(heavyComposite.center_of_mass_offset.x).toBeGreaterThan(lightComposite.center_of_mass_offset.x);
    expect(heavyDamage).toBeGreaterThan(lightDamage);
  });

  test('grind wear increases with brittleness', () => {
    const abrasive = obj(9, baseProps({ roughness: 0.9, friction_coeff: 0.9 }));
    const lowBrittle = grind(obj(1, baseProps({ brittleness: 0.1 })), abrasive).wear;
    const highBrittle = grind(obj(2, baseProps({ brittleness: 0.9 })), abrasive).wear;
    expect(highBrittle).toBeGreaterThan(lowBrittle);
  });
});

describe('perception learning', () => {
  test('perception observes geometric proxy cues', () => {
    const rng = new RNG(321);
    const head = new PerceptionHead(654);
    const symmetric = { ...obj(1, baseProps({ mass: 0.35 })), shapeType: 'rod' as const, center_of_mass_offset: { x: 0, y: 0 } };
    const weightedTip = { ...obj(2, baseProps({ mass: 0.8 })), shapeType: 'rod' as const, length: 2.0, center_of_mass_offset: { x: 0.35, y: 0 } };
    const symObs = head.observe(symmetric, rng);
    const tipObs = head.observe(weightedTip, rng);

    expect(symObs.observed_length).toBeGreaterThan(0);
    expect(symObs.observed_mass_estimate).toBeGreaterThan(0);
    expect(symObs.contact_area_estimate).toBeGreaterThan(0);
    expect(tipObs.visual_symmetry).toBeLessThan(symObs.visual_symmetry);
  });

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
  test('curiosity policy improves discovery score over random baseline', () => {
    const summary = trainPolicy(1337, 80);
    expect(summary.improvementPct).toBeGreaterThan(10);
  });
});

describe('phase 2 predictive scientist modules', () => {
  test('predictive scientist episodes emit discovery metrics', () => {
    const random = runEpisode(2026, 'RANDOM_STRIKE');
    const predictive = runEpisode(2026, 'BIND_THEN_STRIKE');
    expect(random.predictionErrorMean).toBe(0);
    expect(predictive.predictionErrorMean).toBeGreaterThan(0);
    expect(predictive.embeddingClusters).toBeGreaterThanOrEqual(1);
  });

  test('world model reduces prediction error with updates', () => {
    const model = new WorldModel();
    const sample = {
      action_verb: 'STRIKE_WITH' as const,
      objectA: { visual_features: 0.6, mass_estimate: 0.75, length_estimate: 0.9, texture_proxy: 0.5, interaction_feedback_history: 0.1 },
      objectB: { visual_features: 0.4, mass_estimate: 0.5, length_estimate: 0.7, texture_proxy: 0.45, interaction_feedback_history: 0.1 },
      geometry_features: 0.62,
      relative_position: 0.3,
    };
    const truth = { expected_damage: 0.48, expected_tool_wear: 0.07, expected_fragments: 1.2, expected_property_changes: 0.18 };
    const before = model.update(sample, truth);
    let after = before;
    for (let i = 0; i < 8; i++) after = model.update(sample, truth);
    expect(after).toBeLessThan(before);
  });

  test('tool embeddings form effect clusters by similarity', () => {
    const embedding = new ToolEmbedding();
    embedding.update(1, { damage: 0.9, toolWear: 0.1, fragments: 1.2, propertyChanges: 0.2 });
    embedding.update(2, { damage: 0.85, toolWear: 0.12, fragments: 1.1, propertyChanges: 0.22 });
    embedding.update(3, { damage: 0.05, toolWear: 0.7, fragments: 0, propertyChanges: 0.05 });
    expect(embedding.similarity(1, 2)).toBeGreaterThan(embedding.similarity(1, 3));
    expect(embedding.clusterCount()).toBeGreaterThanOrEqual(2);
    expect(embedding.entries().length).toBe(3);
  });

  test('world model freeze blocks learning updates', () => {
    const model = new WorldModel();
    const sample = {
      action_verb: 'STRIKE_WITH' as const,
      objectA: { visual_features: 0.5, mass_estimate: 0.7, length_estimate: 0.8, texture_proxy: 0.4, interaction_feedback_history: 0.2 },
      objectB: { visual_features: 0.4, mass_estimate: 0.45, length_estimate: 0.6, texture_proxy: 0.3, interaction_feedback_history: 0.2 },
      geometry_features: 0.61,
      relative_position: 0.3,
    };
    const truth = { expected_damage: 0.65, expected_tool_wear: 0.09, expected_fragments: 1.5, expected_property_changes: 0.2 };
    model.setFrozen(true);
    const before = model.update(sample, truth);
    let after = before;
    for (let i = 0; i < 6; i++) after = model.update(sample, truth);
    expect(after).toBeCloseTo(before, 8);
  });

  test('episode trace is deterministic and intervention changes behavior', () => {
    const first = runEpisode(4242, 'BIND_THEN_STRIKE', undefined, 35, { collectTrace: true });
    const second = runEpisode(4242, 'BIND_THEN_STRIKE', undefined, 35, { collectTrace: true });
    const intervention = runEpisode(4242, 'BIND_THEN_STRIKE', undefined, 35, { collectTrace: true, disablePredictionModel: true });
    expect(first.replaySignature).toBe(second.replaySignature);
    expect(first.replaySignature).not.toBe(intervention.replaySignature);
  });

  test('episode returns prediction snapshot and embedding projection for reality mode', () => {
    const result = runEpisode(5151, 'BIND_THEN_STRIKE', undefined, 35, { collectTrace: true });
    expect(result.predictionSnapshot).toBeDefined();
    expect(result.embeddingSnapshot.length).toBeGreaterThan(0);
    expect(result.embeddingSnapshot[0]).toHaveProperty('point');
  });
});
