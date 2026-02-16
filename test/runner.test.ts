import { describe, expect, test } from 'vitest';
import { runEpisode, runSweep, trainEpisodes } from '../src/runner/runner';
import { LiveModeEngine } from '../src/runner/live_mode';
import { ClosedLoopController } from '../src/ai/controller';
import { World } from '../src/sim/world';

describe('runner shared API', () => {
  test('runEpisode supports interventions and deterministic replay checks', () => {
    const metrics = runEpisode({
      seed: 2027,
      strategy: 'BIND_THEN_STRIKE',
      steps: 20,
      interventions: { disablePredictionModel: true, determinismReplayCheck: true },
    });
    expect(metrics.seed).toBe(2027);
    expect(metrics.result.trace.length).toBeGreaterThan(0);
    expect(metrics.deterministicReplay).toBe(true);
  });

  test('runSweep iterates seeds and episodes', async () => {
    const calls: number[] = [];
    const results = await runSweep({
      seed: 11,
      seeds: [11, 12],
      episodes: 2,
      stepsPerEpisode: 5,
      onEpisodeEnd: (metrics) => {
        calls.push(metrics.seed);
      },
    });
    expect(results).toHaveLength(4);
    expect(calls).toHaveLength(4);
  });

  test('trainEpisodes proxies policy training summary', () => {
    const summary = trainEpisodes(1337, 10);
    expect(summary.learningCurve).toHaveLength(10);
  });

  test('live mode loop runs for many ticks and logs milestones', () => {
    const engine = new LiveModeEngine({
      seed: 1337,
      populationSize: 3,
      ticksPerSecond: 20,
      deterministic: true,
      rollingSeconds: 30,
    });
    for (let i = 0; i < 600; i++) {
      engine.tickOnce();
      if (i % 8 === 0) {
        engine.trainChunk({
          trainEveryMs: 50,
          batchSize: 6,
          maxTrainMsPerSecond: 50,
          stepsPerTick: 2,
        });
      }
    }
    expect(engine.tick).toBe(600);
    expect(engine.milestones.all().length).toBeGreaterThan(0);
  });

  test('controller improves planarity over baseline across trials', () => {
    const runTrial = (seed: number, useController: boolean): number => {
      const world = new World(seed);
      const nearby = world.getNearbyObjectIds();
      const heldId = nearby[0];
      const abrasiveId = nearby.find((id) => id !== heldId);
      if (!heldId || !abrasiveId) return 0;
      world.apply({ type: 'PICK_UP', objId: heldId });
      if (useController) {
        const controller = new ClosedLoopController();
        for (let i = 0; i < 16; i++) {
          const held = world.objects.get(heldId);
          if (!held) break;
          controller.step(world, held, { metric: 'surface_planarity', target: 0.82 });
        }
      } else {
        for (let i = 0; i < 16; i++) {
          const roll = world.rng.float();
          if (roll < 0.25) world.apply({ type: 'HEAT', intensity: world.rng.range(0.2, 0.9) });
          else if (roll < 0.5) world.apply({ type: 'SOAK', intensity: world.rng.range(0.2, 0.9) });
          else if (roll < 0.75) world.apply({ type: 'COOL', intensity: world.rng.range(0.2, 0.9) });
          else world.apply({ type: 'GRIND', abrasiveId, intensity: world.rng.range(0.2, 0.9) });
        }
      }
      return world.objects.get(heldId)?.latentPrecision.surface_planarity ?? 0;
    };

    const seeds = [301, 302, 303, 304, 305];
    const avg = (values: number[]): number => values.reduce((sum, value) => sum + value, 0) / Math.max(1, values.length);
    const baseline = avg(seeds.map((seed) => runTrial(seed, false)));
    const controlled = avg(seeds.map((seed) => runTrial(seed, true)));
    expect(controlled).toBeGreaterThan(baseline);
  });
});
