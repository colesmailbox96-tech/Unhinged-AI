import { describe, expect, test } from 'vitest';
import { runEpisode, runSweep, trainEpisodes } from '../src/runner/runner';
import { LiveModeEngine } from '../src/runner/live_mode';

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
});
