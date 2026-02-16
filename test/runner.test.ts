import { describe, expect, test } from 'vitest';
import { runEpisode, runSweep, trainEpisodes } from '../src/runner/runner';

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
});
