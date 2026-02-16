import { runEpisode } from './agent';
import type { Strategy } from './agent';
import { ReplayBuffer } from './replay_buffer';
import { RNG } from '../sim/rng';

export interface TrainingSummary {
  baselineMean: number;
  trainedMean: number;
  improvementPct: number;
  learningCurve: number[];
}

class Bandit {
  private readonly values: Record<Strategy, number> = {
    RANDOM_STRIKE: 0,
    BIND_THEN_STRIKE: 0,
  };
  private readonly counts: Record<Strategy, number> = {
    RANDOM_STRIKE: 0,
    BIND_THEN_STRIKE: 0,
  };

  select(rng: RNG, epsilon: number): Strategy {
    if (rng.float() < epsilon) return rng.float() < 0.5 ? 'RANDOM_STRIKE' : 'BIND_THEN_STRIKE';
    return this.values.BIND_THEN_STRIKE >= this.values.RANDOM_STRIKE ? 'BIND_THEN_STRIKE' : 'RANDOM_STRIKE';
  }

  update(action: Strategy, reward: number): void {
    this.counts[action] += 1;
    this.values[action] += (reward - this.values[action]) / this.counts[action];
  }

  best(): Strategy {
    return this.values.BIND_THEN_STRIKE >= this.values.RANDOM_STRIKE ? 'BIND_THEN_STRIKE' : 'RANDOM_STRIKE';
  }
}

function average(nums: number[]): number {
  return nums.reduce((a, b) => a + b, 0) / Math.max(1, nums.length);
}

export function trainPolicy(seed: number, episodes = 100): TrainingSummary {
  const rng = new RNG(seed);
  const bandit = new Bandit();
  const replay = new ReplayBuffer<{ episode: number }, Strategy>(256);
  const learningCurve: number[] = [];

  const baselineRuns = Array.from({ length: 20 }, (_, i) => runEpisode(seed + 10_000 + i, 'RANDOM_STRIKE').woodPerMinute);

  for (let ep = 0; ep < episodes; ep++) {
    const action = bandit.select(rng, Math.max(0.05, 0.5 - ep / episodes));
    const result = runEpisode(seed + ep, action);
    const reward = result.woodPerMinute;
    replay.push({ state: { episode: ep }, action, reward });
    bandit.update(action, reward);

    const recent = replay.sampleLast(8).map((s) => s.reward);
    learningCurve.push(average(recent));
  }

  const best = bandit.best();
  const trainedRuns = Array.from({ length: 20 }, (_, i) => runEpisode(seed + 20_000 + i, best).woodPerMinute);

  const baselineMean = average(baselineRuns);
  const trainedMean = average(trainedRuns);
  return {
    baselineMean,
    trainedMean,
    improvementPct: ((trainedMean - baselineMean) / Math.max(0.0001, baselineMean)) * 100,
    learningCurve,
  };
}
