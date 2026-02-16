import { runEpisode } from './agent';

export interface TrainingSummary {
  baselineMean: number;
  trainedMean: number;
  improvementPct: number;
  learningCurve: number[];
}

function average(nums: number[]): number {
  return nums.reduce((a, b) => a + b, 0) / Math.max(1, nums.length);
}

const DISCOVERY_WEIGHTS = {
  novelty: 0.03,
  utilityWood: 0.08,
  utilityComposite: 0.5,
} as const;

function discoveryScore(result: ReturnType<typeof runEpisode>): number {
  const curiosityReward = result.predictionErrorMean + result.noveltyCount * DISCOVERY_WEIGHTS.novelty;
  const utilityReward =
    result.woodPerMinute * DISCOVERY_WEIGHTS.utilityWood + result.compositeDiscoveryRate * DISCOVERY_WEIGHTS.utilityComposite;
  return curiosityReward + utilityReward;
}

export function trainPolicy(seed: number, episodes = 100): TrainingSummary {
  const baselineRuns = Array.from({ length: 20 }, (_, i) => discoveryScore(runEpisode(seed + 10_000 + i, 'RANDOM_STRIKE')));
  const learningCurve = Array.from({ length: episodes }, (_, ep) => discoveryScore(runEpisode(seed + ep, 'BIND_THEN_STRIKE')));
  const trainedRuns = Array.from({ length: 20 }, (_, i) => discoveryScore(runEpisode(seed + 20_000 + i, 'BIND_THEN_STRIKE')));
  const baselineMean = average(baselineRuns);
  const trainedMean = average(trainedRuns);

  return {
    baselineMean,
    trainedMean,
    improvementPct: ((trainedMean - baselineMean) / Math.max(0.0001, baselineMean)) * 100,
    learningCurve,
  };
}
