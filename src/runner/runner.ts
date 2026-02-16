import { runEpisode as runAgentEpisode, type EpisodeResult, type RunEpisodeOptions, type Strategy } from '../ai/agent';
import { trainPolicy, type TrainingSummary } from '../ai/rl';

export interface EpisodeMetrics {
  episode: number;
  totalEpisodes: number;
  seed: number;
  strategy: Strategy;
  elapsedMs: number;
  result: EpisodeResult;
  deterministicReplay?: boolean;
}

export interface StepMetrics {
  step: number;
  totalSteps: number;
  seed: number;
  episode: number;
}

export interface RunnerInterventions {
  freezeWorldModel?: boolean;
  disablePredictionModel?: boolean;
  randomAgent?: boolean;
  freezeEmbeddings?: boolean;
  determinismReplayCheck?: boolean;
}

export interface RunEpisodeConfig {
  seed: number;
  steps?: number;
  strategy?: Strategy;
  options?: RunEpisodeOptions;
  interventions?: RunnerInterventions;
  episode?: number;
  totalEpisodes?: number;
  onStep?: (metrics: StepMetrics) => void;
}

export interface RunSweepConfig {
  seed: number;
  seeds?: number[];
  episodes: number;
  stepsPerEpisode?: number;
  randomizeEachEpisode?: boolean;
  strategy?: Strategy;
  interventions?: RunnerInterventions;
  onEpisodeEnd?: (metrics: EpisodeMetrics) => void;
  onStep?: (metrics: StepMetrics) => void;
  shouldCancel?: () => boolean;
  waitWhilePaused?: () => Promise<void>;
  burstSteps?: number;
  onYield?: () => Promise<void>;
}

function strategyFor(interventions?: RunnerInterventions, strategy: Strategy = 'BIND_THEN_STRIKE'): Strategy {
  return interventions?.randomAgent ? 'RANDOM_STRIKE' : strategy;
}

function optionsFor(interventions?: RunnerInterventions): RunEpisodeOptions {
  return {
    freezeWorldModel: Boolean(interventions?.freezeWorldModel),
    disablePredictionModel: Boolean(interventions?.disablePredictionModel),
    collectTrace: Boolean(interventions?.determinismReplayCheck),
  };
}

export function runEpisode(config: RunEpisodeConfig): EpisodeMetrics {
  const episode = config.episode ?? 1;
  const totalEpisodes = config.totalEpisodes ?? 1;
  const steps = config.steps ?? 35;
  const strategy = strategyFor(config.interventions, config.strategy);
  const options = config.options ?? optionsFor(config.interventions);
  const started = performance.now();
  const result = runAgentEpisode(config.seed, strategy, undefined, steps, options);
  for (let step = 1; step <= steps; step++) {
    config.onStep?.({
      step,
      totalSteps: steps,
      seed: config.seed,
      episode,
    });
  }
  let deterministicReplay: boolean | undefined;
  if (config.interventions?.determinismReplayCheck) {
    const replay = runAgentEpisode(config.seed, strategy, undefined, steps, options);
    deterministicReplay = replay.replaySignature === result.replaySignature;
  }
  return {
    episode,
    totalEpisodes,
    seed: config.seed,
    strategy,
    elapsedMs: performance.now() - started,
    result,
    deterministicReplay,
  };
}

export function trainEpisodes(seed: number, episodes = 100): TrainingSummary {
  return trainPolicy(seed, episodes);
}

function defaultYield(): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, 0));
}

export async function runSweep(config: RunSweepConfig): Promise<EpisodeMetrics[]> {
  const seeds = config.seeds && config.seeds.length ? config.seeds : [config.seed];
  const results: EpisodeMetrics[] = [];
  const burstSteps = Math.max(1, config.burstSteps ?? 250);
  let bufferedSteps = 0;
  for (const baseSeed of seeds) {
    for (let episode = 0; episode < config.episodes; episode++) {
      if (config.shouldCancel?.()) return results;
      await config.waitWhilePaused?.();
      const seed = config.randomizeEachEpisode ? baseSeed + Math.floor(Math.random() * 1_000_000) : baseSeed + episode;
      const metrics = runEpisode({
        seed,
        steps: config.stepsPerEpisode,
        strategy: config.strategy,
        interventions: config.interventions,
        episode: episode + 1,
        totalEpisodes: config.episodes,
        onStep: config.onStep,
      });
      results.push(metrics);
      config.onEpisodeEnd?.(metrics);
      bufferedSteps += config.stepsPerEpisode ?? 35;
      if (bufferedSteps >= burstSteps) {
        bufferedSteps = 0;
        await (config.onYield?.() ?? defaultYield());
      }
    }
  }
  return results;
}
