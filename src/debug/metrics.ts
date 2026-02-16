import type { TrainingSummary } from '../ai/rl';
import type { TrainingState } from '../sim/trainingScheduler';

export class MetricsStore {
  training: TrainingSummary | null = null;
  trainingState: TrainingState = 'off';
  lastEpisode = 0;
  woodPerMinute = 0;
  hardnessMae = 0;
  predictionError = 0;
  noveltyInteractions = 0;
  compositeRate = 0;
  embeddingClusters = 0;
  freezePredictionError = 0;
  randomToolDiscovery = 0;
  agentToolDiscovery = 0;
  interventionConfidence = 1;
  replayDeterministic: boolean | null = null;

  toHtml(): string {
    const training = this.training
      ? `baseline=${this.training.baselineMean.toFixed(2)} | trained=${this.training.trainedMean.toFixed(2)} | improve=${this.training.improvementPct.toFixed(1)}%`
      : `training: ${this.trainingState}`;

    return [
      `episode: ${this.lastEpisode}`,
      `wood/min: ${this.woodPerMinute.toFixed(2)}`,
      `hardness MAE: ${this.hardnessMae.toFixed(3)}`,
      `prediction error: ${this.predictionError.toFixed(3)}`,
      `novel interactions: ${this.noveltyInteractions}`,
      `composite discovery rate: ${this.compositeRate.toFixed(2)}`,
      `tool embedding clusters: ${this.embeddingClusters}`,
      `freeze prediction error trend: ${this.freezePredictionError.toFixed(3)}`,
      `random vs agent tool discovery: ${this.randomToolDiscovery.toFixed(1)} vs ${this.agentToolDiscovery.toFixed(1)}`,
      `decision confidence: ${this.interventionConfidence.toFixed(2)}`,
      `deterministic replay: ${this.replayDeterministic === null ? 'not run' : this.replayDeterministic ? 'pass' : 'fail'}`,
      training,
    ].join('<br/>');
  }
}
