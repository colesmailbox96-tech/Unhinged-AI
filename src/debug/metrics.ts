import type { TrainingSummary } from '../ai/rl';

export class MetricsStore {
  training: TrainingSummary | null = null;
  lastEpisode = 0;
  woodPerMinute = 0;
  hardnessMae = 0;
  predictionError = 0;
  noveltyInteractions = 0;
  compositeRate = 0;
  embeddingClusters = 0;

  toHtml(): string {
    const training = this.training
      ? `baseline=${this.training.baselineMean.toFixed(2)} | trained=${this.training.trainedMean.toFixed(2)} | improve=${this.training.improvementPct.toFixed(1)}%`
      : 'training: not run';

    return [
      `episode: ${this.lastEpisode}`,
      `wood/min: ${this.woodPerMinute.toFixed(2)}`,
      `hardness MAE: ${this.hardnessMae.toFixed(3)}`,
      `prediction error: ${this.predictionError.toFixed(3)}`,
      `novel interactions: ${this.noveltyInteractions}`,
      `composite discovery rate: ${this.compositeRate.toFixed(2)}`,
      `tool embedding clusters: ${this.embeddingClusters}`,
      training,
    ].join('<br/>');
  }
}
