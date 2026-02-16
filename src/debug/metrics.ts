import type { TrainingSummary } from '../ai/rl';

export class MetricsStore {
  training: TrainingSummary | null = null;
  lastEpisode = 0;
  woodPerMinute = 0;
  hardnessMae = 0;

  toHtml(): string {
    const training = this.training
      ? `baseline=${this.training.baselineMean.toFixed(2)} | trained=${this.training.trainedMean.toFixed(2)} | improve=${this.training.improvementPct.toFixed(1)}%`
      : 'training: not run';

    return [
      `episode: ${this.lastEpisode}`,
      `wood/min: ${this.woodPerMinute.toFixed(2)}`,
      `hardness MAE: ${this.hardnessMae.toFixed(3)}`,
      training,
    ].join('<br/>');
  }
}
