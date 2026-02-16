import { trainEpisodes } from './runner/runner';

const summary = trainEpisodes(1337, 100);
console.log(`Baseline wood/min: ${summary.baselineMean.toFixed(3)}`);
console.log(`Trained wood/min: ${summary.trainedMean.toFixed(3)}`);
console.log(`Improvement: ${summary.improvementPct.toFixed(2)}%`);
