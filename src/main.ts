import { runEpisode } from './ai/agent';
import { PerceptionHead } from './ai/perception';
import { trainPolicy } from './ai/rl';
import { MetricsStore } from './debug/metrics';
import { CanvasView } from './ui/canvas_view';
import { World } from './sim/world';

const canvas = document.querySelector<HTMLCanvasElement>('#worldCanvas')!;
const metricsEl = document.querySelector<HTMLElement>('#metrics')!;
const selectedEl = document.querySelector<HTMLElement>('#selected')!;
const logEl = document.querySelector<HTMLElement>('#log')!;
const resetBtn = document.querySelector<HTMLButtonElement>('#resetBtn')!;
const trainBtn = document.querySelector<HTMLButtonElement>('#trainBtn')!;
const runBtn = document.querySelector<HTMLButtonElement>('#runBtn')!;
const pauseBtn = document.querySelector<HTMLButtonElement>('#pauseBtn')!;
const stepBtn = document.querySelector<HTMLButtonElement>('#stepBtn')!;

let seed = 1337;
let world = new World(seed);
const perception = new PerceptionHead(seed + 99);
const metrics = new MetricsStore();
let paused = false;

let view = new CanvasView(canvas, world, perception);

function refresh(): void {
  metricsEl.innerHTML = metrics.toHtml();
  view.render(selectedEl, logEl);
}

function runSingle(): void {
  const mode = metrics.lastEpisode % 2 === 0 ? 'RANDOM_STRIKE' : 'BIND_THEN_STRIKE';
  const result = runEpisode(seed + metrics.lastEpisode, mode, perception);
  metrics.lastEpisode += 1;
  metrics.woodPerMinute = result.woodPerMinute;
  metrics.hardnessMae = result.hardnessMaeAfter;
  metrics.predictionError = result.predictionErrorMean;
  metrics.noveltyInteractions = result.noveltyCount;
  metrics.compositeRate = result.compositeDiscoveryRate;
  metrics.embeddingClusters = result.embeddingClusters;
  world.logs.unshift(...result.logs.slice(0, 6));
  refresh();
}

resetBtn.onclick = () => {
  seed += 1;
  world = new World(seed);
  view = new CanvasView(canvas, world, perception);
  world.logs.unshift(`Reset with seed ${seed}`);
  refresh();
};

trainBtn.onclick = () => {
  metrics.training = trainPolicy(seed, 100);
  world.logs.unshift(
    `Training done improve=${metrics.training.improvementPct.toFixed(1)}% baseline=${metrics.training.baselineMean.toFixed(2)} trained=${metrics.training.trainedMean.toFixed(2)}`,
  );
  refresh();
};

runBtn.onclick = runSingle;
pauseBtn.onclick = () => {
  paused = !paused;
  pauseBtn.textContent = paused ? 'Resume' : 'Pause';
};
stepBtn.onclick = () => {
  runSingle();
};

let accumulator = 0;
let last = performance.now();
const fixedMs = 100;

function loop(now: number): void {
  const dt = now - last;
  last = now;
  accumulator += dt;

  while (!paused && accumulator >= fixedMs) {
    accumulator -= fixedMs;
    // deterministic fixed-step demo loop
    if (metrics.lastEpisode < 1) runSingle();
  }

  refresh();
  requestAnimationFrame(loop);
}

requestAnimationFrame(loop);
