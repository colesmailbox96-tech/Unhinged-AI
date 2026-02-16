import { runEpisode, type EmbeddingSnapshot, type Strategy } from './ai/agent';
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
const realityCheckBtn = document.querySelector<HTMLButtonElement>('#realityCheckBtn')!;
const freezeModelBtn = document.querySelector<HTMLButtonElement>('#freezeModelBtn')!;
const randomSwapBtn = document.querySelector<HTMLButtonElement>('#randomSwapBtn')!;
const recordEpisodeBtn = document.querySelector<HTMLButtonElement>('#recordEpisodeBtn')!;
const replayEpisodeBtn = document.querySelector<HTMLButtonElement>('#replayEpisodeBtn')!;
const disablePredictionBtn = document.querySelector<HTMLButtonElement>('#disablePredictionBtn')!;
const realityCheckStatsEl = document.querySelector<HTMLElement>('#realityCheckStats')!;
const embeddingVizEl = document.querySelector<HTMLElement>('#embeddingViz')!;

let seed = 1337;
let world = new World(seed);
const perception = new PerceptionHead(seed + 99);
const metrics = new MetricsStore();
let paused = false;
let realityCheckEnabled = false;
let freezeWorldModel = false;
let disablePredictionModel = false;
let lastEmbeddingSnapshot: EmbeddingSnapshot[] = [];
let recordedEpisode:
  | {
      seed: number;
      strategy: Strategy;
      replaySignature: string;
      actions: string[];
      outcomes: string[];
    }
  | undefined;
const predictionErrorTrend: number[] = [];
const freezeErrorTrend: number[] = [];
const novelInteractionTrend: number[] = [];
const compositeRateTrend: number[] = [];
let randomComparison:
  | {
      random: { toolDiscovery: number; compositeRate: number; predictionAccuracy: number };
      agent: { toolDiscovery: number; compositeRate: number; predictionAccuracy: number };
    }
  | undefined;

let view = new CanvasView(canvas, world, perception);

function refresh(): void {
  metricsEl.innerHTML = metrics.toHtml();
  realityCheckStatsEl.innerHTML = [
    `mode: ${realityCheckEnabled ? 'on' : 'off'}`,
    `freeze world model: ${freezeWorldModel ? 'on' : 'off'}`,
    `disable prediction model: ${disablePredictionModel ? 'on' : 'off'}`,
    `prediction error trend: ${predictionErrorTrend.slice(-6).map((v) => v.toFixed(3)).join(' → ') || 'n/a'}`,
    `novel interactions trend: ${novelInteractionTrend.slice(-6).join(' → ') || 'n/a'}`,
    `composite discovery trend: ${compositeRateTrend.slice(-6).map((v) => v.toFixed(2)).join(' → ') || 'n/a'}`,
    randomComparison
      ? `random(agent same seed): discovery ${randomComparison.random.toolDiscovery} vs ${randomComparison.agent.toolDiscovery} | composite ${randomComparison.random.compositeRate.toFixed(2)} vs ${randomComparison.agent.compositeRate.toFixed(2)} | pred acc ${randomComparison.random.predictionAccuracy.toFixed(2)} vs ${randomComparison.agent.predictionAccuracy.toFixed(2)}`
      : 'random-policy swap: not run',
    recordedEpisode ? `recorded: seed=${recordedEpisode.seed} strategy=${recordedEpisode.strategy}` : 'recorded: none',
  ].join('<br/>');
  embeddingVizEl.innerHTML = renderEmbedding(lastEmbeddingSnapshot);
  view.render(selectedEl, logEl);
}

function runSingle(): void {
  const mode = metrics.lastEpisode % 2 === 0 ? 'RANDOM_STRIKE' : 'BIND_THEN_STRIKE';
  const result = runEpisode(seed + metrics.lastEpisode, mode, perception, 35, {
    freezeWorldModel: realityCheckEnabled && freezeWorldModel,
    disablePredictionModel: realityCheckEnabled && disablePredictionModel,
    collectTrace: realityCheckEnabled,
  });
  metrics.lastEpisode += 1;
  metrics.woodPerMinute = result.woodPerMinute;
  metrics.hardnessMae = result.hardnessMaeAfter;
  metrics.predictionError = result.predictionErrorMean;
  metrics.noveltyInteractions = result.novelInteractionCount;
  metrics.compositeRate = result.compositeDiscoveryRate;
  metrics.embeddingClusters = result.embeddingClusters;
  metrics.interventionConfidence = disablePredictionModel ? 0 : 1;
  predictionErrorTrend.push(result.predictionErrorMean);
  if (freezeWorldModel) freezeErrorTrend.push(result.predictionErrorMean);
  metrics.freezePredictionError =
    freezeErrorTrend.reduce((acc, value) => acc + value, 0) / Math.max(1, freezeErrorTrend.length);
  novelInteractionTrend.push(result.novelInteractionCount);
  compositeRateTrend.push(result.compositeDiscoveryRate);
  lastEmbeddingSnapshot = result.embeddingSnapshot;
  world.predictedStrikeArc = result.predictedStrikeArc ? { ...result.predictedStrikeArc, alpha: 0.45 } : undefined;
  world.lastStrikeArc = result.actualStrikeArc ? { ...result.actualStrikeArc, alpha: 0.65 } : undefined;
  world.predictionRealityOverlay = result.predictionSnapshot
    ? {
        predicted: {
          damage: result.predictionSnapshot.predicted.expected_damage,
          toolWear: result.predictionSnapshot.predicted.expected_tool_wear,
          fragments: result.predictionSnapshot.predicted.expected_fragments,
        },
        actual: result.predictionSnapshot.actual,
        error: result.predictionSnapshot.error,
      }
    : undefined;
  world.logs.unshift(...result.logs.slice(0, 6));
  refresh();
}

function renderEmbedding(points: EmbeddingSnapshot[]): string {
  if (!realityCheckEnabled || points.length === 0) return '';
  const xs = points.map((point) => point.point.x);
  const ys = points.map((point) => point.point.y);
  const minX = Math.min(...xs);
  const maxX = Math.max(...xs);
  const minY = Math.min(...ys);
  const maxY = Math.max(...ys);
  const width = 280;
  const height = 180;
  const rangeX = Math.max(1e-9, maxX - minX);
  const rangeY = Math.max(1e-9, maxY - minY);
  const toX = (v: number): number => 10 + ((v - minX) / rangeX) * (width - 20);
  const toY = (v: number): number => 10 + ((v - minY) / rangeY) * (height - 20);
  const circles = points
    .map((point) => {
      const x = toX(point.point.x);
      const y = toY(point.point.y);
      const color = `hsl(${Math.min(300, point.vector[0] * 210 + 80)}, 70%, ${Math.min(70, 30 + point.vector[1] * 25)}%)`;
      const vectorLabel = point.vector.map((v) => v.toFixed(3)).join(', ');
      return `<circle cx="${x.toFixed(1)}" cy="${y.toFixed(1)}" r="4" fill="${color}"><title>tool ${point.toolId} | vector [${vectorLabel}]</title></circle>`;
    })
    .join('');
  const links = points
    .flatMap((left, i) =>
      points.slice(i + 1).map((right) => {
        const similarity =
          left.vector[0] * right.vector[0] +
          left.vector[1] * right.vector[1] +
          left.vector[2] * right.vector[2] +
          left.vector[3] * right.vector[3];
        if (similarity < 0.9) return '';
        return `<line x1="${toX(left.point.x).toFixed(1)}" y1="${toY(left.point.y).toFixed(1)}" x2="${toX(right.point.x).toFixed(1)}" y2="${toY(right.point.y).toFixed(1)}" stroke="#8ad2ff" stroke-opacity="${Math.min(0.7, similarity).toFixed(2)}" />`;
      }),
    )
    .join('');
  const svg = `<svg width="${width}" height="${height}" viewBox="0 0 ${width} ${height}" style="background:#0d1420;border:1px solid #273248;border-radius:6px">${links}${circles}</svg>`;
  return `Embedding PCA-like view<br/>${svg}`;
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
realityCheckBtn.onclick = () => {
  realityCheckEnabled = !realityCheckEnabled;
  realityCheckBtn.textContent = realityCheckEnabled ? 'Disable' : 'Enable';
  refresh();
};
freezeModelBtn.onclick = () => {
  freezeWorldModel = !freezeWorldModel;
  freezeModelBtn.textContent = freezeWorldModel ? 'Unfreeze World Model' : 'Freeze World Model';
  refresh();
};
disablePredictionBtn.onclick = () => {
  disablePredictionModel = !disablePredictionModel;
  disablePredictionBtn.textContent = disablePredictionModel ? 'Enable Prediction Model' : 'Disable Prediction Model';
  refresh();
};
randomSwapBtn.onclick = () => {
  const compareSeed = seed + metrics.lastEpisode;
  const randomResult = runEpisode(compareSeed, 'RANDOM_STRIKE', new PerceptionHead(compareSeed + 99), 35, {
    freezeWorldModel: freezeWorldModel,
    disablePredictionModel: false,
    collectTrace: true,
  });
  const agentResult = runEpisode(compareSeed, 'BIND_THEN_STRIKE', new PerceptionHead(compareSeed + 99), 35, {
    freezeWorldModel: freezeWorldModel,
    disablePredictionModel: disablePredictionModel,
    collectTrace: true,
  });
  randomComparison = {
    random: {
      toolDiscovery: randomResult.novelInteractionCount,
      compositeRate: randomResult.compositeDiscoveryRate,
      predictionAccuracy: 1 / (1 + randomResult.predictionErrorMean),
    },
    agent: {
      toolDiscovery: agentResult.novelInteractionCount,
      compositeRate: agentResult.compositeDiscoveryRate,
      predictionAccuracy: 1 / (1 + agentResult.predictionErrorMean),
    },
  };
  metrics.randomToolDiscovery = randomComparison.random.toolDiscovery;
  metrics.agentToolDiscovery = randomComparison.agent.toolDiscovery;
  refresh();
};
recordEpisodeBtn.onclick = () => {
  const recordSeed = seed + metrics.lastEpisode;
  const strategy: Strategy = 'BIND_THEN_STRIKE';
  const result = runEpisode(recordSeed, strategy, new PerceptionHead(recordSeed + 99), 35, {
    freezeWorldModel: freezeWorldModel,
    disablePredictionModel: disablePredictionModel,
    collectTrace: true,
  });
  recordedEpisode = {
    seed: recordSeed,
    strategy,
    replaySignature: result.replaySignature,
    actions: result.trace.filter((entry) => entry.includes('action=')),
    outcomes: result.trace.filter((entry) => entry.includes('outcome=')),
  };
  world.logs.unshift(`Recorded episode seed=${recordSeed} actions=${recordedEpisode.actions.length}`);
  refresh();
};
replayEpisodeBtn.onclick = () => {
  if (!recordedEpisode) {
    world.logs.unshift('Replay requested with no recorded episode');
    refresh();
    return;
  }
  const replay = runEpisode(recordedEpisode.seed, recordedEpisode.strategy, new PerceptionHead(recordedEpisode.seed + 99), 35, {
    freezeWorldModel: freezeWorldModel,
    disablePredictionModel: disablePredictionModel,
    collectTrace: true,
  });
  const deterministic =
    replay.replaySignature === recordedEpisode.replaySignature &&
    replay.trace.filter((entry) => entry.includes('action=')).join('|') === recordedEpisode.actions.join('|') &&
    replay.trace.filter((entry) => entry.includes('outcome=')).join('|') === recordedEpisode.outcomes.join('|');
  metrics.replayDeterministic = deterministic;
  world.logs.unshift(`Replay ${deterministic ? 'matched' : 'mismatch'} seed=${recordedEpisode.seed}`);
  refresh();
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
