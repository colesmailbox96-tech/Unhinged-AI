import { type EmbeddingSnapshot, type Strategy } from './ai/agent';
import { PerceptionHead } from './ai/perception';
import { MetricsStore } from './debug/metrics';
import { LiveModeEngine, type LiveTrainingConfig, type LiveTickResult } from './runner/live_mode';
import { runEpisode as runRunnerEpisode, runSweep, trainEpisodes, type EpisodeMetrics, type RunnerInterventions } from './runner/runner';
import { CanvasView } from './ui/canvas_view';
import { AutopilotPanel, type AutopilotConfig, type ExportFormat } from './ui/autopilot_panel';
import { LiveModePanel, type LiveModePanelConfig } from './ui/live_mode_panel';
import { World } from './sim/world';

const canvas = document.querySelector<HTMLCanvasElement>('#worldCanvas')!;
const panelEl = document.querySelector<HTMLElement>('#panel')!;
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
const autopilotHistory: EpisodeMetrics[] = [];
let autopilotRunning = false;
let autopilotPaused = false;
let autopilotCancelRequested = false;
let autopilotStartedAt = 0;
let liveEngine: LiveModeEngine | null = null;
let liveConfig: LiveModePanelConfig | null = null;
let liveSimulationTimer: number | undefined;
let liveTrainingTimer: number | undefined;
let livePaused = false;
let liveFastForward = false;
let liveLastRenderedTick = 0;
const liveTimeline: import('./sim/milestones').MilestoneEvent[] = [];
const liveMetricHistory: LiveTickResult[] = [];

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

function currentInterventions(): RunnerInterventions {
  return {
    freezeWorldModel: realityCheckEnabled && freezeWorldModel,
    disablePredictionModel: realityCheckEnabled && disablePredictionModel,
  };
}

function applyEpisodeMetrics(episode: EpisodeMetrics): void {
  const result = episode.result;
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
  if (episode.deterministicReplay !== undefined) metrics.replayDeterministic = episode.deterministicReplay;
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

function runSingle(): void {
  const mode = metrics.lastEpisode % 2 === 0 ? 'RANDOM_STRIKE' : 'BIND_THEN_STRIKE';
  const episode = runRunnerEpisode({
    seed: seed + metrics.lastEpisode,
    strategy: mode,
    steps: 35,
    interventions: currentInterventions(),
  });
  applyEpisodeMetrics(episode);
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

function toCsv(rows: EpisodeMetrics[]): string {
  const head = [
    'episode',
    'totalEpisodes',
    'seed',
    'strategy',
    'woodPerMinute',
    'predictionError',
    'compositeDiscoveryRate',
    'embeddingClusters',
    'novelInteractions',
    'deterministicReplay',
    'elapsedMs',
  ];
  const body = rows.map((row) => [
    row.episode,
    row.totalEpisodes,
    row.seed,
    row.strategy,
    row.result.woodPerMinute.toFixed(4),
    row.result.predictionErrorMean.toFixed(6),
    row.result.compositeDiscoveryRate.toFixed(6),
    row.result.embeddingClusters,
    row.result.novelInteractionCount,
    row.deterministicReplay ?? '',
    row.elapsedMs.toFixed(3),
  ]);
  return [head.join(','), ...body.map((line) => line.join(','))].join('\n');
}

function downloadText(filename: string, content: string, contentType: string): void {
  const blob = new Blob([content], { type: contentType });
  const href = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = href;
  a.download = filename;
  a.click();
  URL.revokeObjectURL(href);
}

function exportHistory(format: ExportFormat): void {
  if (!autopilotHistory.length) return;
  if (format === 'json') {
    downloadText('metrics.json', JSON.stringify(autopilotHistory, null, 2), 'application/json');
    return;
  }
  downloadText('metrics.csv', toCsv(autopilotHistory), 'text/csv');
}

function clearLiveTimers(): void {
  if (liveSimulationTimer !== undefined) window.clearInterval(liveSimulationTimer);
  if (liveTrainingTimer !== undefined) window.clearInterval(liveTrainingTimer);
  liveSimulationTimer = undefined;
  liveTrainingTimer = undefined;
}

function applyLiveMetrics(result: LiveTickResult): void {
  metrics.lastEpisode = result.tick;
  metrics.woodPerMinute = result.woodPerMinute;
  metrics.predictionError = result.predictionErrorMean;
  metrics.noveltyInteractions = Number(result.novelInteractionsPerMinute.toFixed(2));
  metrics.compositeRate = result.compositeDiscoveryRate;
  metrics.embeddingClusters = result.embeddingClusters;
}

function exportLiveSnapshot(): void {
  if (!liveEngine) return;
  downloadText('live-snapshot.json', JSON.stringify(liveEngine.createSnapshot(), null, 2), 'application/json');
}

function exportMilestones(): void {
  if (!liveTimeline.length) return;
  downloadText('milestones.json', JSON.stringify(liveTimeline, null, 2), 'application/json');
}

function replayBookmark(bookmarkId: string): void {
  if (!liveEngine) return;
  const frames = liveEngine.replayBookmark(bookmarkId);
  if (!frames.length) return;
  world.logs.unshift(`Replay ${bookmarkId} (${frames.length} frames, deterministic playback)`);
  for (const frame of frames.slice(-6).reverse()) {
    world.logs.unshift(`t=${frame.simTimeSeconds.toFixed(2)} action=${frame.action} sig=${frame.replaySignature}`);
  }
  refresh();
}

function maybeRenderLive(result: LiveTickResult, livePanel: LiveModePanel): void {
  if (!liveEngine || !liveConfig) return;
  liveMetricHistory.push(result);
  while (liveMetricHistory.length && result.simTimeSeconds - liveMetricHistory[0].simTimeSeconds > 300) liveMetricHistory.shift();
  applyLiveMetrics(result);
  const shouldRender = liveFastForward ? result.tick - liveLastRenderedTick >= liveConfig.renderEveryNTicks : true;
  if (!shouldRender) return;
  liveLastRenderedTick = result.tick;
  world = liveEngine.world;
  view = new CanvasView(canvas, world, perception);
  const windowAverages = (seconds: number): { wood: number; pred: number; novelty: number; composite: number; clusters: number } => {
    const windowRows = liveMetricHistory.filter((entry) => result.simTimeSeconds - entry.simTimeSeconds <= seconds);
    const avg = (pick: (entry: LiveTickResult) => number): number =>
      windowRows.reduce((sum, entry) => sum + pick(entry), 0) / Math.max(1, windowRows.length);
    return {
      wood: avg((entry) => entry.woodPerMinute),
      pred: avg((entry) => entry.predictionErrorMean),
      novelty: avg((entry) => entry.novelInteractionsPerMinute),
      composite: avg((entry) => entry.compositeDiscoveryRate),
      clusters: avg((entry) => entry.embeddingClusters),
    };
  };
  const recent60 = windowAverages(60);
  const recent300 = windowAverages(300);
  livePanel.setStatus(
    [
      `live t=${result.simTimeSeconds.toFixed(1)}s tick=${result.tick}`,
      `wood/min 60s=${recent60.wood.toFixed(2)} 5m=${recent300.wood.toFixed(2)}`,
      `pred err 60s=${recent60.pred.toFixed(3)} 5m=${recent300.pred.toFixed(3)}`,
      `novel/min 60s=${recent60.novelty.toFixed(2)} 5m=${recent300.novelty.toFixed(2)}`,
      `composite 60s=${recent60.composite.toFixed(3)} 5m=${recent300.composite.toFixed(3)} clusters 60s=${recent60.clusters.toFixed(2)}`,
    ].join(' | '),
  );
  livePanel.setTimeline(liveTimeline);
  refresh();
}

function startLiveMode(config: LiveModePanelConfig, livePanel: LiveModePanel): void {
  clearLiveTimers();
  liveConfig = config;
  livePaused = false;
  liveFastForward = false;
  liveTimeline.length = 0;
  liveMetricHistory.length = 0;
  liveEngine = new LiveModeEngine({
    seed: config.seed,
    populationSize: config.populationSize,
    ticksPerSecond: config.ticksPerSecond,
    deterministic: config.deterministic,
    rollingSeconds: config.rollingSeconds,
  });
  world = liveEngine.world;
  view = new CanvasView(canvas, world, perception);
  const startAt = performance.now();
  const maxRuntimeMs = config.runIndefinitely ? Number.POSITIVE_INFINITY : config.durationMinutes * 60_000;
  const simulationBatch = (): void => {
    if (!liveEngine || !liveConfig) return;
    if (livePaused) return;
    const ticksToRun = liveFastForward ? Math.max(10, liveConfig.renderEveryNTicks * 2) : 1;
    for (let i = 0; i < ticksToRun; i++) {
      const tickResult = liveEngine.tickOnce();
      if (tickResult.milestones.length) {
        liveTimeline.push(...tickResult.milestones);
        livePanel.setTimeline(liveTimeline);
      }
      maybeRenderLive(tickResult, livePanel);
    }
    if (performance.now() - startAt >= maxRuntimeMs) {
      clearLiveTimers();
      livePanel.setStatus('live completed');
    }
  };
  liveSimulationTimer = window.setInterval(simulationBatch, Math.max(1, Math.round(1000 / Math.max(1, config.ticksPerSecond))));
  const trainConfig: LiveTrainingConfig = {
    trainEveryMs: config.trainEveryMs,
    batchSize: config.batchSize,
    maxTrainMsPerSecond: config.maxTrainMsPerSecond,
    stepsPerTick: 3,
  };
  liveTrainingTimer = window.setInterval(() => {
    if (!liveEngine || livePaused) return;
    liveEngine.trainChunk(trainConfig);
  }, trainConfig.trainEveryMs);
  livePanel.setStatus('live running');
  refresh();
}

const livePanel = new LiveModePanel(panelEl, {
  onStart: (config) => startLiveMode(config, livePanel),
  onPause: () => {
    livePaused = true;
    livePanel.setStatus('live paused');
  },
  onResume: () => {
    livePaused = false;
    livePanel.setStatus('live running');
  },
  onFastForwardToggle: () => {
    liveFastForward = !liveFastForward;
    livePanel.setStatus(liveFastForward ? 'live fast-forward' : 'live running');
  },
  onReset: (nextSeed) => {
    if (!liveConfig) return;
    startLiveMode({ ...liveConfig, seed: nextSeed }, livePanel);
  },
  onExportSnapshot: exportLiveSnapshot,
  onExportMilestones: exportMilestones,
  onBookmark: () => {
    if (!liveEngine) return;
    const bookmark = liveEngine.bookmark();
    livePanel.setBookmarks(liveEngine.bookmarks.map((entry) => entry.id));
    livePanel.setStatus(`bookmarked ${bookmark.id}`);
  },
  onReplayBookmark: replayBookmark,
  onTimelineJump: (milestoneId) => {
    const milestone = liveTimeline.find((entry) => entry.id === milestoneId);
    if (!milestone || !liveEngine) return;
    const focusId = milestone.objectIds[0];
    if (focusId && liveEngine.focusOnObject(focusId)) {
      world.logs.unshift(`Timeline jump: milestone ${milestone.kind}, focusing object ${focusId}`);
    } else {
      world.logs.unshift(`Timeline jump: milestone ${milestone.kind}`);
    }
    refresh();
  },
});

const autopilotPanel = new AutopilotPanel(panelEl, {
  onStart: (config) => {
    void startAutopilot(config);
  },
  onPause: () => {
    autopilotPaused = true;
    autopilotPanel.setStatus('paused');
  },
  onResume: () => {
    autopilotPaused = false;
    autopilotPanel.setStatus('running');
  },
  onCancel: () => {
    autopilotCancelRequested = true;
    autopilotPaused = false;
    autopilotPanel.setStatus('cancelling...');
  },
  onExportNow: exportHistory,
});

async function waitWhilePaused(): Promise<void> {
  while (autopilotPaused && !autopilotCancelRequested) {
    await new Promise((resolve) => setTimeout(resolve, 50));
  }
}

function seedsFor(config: AutopilotConfig): number[] {
  if (config.preset === 'Baseline 20 seeds x 10 episodes') {
    return Array.from({ length: 20 }, (_, i) => config.seed + i);
  }
  if (config.mode === 'sweep') {
    return Array.from({ length: 10 }, (_, i) => config.seed + i);
  }
  return [config.seed];
}

function statusText(lastEpisode: EpisodeMetrics, seedCount: number): string {
  const elapsedSeconds = (performance.now() - autopilotStartedAt) / 1000;
  return `episode ${lastEpisode.episode}/${lastEpisode.totalEpisodes} | seed ${lastEpisode.seed} (${seedCount} seeds) | elapsed ${elapsedSeconds.toFixed(1)}s`;
}

async function runPhase(config: AutopilotConfig, interventions: RunnerInterventions, strategy?: Strategy): Promise<EpisodeMetrics[]> {
  const speedDelayMs = config.mode === 'step-auto' ? Math.max(1, Math.round(1000 / Math.max(1, config.speed))) : 0;
  const results = await runSweep({
    seed: config.seed,
    seeds: seedsFor(config),
    episodes: Math.max(1, config.mode === 'continuous' ? config.episodes : config.episodes),
    stepsPerEpisode: config.mode === 'step-auto' ? 1 : config.stepsPerEpisode,
    randomizeEachEpisode: config.randomizeEachEpisode,
    strategy,
    interventions,
    shouldCancel: () => autopilotCancelRequested,
    waitWhilePaused,
    burstSteps: 250,
    onYield: () => new Promise((resolve) => setTimeout(resolve, speedDelayMs)),
    onEpisodeEnd: (episode) => {
      autopilotHistory.push(episode);
      applyEpisodeMetrics(episode);
      autopilotPanel.setLastEpisode(episode);
      autopilotPanel.setStatus(statusText(episode, seedsFor(config).length));
      if (config.exportEvery === 'episode') exportHistory(config.exportFormat);
    },
  });
  return results;
}

async function startAutopilot(config: AutopilotConfig): Promise<void> {
  if (autopilotRunning) return;
  autopilotRunning = true;
  autopilotPaused = false;
  autopilotCancelRequested = false;
  autopilotStartedAt = performance.now();
  if (config.exportEvery === 'batch') autopilotHistory.length = 0;
  autopilotPanel.setStatus('running');

  const interventions: RunnerInterventions = {
    ...config.interventions,
    disablePredictionModel: realityCheckEnabled ? disablePredictionModel : config.interventions.disablePredictionModel,
    freezeWorldModel: realityCheckEnabled ? freezeWorldModel : config.interventions.freezeWorldModel,
  };

  try {
    if (config.preset === 'Random vs Agent comparison') {
      const random = await runPhase(config, { ...interventions, randomAgent: true }, 'RANDOM_STRIKE');
      const agent = await runPhase(config, { ...interventions, randomAgent: false }, 'BIND_THEN_STRIKE');
      const avg = (values: number[]): number => values.reduce((acc, value) => acc + value, 0) / Math.max(1, values.length);
      randomComparison = {
        random: {
          toolDiscovery: avg(random.map((entry) => entry.result.novelInteractionCount)),
          compositeRate: avg(random.map((entry) => entry.result.compositeDiscoveryRate)),
          predictionAccuracy: avg(random.map((entry) => 1 / (1 + entry.result.predictionErrorMean))),
        },
        agent: {
          toolDiscovery: avg(agent.map((entry) => entry.result.novelInteractionCount)),
          compositeRate: avg(agent.map((entry) => entry.result.compositeDiscoveryRate)),
          predictionAccuracy: avg(agent.map((entry) => 1 / (1 + entry.result.predictionErrorMean))),
        },
      };
      refresh();
    } else {
      const strategy: Strategy = config.interventions.randomAgent ? 'RANDOM_STRIKE' : 'BIND_THEN_STRIKE';
      await runPhase(config, interventions, strategy);
    }
    if (config.exportEvery === 'batch') exportHistory(config.exportFormat);
    autopilotPanel.setStatus(autopilotCancelRequested ? 'cancelled' : 'completed');
  } finally {
    autopilotRunning = false;
    autopilotPaused = false;
    autopilotCancelRequested = false;
  }
}

resetBtn.onclick = () => {
  seed += 1;
  world = new World(seed);
  view = new CanvasView(canvas, world, perception);
  world.logs.unshift(`Reset with seed ${seed}`);
  refresh();
};

trainBtn.onclick = () => {
  metrics.training = trainEpisodes(seed, 100);
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
  const randomResult = runRunnerEpisode({
    seed: compareSeed,
    strategy: 'RANDOM_STRIKE',
    steps: 35,
    interventions: { freezeWorldModel: freezeWorldModel },
  }).result;
  const agentResult = runRunnerEpisode({
    seed: compareSeed,
    strategy: 'BIND_THEN_STRIKE',
    steps: 35,
    interventions: { freezeWorldModel: freezeWorldModel, disablePredictionModel: disablePredictionModel },
  }).result;
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
  const result = runRunnerEpisode({
    seed: recordSeed,
    strategy,
    steps: 35,
    options: {
      freezeWorldModel: freezeWorldModel,
      disablePredictionModel: disablePredictionModel,
      collectTrace: true,
    },
  }).result;
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
  const replay = runRunnerEpisode({
    seed: recordedEpisode.seed,
    strategy: recordedEpisode.strategy,
    steps: 35,
    options: {
      freezeWorldModel: freezeWorldModel,
      disablePredictionModel: disablePredictionModel,
      collectTrace: true,
    },
  }).result;
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
