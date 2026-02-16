/**
 * EvolutionPanel — "Start Evolution" button, progress dashboard with mini charts,
 * milestones feed, and "Prove Learning" button.
 */

export type EvolutionStrategy = 'online' | 'rollout';

export interface EvolutionPanelConfig {
  strategy: EvolutionStrategy;
}

export interface EvolutionDashboardData {
  rewardPerMin: number;
  novelPerMin: number;
  toolClusters: number;
  predictionError: number;
  trainingStepsPerMin: number;
  lossEMA: number;
  policyEntropy: number;
  stallScore: number;
  stallEventsPerMin: number;
  timeInStallPct: number;
  spawnThrottle: number;
  debrisCleanupRate: number;
  trainingState: string;
  trainingStepsTotal: number;
  replaySize: number;
}

export interface EvolutionMilestone {
  time: number;
  label: string;
}

export interface ProveLearningReport {
  beforeAvgRewardPerMin: number;
  afterAvgRewardPerMin: number;
  beforeNovelPerMin: number;
  afterNovelPerMin: number;
  beforeClusters: number;
  afterClusters: number;
  beforeStallPct: number;
  afterStallPct: number;
  improved: boolean;
}

export interface EvolutionPanelHandlers {
  onStartEvolution: (config: EvolutionPanelConfig) => void;
  onStopEvolution: () => void;
  onProveLearning: () => void;
}

interface TimeSeriesPoint {
  t: number;
  v: number;
}

const MAX_SERIES = 600; // 10 minutes at 1/s

function drawMiniChart(canvas: HTMLCanvasElement, data: TimeSeriesPoint[], label: string, color: string): void {
  const ctx = canvas.getContext('2d');
  if (!ctx) return;
  const w = canvas.width;
  const h = canvas.height;
  ctx.clearRect(0, 0, w, h);

  // Background
  ctx.fillStyle = '#0d1420';
  ctx.fillRect(0, 0, w, h);

  if (data.length < 2) {
    ctx.fillStyle = '#556';
    ctx.font = '10px system-ui';
    ctx.fillText('waiting…', 4, h / 2 + 3);
    ctx.fillText(label, 4, 10);
    return;
  }

  const vals = data.map(p => p.v);
  const minV = Math.min(...vals);
  const maxV = Math.max(...vals);
  const rangeV = Math.max(1e-9, maxV - minV);
  const tMin = data[0].t;
  const tMax = data[data.length - 1].t;
  const rangeT = Math.max(1e-9, tMax - tMin);

  ctx.strokeStyle = color;
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  for (let i = 0; i < data.length; i++) {
    const x = ((data[i].t - tMin) / rangeT) * (w - 4) + 2;
    const y = h - 2 - ((data[i].v - minV) / rangeV) * (h - 14);
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  }
  ctx.stroke();

  // Label + last value
  ctx.fillStyle = '#aab';
  ctx.font = '9px system-ui';
  ctx.fillText(label, 3, 9);
  ctx.fillStyle = color;
  const lastV = vals[vals.length - 1];
  ctx.fillText(lastV.toFixed(2), w - 36, 9);
}

export class EvolutionPanel {
  readonly element: HTMLElement;
  private readonly statusEl: HTMLElement;
  private readonly dashboardEl: HTMLElement;
  private readonly milestonesEl: HTMLElement;
  private readonly reportEl: HTMLElement;
  private readonly chartCanvases: HTMLCanvasElement[] = [];
  private readonly series: TimeSeriesPoint[][] = [[], [], [], []];

  constructor(container: HTMLElement, handlers: EvolutionPanelHandlers) {
    const panel = document.createElement('section');
    panel.innerHTML = `
      <div style="margin-top:8px;border-top:1px solid #273248;padding-top:8px">
        <strong>Evolution Mode</strong><br />
        <div style="margin:6px 0">
          <label>Strategy: <select data-evo="strategy">
            <option value="online">Online</option>
            <option value="rollout">Rollout</option>
          </select></label>
        </div>
        <div style="margin:4px 0">
          <button data-evo="start">Start Evolution</button>
          <button data-evo="stop" disabled>Stop Evolution</button>
          <button data-evo="prove">Prove Learning</button>
        </div>
        <div data-evo="status" class="metric">idle</div>
        <div data-evo="charts" style="display:grid;grid-template-columns:1fr 1fr;gap:4px;margin-top:6px">
          <canvas data-evo="chart0" width="160" height="60" style="border:1px solid #273248;border-radius:4px"></canvas>
          <canvas data-evo="chart1" width="160" height="60" style="border:1px solid #273248;border-radius:4px"></canvas>
          <canvas data-evo="chart2" width="160" height="60" style="border:1px solid #273248;border-radius:4px"></canvas>
          <canvas data-evo="chart3" width="160" height="60" style="border:1px solid #273248;border-radius:4px"></canvas>
        </div>
        <div data-evo="dashboard" class="metric" style="margin-top:6px"></div>
        <div data-evo="milestones" class="metric" style="max-height:120px;overflow:auto;margin-top:4px"></div>
        <div data-evo="report" class="metric" style="margin-top:4px"></div>
      </div>
    `;
    this.element = panel;
    this.statusEl = panel.querySelector('[data-evo="status"]') as HTMLElement;
    this.dashboardEl = panel.querySelector('[data-evo="dashboard"]') as HTMLElement;
    this.milestonesEl = panel.querySelector('[data-evo="milestones"]') as HTMLElement;
    this.reportEl = panel.querySelector('[data-evo="report"]') as HTMLElement;
    for (let i = 0; i < 4; i++) {
      this.chartCanvases.push(panel.querySelector(`[data-evo="chart${i}"]`) as HTMLCanvasElement);
    }

    const startBtn = panel.querySelector('[data-evo="start"]') as HTMLButtonElement;
    const stopBtn = panel.querySelector('[data-evo="stop"]') as HTMLButtonElement;
    const proveBtn = panel.querySelector('[data-evo="prove"]') as HTMLButtonElement;
    const strategySelect = panel.querySelector('[data-evo="strategy"]') as HTMLSelectElement;

    startBtn.onclick = () => {
      const strategy = strategySelect.value as EvolutionStrategy;
      handlers.onStartEvolution({ strategy });
      startBtn.disabled = true;
      stopBtn.disabled = false;
    };
    stopBtn.onclick = () => {
      handlers.onStopEvolution();
      startBtn.disabled = false;
      stopBtn.disabled = true;
    };
    proveBtn.onclick = () => handlers.onProveLearning();

    container.insertBefore(panel, container.firstChild);
  }

  setStatus(status: string): void {
    this.statusEl.textContent = status;
  }

  updateDashboard(data: EvolutionDashboardData, timeSec: number): void {
    // Push to series
    const pushSeries = (idx: number, v: number): void => {
      this.series[idx].push({ t: timeSec, v });
      while (this.series[idx].length > MAX_SERIES) this.series[idx].shift();
    };
    pushSeries(0, data.rewardPerMin);
    pushSeries(1, data.novelPerMin);
    pushSeries(2, data.toolClusters);
    pushSeries(3, data.predictionError);

    // Draw charts
    drawMiniChart(this.chartCanvases[0], this.series[0], 'Reward/min', '#4fc47a');
    drawMiniChart(this.chartCanvases[1], this.series[1], 'Novel/min', '#5bafff');
    drawMiniChart(this.chartCanvases[2], this.series[2], 'Clusters', '#d6a74f');
    drawMiniChart(this.chartCanvases[3], this.series[3], 'Pred Error', '#e06060');

    // Dashboard text
    const safe = (s: string): string => s.replace(/</g, '&lt;');
    this.dashboardEl.innerHTML = [
      `training: <b>${safe(data.trainingState)}</b> steps=${data.trainingStepsTotal} steps/min=${data.trainingStepsPerMin.toFixed(1)}`,
      `loss(EMA)=${data.lossEMA.toFixed(4)} entropy(EMA)=${data.policyEntropy.toFixed(3)} replay=${data.replaySize}`,
      `stall score=${data.stallScore.toFixed(2)} events/min=${data.stallEventsPerMin.toFixed(1)} inStall=${data.timeInStallPct.toFixed(1)}%`,
      `spawn throttle=${data.spawnThrottle.toFixed(2)} debris cleanup=${data.debrisCleanupRate}`,
    ].join('<br/>');
  }

  addMilestone(milestone: EvolutionMilestone): void {
    const div = document.createElement('div');
    div.style.fontSize = '11px';
    div.style.padding = '2px 0';
    div.textContent = `t=${milestone.time.toFixed(1)}s ${milestone.label}`;
    this.milestonesEl.insertBefore(div, this.milestonesEl.firstChild);
    // Cap visible milestones
    while (this.milestonesEl.children.length > 30) {
      this.milestonesEl.removeChild(this.milestonesEl.lastChild!);
    }
  }

  setReport(report: ProveLearningReport): void {
    const fmt = (v: number): string => v.toFixed(3);
    this.reportEl.innerHTML = [
      `<b>Prove Learning Report</b>`,
      `reward/min: before=${fmt(report.beforeAvgRewardPerMin)} after=${fmt(report.afterAvgRewardPerMin)}`,
      `novel/min: before=${fmt(report.beforeNovelPerMin)} after=${fmt(report.afterNovelPerMin)}`,
      `clusters: before=${report.beforeClusters} after=${report.afterClusters}`,
      `stall%: before=${fmt(report.beforeStallPct)} after=${fmt(report.afterStallPct)}`,
      report.improved ? '<b style="color:#4fc47a">✓ Improvement detected</b>' : '<b style="color:#e06060">✗ No improvement detected</b>',
    ].join('<br/>');
  }

  setReportJSON(json: string): void {
    this.reportEl.innerHTML = `<pre style="max-height:200px;overflow:auto">${json}</pre>`;
  }

  enableStartButton(): void {
    const startBtn = this.element.querySelector('[data-evo="start"]') as HTMLButtonElement;
    const stopBtn = this.element.querySelector('[data-evo="stop"]') as HTMLButtonElement;
    if (startBtn) startBtn.disabled = false;
    if (stopBtn) stopBtn.disabled = true;
  }
}
