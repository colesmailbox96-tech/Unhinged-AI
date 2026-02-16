import type { EpisodeMetrics } from '../runner/runner';

export type AutopilotMode = 'continuous' | 'train' | 'sweep' | 'step-auto';
export type ExportFormat = 'csv' | 'json';
export type ExportEvery = 'episode' | 'batch';

export interface AutopilotConfig {
  mode: AutopilotMode;
  preset: string;
  seed: number;
  episodes: number;
  stepsPerEpisode: number;
  speed: number;
  randomizeEachEpisode: boolean;
  exportFormat: ExportFormat;
  exportEvery: ExportEvery;
  interventions: {
    freezeWorldModel: boolean;
    disablePredictionModel: boolean;
    randomAgent: boolean;
    freezeEmbeddings: boolean;
    determinismReplayCheck: boolean;
  };
}

export interface AutopilotPanelHandlers {
  onStart: (config: AutopilotConfig) => void;
  onPause: () => void;
  onResume: () => void;
  onCancel: () => void;
  onExportNow: (format: ExportFormat) => void;
}

const PRESETS: Record<string, Partial<AutopilotConfig>> = {
  'Baseline 20 seeds x 10 episodes': {
    mode: 'sweep',
    episodes: 10,
    randomizeEachEpisode: false,
    interventions: {
      freezeWorldModel: false,
      disablePredictionModel: false,
      randomAgent: false,
      freezeEmbeddings: false,
      determinismReplayCheck: false,
    },
  },
  'Causality test: disable prediction': {
    mode: 'train',
    episodes: 20,
    interventions: {
      freezeWorldModel: false,
      disablePredictionModel: true,
      randomAgent: false,
      freezeEmbeddings: false,
      determinismReplayCheck: false,
    },
  },
  'Random vs Agent comparison': {
    mode: 'sweep',
    episodes: 10,
    interventions: {
      freezeWorldModel: false,
      disablePredictionModel: false,
      randomAgent: true,
      freezeEmbeddings: false,
      determinismReplayCheck: false,
    },
  },
  'Determinism replay verification': {
    mode: 'train',
    episodes: 10,
    interventions: {
      freezeWorldModel: false,
      disablePredictionModel: false,
      randomAgent: false,
      freezeEmbeddings: false,
      determinismReplayCheck: true,
    },
  },
};

export class AutopilotPanel {
  readonly element: HTMLElement;
  private readonly statusEl: HTMLElement;
  private readonly metricsEl: HTMLElement;

  constructor(container: HTMLElement, handlers: AutopilotPanelHandlers) {
    const presetOptions = ['Custom', ...Object.keys(PRESETS)].map((preset) => `<option value="${preset}">${preset}</option>`).join('');
    const radio = (value: AutopilotMode, label: string, checked = false) =>
      `<label><input type="radio" name="autopilotMode" value="${value}" ${checked ? 'checked' : ''}/> ${label}</label>`;

    const panel = document.createElement('section');
    panel.innerHTML = `
      <div style="margin-top:8px;border-top:1px solid #273248;padding-top:8px">
        <strong>Autopilot / Batch Mode</strong><br />
        <label>Scenario Preset <select data-autopilot="preset">${presetOptions}</select></label><br />
        ${radio('continuous', 'Continuous Episodes', true)}
        ${radio('train', 'Train N Episodes')}
        ${radio('sweep', 'Seed Sweep')}
        ${radio('step-auto', 'Step Auto-Run (K steps/sec)')}
        <div class="metric" style="display:grid;grid-template-columns:1fr 1fr;gap:6px;margin-top:6px">
          <label>Seed <input data-autopilot="seed" type="number" value="1337" /></label>
          <label>Episodes <input data-autopilot="episodes" type="number" min="1" value="100" /></label>
          <label>Steps/episode <input data-autopilot="steps" type="number" min="1" value="35" /></label>
          <label>Speed <input data-autopilot="speed" type="range" min="1" max="60" value="10" /> <span data-autopilot="speedValue">10</span>/sec</label>
          <label><input data-autopilot="randomize" type="checkbox" /> Randomize each episode</label>
          <label>Export every <select data-autopilot="exportEvery"><option value="episode">episode</option><option value="batch">batch complete</option></select></label>
          <label>Export format <select data-autopilot="format"><option value="csv">CSV</option><option value="json">JSON</option></select></label>
          <span></span>
        </div>
        <div class="metric" style="margin-top:6px">
          <label><input data-autopilot="freezeWorldModel" type="checkbox" /> Freeze world model</label><br />
          <label><input data-autopilot="disablePredictionModel" type="checkbox" /> Disable prediction model</label><br />
          <label><input data-autopilot="randomAgent" type="checkbox" /> Random agent</label><br />
          <label><input data-autopilot="freezeEmbeddings" type="checkbox" /> Freeze embeddings</label><br />
          <label><input data-autopilot="determinismReplayCheck" type="checkbox" /> Determinism replay check</label>
        </div>
        <div style="margin-top:6px">
          <button data-autopilot="start">Start</button>
          <button data-autopilot="pause">Pause</button>
          <button data-autopilot="resume">Resume</button>
          <button data-autopilot="cancel">Cancel</button>
          <button data-autopilot="export">Export Now</button>
        </div>
        <div data-autopilot="status" class="metric">idle</div>
        <div data-autopilot="last" class="metric"></div>
      </div>
    `;
    this.element = panel;
    this.statusEl = panel.querySelector('[data-autopilot="status"]') as HTMLElement;
    this.metricsEl = panel.querySelector('[data-autopilot="last"]') as HTMLElement;
    const speedInput = panel.querySelector('[data-autopilot="speed"]') as HTMLInputElement;
    const speedValue = panel.querySelector('[data-autopilot="speedValue"]') as HTMLElement;
    speedInput.addEventListener('input', () => {
      speedValue.textContent = speedInput.value;
    });
    const preset = panel.querySelector('[data-autopilot="preset"]') as HTMLSelectElement;
    preset.addEventListener('change', () => {
      this.applyPreset(preset.value);
    });

    (panel.querySelector('[data-autopilot="start"]') as HTMLButtonElement).onclick = () => handlers.onStart(this.readConfig());
    (panel.querySelector('[data-autopilot="pause"]') as HTMLButtonElement).onclick = handlers.onPause;
    (panel.querySelector('[data-autopilot="resume"]') as HTMLButtonElement).onclick = handlers.onResume;
    (panel.querySelector('[data-autopilot="cancel"]') as HTMLButtonElement).onclick = handlers.onCancel;
    (panel.querySelector('[data-autopilot="export"]') as HTMLButtonElement).onclick = () =>
      handlers.onExportNow(((panel.querySelector('[data-autopilot="format"]') as HTMLSelectElement).value as ExportFormat) ?? 'csv');

    container.appendChild(panel);
  }

  private setChecked(name: string, value: boolean): void {
    (this.element.querySelector(`[data-autopilot="${name}"]`) as HTMLInputElement).checked = value;
  }

  private applyPreset(name: string): void {
    const preset = PRESETS[name];
    if (!preset) return;
    if (preset.mode) {
      const modeInput = this.element.querySelector(`input[name="autopilotMode"][value="${preset.mode}"]`) as HTMLInputElement | null;
      if (modeInput) modeInput.checked = true;
    }
    if (typeof preset.episodes === 'number') (this.element.querySelector('[data-autopilot="episodes"]') as HTMLInputElement).value = String(preset.episodes);
    if (typeof preset.randomizeEachEpisode === 'boolean') this.setChecked('randomize', preset.randomizeEachEpisode);
    if (preset.interventions) {
      this.setChecked('freezeWorldModel', Boolean(preset.interventions.freezeWorldModel));
      this.setChecked('disablePredictionModel', Boolean(preset.interventions.disablePredictionModel));
      this.setChecked('randomAgent', Boolean(preset.interventions.randomAgent));
      this.setChecked('freezeEmbeddings', Boolean(preset.interventions.freezeEmbeddings));
      this.setChecked('determinismReplayCheck', Boolean(preset.interventions.determinismReplayCheck));
    }
  }

  readConfig(): AutopilotConfig {
    const selectedMode = this.element.querySelector('input[name="autopilotMode"]:checked') as HTMLInputElement;
    return {
      mode: (selectedMode?.value as AutopilotMode) ?? 'continuous',
      preset: (this.element.querySelector('[data-autopilot="preset"]') as HTMLSelectElement).value,
      seed: Number((this.element.querySelector('[data-autopilot="seed"]') as HTMLInputElement).value) || 1337,
      episodes: Math.max(1, Number((this.element.querySelector('[data-autopilot="episodes"]') as HTMLInputElement).value) || 1),
      stepsPerEpisode: Math.max(1, Number((this.element.querySelector('[data-autopilot="steps"]') as HTMLInputElement).value) || 35),
      speed: Math.max(1, Number((this.element.querySelector('[data-autopilot="speed"]') as HTMLInputElement).value) || 1),
      randomizeEachEpisode: (this.element.querySelector('[data-autopilot="randomize"]') as HTMLInputElement).checked,
      exportFormat: ((this.element.querySelector('[data-autopilot="format"]') as HTMLSelectElement).value as ExportFormat) ?? 'csv',
      exportEvery: ((this.element.querySelector('[data-autopilot="exportEvery"]') as HTMLSelectElement).value as ExportEvery) ?? 'batch',
      interventions: {
        freezeWorldModel: (this.element.querySelector('[data-autopilot="freezeWorldModel"]') as HTMLInputElement).checked,
        disablePredictionModel: (this.element.querySelector('[data-autopilot="disablePredictionModel"]') as HTMLInputElement).checked,
        randomAgent: (this.element.querySelector('[data-autopilot="randomAgent"]') as HTMLInputElement).checked,
        freezeEmbeddings: (this.element.querySelector('[data-autopilot="freezeEmbeddings"]') as HTMLInputElement).checked,
        determinismReplayCheck: (this.element.querySelector('[data-autopilot="determinismReplayCheck"]') as HTMLInputElement).checked,
      },
    };
  }

  setStatus(message: string): void {
    this.statusEl.textContent = message;
  }

  setLastEpisode(metrics: EpisodeMetrics): void {
    this.metricsEl.innerHTML = `episode ${metrics.episode}/${metrics.totalEpisodes} | seed ${metrics.seed} | wood/min ${metrics.result.woodPerMinute.toFixed(2)} | pred err ${metrics.result.predictionErrorMean.toFixed(3)} | composite ${metrics.result.compositeDiscoveryRate.toFixed(2)} | cluster ${metrics.result.embeddingClusters}`;
  }
}
