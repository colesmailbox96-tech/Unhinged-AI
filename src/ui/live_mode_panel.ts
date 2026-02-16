import type { MilestoneEvent } from '../sim/milestones';

export interface LiveModePanelConfig {
  seed: number;
  durationMinutes: number;
  runIndefinitely: boolean;
  populationSize: number;
  ticksPerSecond: number;
  trainEveryMs: number;
  batchSize: number;
  maxTrainMsPerSecond: number;
  deterministic: boolean;
  renderEveryNTicks: number;
  rollingSeconds: number;
  showTrueLatentState: boolean;
}

export interface LiveModePanelHandlers {
  onStart: (config: LiveModePanelConfig) => void;
  onPause: () => void;
  onResume: () => void;
  onFastForwardToggle: () => void;
  onReset: (seed: number) => void;
  onExportSnapshot: () => void;
  onExportMilestones: () => void;
  onBookmark: () => void;
  onReplayBookmark: (bookmarkId: string) => void;
  onTimelineJump: (milestoneId: number) => void;
}

export class LiveModePanel {
  readonly element: HTMLElement;
  private readonly statusEl: HTMLElement;
  private readonly timelineEl: HTMLElement;
  private readonly bookmarkEl: HTMLSelectElement;
  private readonly dashboardEl: HTMLElement;

  constructor(container: HTMLElement, handlers: LiveModePanelHandlers) {
    const panel = document.createElement('section');
    panel.innerHTML = `
      <div style="margin-top:8px;border-top:1px solid #273248;padding-top:8px">
        <strong>Live Mode</strong><br />
        <div class="metric" style="display:grid;grid-template-columns:1fr 1fr;gap:6px;margin-top:6px">
          <label>Seed <input data-live="seed" type="number" value="1337" /></label>
          <label>Duration (min) <input data-live="duration" type="number" min="1" value="10" /></label>
          <label><input data-live="indefinite" type="checkbox" /> Run indefinitely</label>
          <label>Population (N) <input data-live="population" type="number" min="1" value="4" /></label>
          <label>Ticks/sec <input data-live="tps" type="range" min="1" max="120" value="20" /> <span data-live="tpsValue">20</span></label>
          <label>Render every N ticks <input data-live="renderEvery" type="number" min="1" value="5" /></label>
          <label>trainEveryMs <input data-live="trainEveryMs" type="number" min="5" value="50" /></label>
          <label>batchSize <input data-live="batchSize" type="number" min="1" value="8" /></label>
          <label>maxTrainMs/sec <input data-live="trainBudget" type="number" min="1" value="35" /></label>
          <label>Rolling record (sec) <input data-live="rollingSeconds" type="number" min="5" value="30" /></label>
          <label><input data-live="deterministic" type="checkbox" /> Deterministic Live</label>
          <label><input data-live="showLatentDebug" type="checkbox" /> Show true latent state (debug)</label>
        </div>
        <div style="margin-top:6px">
          <button data-live="start">Start Live</button>
          <button data-live="pause">Pause</button>
          <button data-live="resume">Resume</button>
          <button data-live="fastForward">Fast-Forward</button>
          <button data-live="reset">Reset World</button>
          <button data-live="snapshot">Export Snapshot</button>
          <button data-live="exportMilestones">Export Milestones</button>
        </div>
        <div style="margin-top:6px">
          <button data-live="bookmark">Bookmark</button>
          <select data-live="bookmarkSelect"></select>
          <button data-live="replayBookmark">Replay bookmark</button>
        </div>
        <div data-live="status" class="metric">idle</div>
        <div data-live="manufacturing" class="metric" style="max-height:180px;overflow:auto"></div>
        <div data-live="timeline" class="metric" style="max-height:180px;overflow:auto"></div>
      </div>
    `;
    this.element = panel;
    this.statusEl = panel.querySelector('[data-live="status"]') as HTMLElement;
    this.timelineEl = panel.querySelector('[data-live="timeline"]') as HTMLElement;
    this.bookmarkEl = panel.querySelector('[data-live="bookmarkSelect"]') as HTMLSelectElement;
    this.dashboardEl = panel.querySelector('[data-live="manufacturing"]') as HTMLElement;
    const tpsInput = panel.querySelector('[data-live="tps"]') as HTMLInputElement;
    const tpsValue = panel.querySelector('[data-live="tpsValue"]') as HTMLElement;
    tpsInput.addEventListener('input', () => {
      tpsValue.textContent = tpsInput.value;
    });
    (panel.querySelector('[data-live="start"]') as HTMLButtonElement).onclick = () => handlers.onStart(this.readConfig());
    (panel.querySelector('[data-live="pause"]') as HTMLButtonElement).onclick = handlers.onPause;
    (panel.querySelector('[data-live="resume"]') as HTMLButtonElement).onclick = handlers.onResume;
    (panel.querySelector('[data-live="fastForward"]') as HTMLButtonElement).onclick = handlers.onFastForwardToggle;
    (panel.querySelector('[data-live="reset"]') as HTMLButtonElement).onclick = () =>
      handlers.onReset(Number((panel.querySelector('[data-live="seed"]') as HTMLInputElement).value) || 1337);
    (panel.querySelector('[data-live="snapshot"]') as HTMLButtonElement).onclick = handlers.onExportSnapshot;
    (panel.querySelector('[data-live="exportMilestones"]') as HTMLButtonElement).onclick = handlers.onExportMilestones;
    (panel.querySelector('[data-live="bookmark"]') as HTMLButtonElement).onclick = handlers.onBookmark;
    (panel.querySelector('[data-live="replayBookmark"]') as HTMLButtonElement).onclick = () => {
      if (this.bookmarkEl.value) handlers.onReplayBookmark(this.bookmarkEl.value);
    };
    panel.addEventListener('click', (event) => {
      const target = event.target as HTMLElement | null;
      const id = target?.getAttribute('data-milestone');
      if (id) handlers.onTimelineJump(Number(id));
    });
    container.insertBefore(panel, container.firstChild);
  }

  readConfig(): LiveModePanelConfig {
    return {
      seed: Number((this.element.querySelector('[data-live="seed"]') as HTMLInputElement).value) || 1337,
      durationMinutes: Math.max(1, Number((this.element.querySelector('[data-live="duration"]') as HTMLInputElement).value) || 10),
      runIndefinitely: (this.element.querySelector('[data-live="indefinite"]') as HTMLInputElement).checked,
      populationSize: Math.max(1, Number((this.element.querySelector('[data-live="population"]') as HTMLInputElement).value) || 1),
      ticksPerSecond: Math.max(1, Number((this.element.querySelector('[data-live="tps"]') as HTMLInputElement).value) || 20),
      trainEveryMs: Math.max(5, Number((this.element.querySelector('[data-live="trainEveryMs"]') as HTMLInputElement).value) || 50),
      batchSize: Math.max(1, Number((this.element.querySelector('[data-live="batchSize"]') as HTMLInputElement).value) || 8),
      maxTrainMsPerSecond: Math.max(1, Number((this.element.querySelector('[data-live="trainBudget"]') as HTMLInputElement).value) || 35),
      deterministic: (this.element.querySelector('[data-live="deterministic"]') as HTMLInputElement).checked,
      renderEveryNTicks: Math.max(1, Number((this.element.querySelector('[data-live="renderEvery"]') as HTMLInputElement).value) || 5),
      rollingSeconds: Math.max(5, Number((this.element.querySelector('[data-live="rollingSeconds"]') as HTMLInputElement).value) || 30),
      showTrueLatentState: (this.element.querySelector('[data-live="showLatentDebug"]') as HTMLInputElement).checked,
    };
  }

  setStatus(status: string): void {
    this.statusEl.textContent = status;
  }

  setTimeline(events: MilestoneEvent[]): void {
    this.timelineEl.innerHTML = events
      .slice()
      .reverse()
      .map((event) => {
        const summary = Object.entries(event.summary)
          .slice(0, 3)
          .map(([key, value]) => `${key}=${value.toFixed(3)}`)
          .join(' ');
        const label = `t=${event.timestamp.toFixed(1)}s agent=${event.agentId} ${event.kind} ${summary}`;
        return `<button data-milestone="${Number(event.id)}" style="display:block;width:100%;text-align:left;margin:2px 0">${label
          .replaceAll('&', '&amp;')
          .replaceAll('<', '&lt;')
          .replaceAll('>', '&gt;')}</button>`;
      })
      .join('');
  }

  setManufacturingDashboard(lines: string[]): void {
    this.dashboardEl.innerHTML = lines.join('<br/>');
  }

  setBookmarks(bookmarkIds: string[]): void {
    this.bookmarkEl.innerHTML = bookmarkIds
      .map((bookmarkId) => {
        const safe = bookmarkId.replaceAll('&', '&amp;').replaceAll('<', '&lt;').replaceAll('>', '&gt;').replaceAll('"', '&quot;');
        return `<option value="${safe}">${safe}</option>`;
      })
      .join('');
  }
}
