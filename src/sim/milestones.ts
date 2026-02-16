export interface MilestoneEvent {
  id: number;
  kind:
    | 'first-composite-created'
    | 'first-repeated-composite-pattern'
    | 'first-high-effectiveness-tool'
    | 'first-repeated-sequence'
    | 'first-repeatability-improvement'
    | 'first-low-prediction-error-after-sequence';
  timestamp: number;
  agentId: number;
  objectIds: number[];
  summary: Record<string, number>;
  replayBookmark?: string;
}

export interface MilestoneInput {
  timestamp: number;
  agentId: number;
  action: string;
  objectIds: number[];
  compositeKey?: string;
  predictionError?: number;
  effectiveness?: number;
}

const DEFAULT_REPEAT_COUNT = 3;

function variance(values: number[]): number {
  if (!values.length) return 0;
  const mean = values.reduce((sum, value) => sum + value, 0) / values.length;
  return values.reduce((sum, value) => sum + (value - mean) ** 2, 0) / values.length;
}

export class MilestoneTracker {
  private nextId = 1;
  private readonly events: MilestoneEvent[] = [];
  private readonly seenKinds = new Set<MilestoneEvent['kind']>();
  private readonly compositeCounts = new Map<string, number>();
  private readonly actionWindow: string[] = [];
  private readonly sequenceCounts = new Map<string, number>();
  private readonly sequenceErrors = new Map<string, number[]>();
  private readonly effectivenessSamples: number[] = [];
  private readonly repeatCount: number;

  constructor(repeatCount = DEFAULT_REPEAT_COUNT) {
    this.repeatCount = Math.max(2, repeatCount);
  }

  private emit(
    kind: MilestoneEvent['kind'],
    input: MilestoneInput,
    summary: Record<string, number>,
    replayBookmark?: string,
  ): MilestoneEvent | undefined {
    if (this.seenKinds.has(kind)) return undefined;
    this.seenKinds.add(kind);
    const event: MilestoneEvent = {
      id: this.nextId++,
      kind,
      timestamp: input.timestamp,
      agentId: input.agentId,
      objectIds: input.objectIds,
      summary,
      replayBookmark,
    };
    this.events.push(event);
    return event;
  }

  ingest(input: MilestoneInput, replayBookmark?: string): MilestoneEvent[] {
    const newEvents: MilestoneEvent[] = [];
    this.actionWindow.push(input.action);
    while (this.actionWindow.length > 3) this.actionWindow.shift();

    if (input.compositeKey) {
      const count = (this.compositeCounts.get(input.compositeKey) ?? 0) + 1;
      this.compositeCounts.set(input.compositeKey, count);
      const firstComposite = this.emit('first-composite-created', input, { count }, replayBookmark);
      if (firstComposite) newEvents.push(firstComposite);
      if (count >= 2) {
        const repeatedComposite = this.emit('first-repeated-composite-pattern', input, { count }, replayBookmark);
        if (repeatedComposite) newEvents.push(repeatedComposite);
      }
    }

    if (typeof input.effectiveness === 'number') {
      this.effectivenessSamples.push(input.effectiveness);
      const sorted = [...this.effectivenessSamples].sort((a, b) => a - b);
      const p95Index = Math.max(0, Math.floor(sorted.length * 0.95) - 1);
      const p95 = sorted[p95Index] ?? input.effectiveness;
      if (sorted.length >= 8 && input.effectiveness >= p95) {
        const highTool = this.emit('first-high-effectiveness-tool', input, { effectiveness: input.effectiveness, p95 }, replayBookmark);
        if (highTool) newEvents.push(highTool);
      }
    }

    if (this.actionWindow.length === 3) {
      const sequence = this.actionWindow.join('→');
      const count = (this.sequenceCounts.get(sequence) ?? 0) + 1;
      this.sequenceCounts.set(sequence, count);
      if (count >= this.repeatCount) {
        const repeatedSequence = this.emit('first-repeated-sequence', input, { count }, replayBookmark);
        if (repeatedSequence) newEvents.push(repeatedSequence);
      }
      if (typeof input.predictionError === 'number') {
        const history = this.sequenceErrors.get(sequence) ?? [];
        history.push(input.predictionError);
        this.sequenceErrors.set(sequence, history);
        if (history.length >= this.repeatCount * 2) {
          const previous = history.slice(0, this.repeatCount);
          const latest = history.slice(-this.repeatCount);
          const previousVar = variance(previous);
          const latestVar = variance(latest);
          if (latestVar < previousVar * 0.8) {
            const repeatability = this.emit(
              'first-repeatability-improvement',
              input,
              { previousVariance: previousVar, latestVariance: latestVar },
              replayBookmark,
            );
            if (repeatability) newEvents.push(repeatability);
          }
        }
        if (input.predictionError <= 0.08) {
          const lowError = this.emit('first-low-prediction-error-after-sequence', input, { predictionError: input.predictionError }, replayBookmark);
          if (lowError) newEvents.push(lowError);
        }
      }
    }

    return newEvents;
  }

  all(): MilestoneEvent[] {
    return [...this.events];
  }
}
