/**
 * Event Bus — decouples simulation events from rendering and UI logic.
 *
 * Any subsystem can emit events (object created, destroyed, skill learned,
 * milestone reached, agent action, etc.) and any listener can subscribe
 * without creating hard dependencies.
 */

export type SimEventKind =
  | 'object-created'
  | 'object-destroyed'
  | 'object-fractured'
  | 'composite-bound'
  | 'station-anchored'
  | 'station-decayed'
  | 'skill-learned'
  | 'milestone-reached'
  | 'agent-action'
  | 'agent-died'
  | 'agent-needs-critical'
  | 'measurement-taken'
  | 'controller-converged'
  | 'stall-detected'
  | 'environmental-event'
  | 'hazard-exposure'
  | 'shelter-used';

export interface SimEvent {
  kind: SimEventKind;
  tick: number;
  /** Affected object/agent identifiers. */
  ids: number[];
  /** Free-form payload. */
  data: Record<string, unknown>;
}

export type SimEventListener = (event: SimEvent) => void;

export class EventBus {
  private readonly listeners = new Map<SimEventKind | '*', SimEventListener[]>();
  private readonly history: SimEvent[] = [];
  private readonly maxHistory: number;

  constructor(maxHistory = 500) {
    this.maxHistory = maxHistory;
  }

  /** Subscribe to a specific event kind, or '*' for all events. */
  on(kind: SimEventKind | '*', listener: SimEventListener): () => void {
    let list = this.listeners.get(kind);
    if (!list) {
      list = [];
      this.listeners.set(kind, list);
    }
    list.push(listener);
    // Return unsubscribe function
    return () => {
      const arr = this.listeners.get(kind);
      if (arr) {
        const idx = arr.indexOf(listener);
        if (idx >= 0) arr.splice(idx, 1);
      }
    };
  }

  /** Emit an event to all matching listeners. */
  emit(event: SimEvent): void {
    this.history.push(event);
    if (this.history.length > this.maxHistory) {
      this.history.splice(0, this.history.length - this.maxHistory);
    }
    const specific = this.listeners.get(event.kind);
    if (specific) for (const fn of specific) fn(event);
    const wildcard = this.listeners.get('*');
    if (wildcard) for (const fn of wildcard) fn(event);
  }

  /** Get recent event history. */
  recent(count?: number): SimEvent[] {
    if (count === undefined) return [...this.history];
    return this.history.slice(-count);
  }

  /** Count events of a given kind in the last N ticks. */
  countSince(kind: SimEventKind, sinceTick: number): number {
    return this.history.filter(e => e.kind === kind && e.tick >= sinceTick).length;
  }

  /** Remove all listeners and history. */
  clear(): void {
    this.listeners.clear();
    this.history.length = 0;
  }
}
