export interface ReplayItem<S, A> {
  state: S;
  action: A;
  reward: number;
  /** TD-error magnitude used for prioritized sampling. */
  priority?: number;
}

export class ReplayBuffer<S, A> {
  private readonly data: ReplayItem<S, A>[] = [];
  private readonly capacity: number;

  constructor(capacity: number) {
    this.capacity = capacity;
  }

  push(item: ReplayItem<S, A>): void {
    if (this.data.length >= this.capacity) this.data.shift();
    this.data.push(item);
  }

  sampleLast(n: number): ReplayItem<S, A>[] {
    return this.data.slice(Math.max(0, this.data.length - n));
  }

  /**
   * Prioritized sampling: items with higher priority (larger prediction error
   * or reward magnitude) are more likely to be selected. Falls back to
   * recency-weighted uniform sampling when no priorities are set.
   *
   * @param n         number of items to sample
   * @param alpha     priority exponent (0 = uniform, 1 = fully prioritized)
   */
  samplePrioritized(n: number, alpha = 0.6): ReplayItem<S, A>[] {
    if (this.data.length === 0) return [];
    const count = Math.min(n, this.data.length);

    // Compute sampling weights
    const weights = this.data.map((item, idx) => {
      const basePriority = item.priority ?? (Math.abs(item.reward) + 0.01);
      const recencyBoost = 1 + (idx / this.data.length) * 0.5; // newer items slightly preferred
      return Math.pow(basePriority * recencyBoost, alpha);
    });
    const totalWeight = weights.reduce((a, b) => a + b, 0);
    if (totalWeight <= 0) return this.sampleLast(count);

    // Weighted sampling without replacement
    const result: ReplayItem<S, A>[] = [];
    const used = new Set<number>();
    for (let i = 0; i < count; i++) {
      let r = Math.random() * totalWeight;
      let chosen = 0;
      for (let j = 0; j < weights.length; j++) {
        if (used.has(j)) continue;
        r -= weights[j];
        if (r <= 0) { chosen = j; break; }
        chosen = j;
      }
      used.add(chosen);
      result.push(this.data[chosen]);
    }
    return result;
  }

  /** Update the priority of the most recently pushed item. */
  updateLastPriority(priority: number): void {
    if (this.data.length > 0) {
      this.data[this.data.length - 1].priority = Math.max(0, priority);
    }
  }

  get size(): number {
    return this.data.length;
  }
}
