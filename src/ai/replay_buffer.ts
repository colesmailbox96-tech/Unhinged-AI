export interface ReplayItem<S, A> {
  state: S;
  action: A;
  reward: number;
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

  get size(): number {
    return this.data.length;
  }
}
