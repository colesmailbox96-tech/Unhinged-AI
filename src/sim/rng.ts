export class RNG {
  private state: number;
  private spareNormal: number | null = null;

  constructor(seed: number) {
    this.state = seed >>> 0;
    if (this.state === 0) this.state = 0x6d2b79f5;
  }

  nextU32(): number {
    let x = this.state;
    x ^= x << 13;
    x ^= x >>> 17;
    x ^= x << 5;
    this.state = x >>> 0;
    return this.state;
  }

  float(): number {
    return this.nextU32() / 0xffffffff;
  }

  range(min: number, max: number): number {
    return min + (max - min) * this.float();
  }

  int(min: number, maxExclusive: number): number {
    return Math.floor(this.range(min, maxExclusive));
  }

  normal(mean = 0, stdDev = 1): number {
    if (this.spareNormal !== null) {
      const value = this.spareNormal;
      this.spareNormal = null;
      return mean + value * stdDev;
    }
    const u = Math.max(1e-9, this.float());
    const v = Math.max(1e-9, this.float());
    const mag = Math.sqrt(-2 * Math.log(u));
    const z0 = mag * Math.cos(2 * Math.PI * v);
    const z1 = mag * Math.sin(2 * Math.PI * v);
    this.spareNormal = z1;
    return mean + z0 * stdDev;
  }

  /** Pick a random element from the array. Returns undefined for empty arrays. */
  pick<T>(arr: readonly T[]): T | undefined {
    if (arr.length === 0) return undefined;
    return arr[this.int(0, arr.length)];
  }

  /** Return a shuffled copy of the array (Fisher-Yates). */
  shuffle<T>(arr: readonly T[]): T[] {
    const out = [...arr];
    for (let i = out.length - 1; i > 0; i--) {
      const j = this.int(0, i + 1);
      [out[i], out[j]] = [out[j], out[i]];
    }
    return out;
  }

  clone(): RNG {
    const rng = new RNG(1); // dummy seed, overwritten below
    rng.state = this.state;
    rng.spareNormal = this.spareNormal;
    return rng;
  }
}
