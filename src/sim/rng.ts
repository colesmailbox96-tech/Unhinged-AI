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

  clone(): RNG {
    const rng = new RNG(this.state);
    rng.spareNormal = this.spareNormal;
    return rng;
  }
}
