/**
 * Reward Breakdown — decomposes the agent reward into extrinsic, intrinsic,
 * and penalty components for auditing and display.
 */

export interface RewardComponents {
  // Extrinsic
  survival: number;
  foodIntake: number;        // energy gained from actions
  waterIntake: number;       // hydration gained from SOAK
  craftingOutcome: number;   // binding quality, precision improvements
  
  // Intrinsic
  novelty: number;
  predictionError: number;
  empowerment: number;
  skillDiscovery: number;
  
  // Penalties
  spamPenalty: number;
  repeatPenalty: number;
  idlePenalty: number;
  
  // Total
  total: number;
}

export interface RewardBreakdownSnapshot {
  tick: number;
  components: RewardComponents;
  topContributors: Array<{ name: string; value: number }>;
  ema: RewardComponents;
}

function emptyComponents(): RewardComponents {
  return {
    survival: 0,
    foodIntake: 0,
    waterIntake: 0,
    craftingOutcome: 0,
    novelty: 0,
    predictionError: 0,
    empowerment: 0,
    skillDiscovery: 0,
    spamPenalty: 0,
    repeatPenalty: 0,
    idlePenalty: 0,
    total: 0,
  };
}

const COMPONENT_KEYS: (keyof Omit<RewardComponents, 'total'>)[] = [
  'survival', 'foodIntake', 'waterIntake', 'craftingOutcome',
  'novelty', 'predictionError', 'empowerment', 'skillDiscovery',
  'spamPenalty', 'repeatPenalty', 'idlePenalty',
];

export class RewardBreakdown {
  private readonly emaAlpha = 0.05;
  private ema: RewardComponents = emptyComponents();
  private current: RewardComponents = emptyComponents();
  private _tick = 0;
  
  /**
   * Record a new tick's reward components.
   */
  record(tick: number, components: Omit<RewardComponents, 'total'>): RewardBreakdownSnapshot {
    this._tick = tick;
    let total = 0;
    for (const key of COMPONENT_KEYS) {
      total += components[key];
    }
    this.current = { ...components, total };
    
    // Update EMA
    for (const key of COMPONENT_KEYS) {
      this.ema[key] = this.ema[key] * (1 - this.emaAlpha) + this.current[key] * this.emaAlpha;
    }
    this.ema.total = this.ema.total * (1 - this.emaAlpha) + total * this.emaAlpha;
    
    // Top 3 contributors by absolute value
    const sorted = COMPONENT_KEYS
      .map(key => ({ name: key, value: this.current[key] }))
      .sort((a, b) => Math.abs(b.value) - Math.abs(a.value));
    
    return {
      tick,
      components: { ...this.current },
      topContributors: sorted.slice(0, 3),
      ema: { ...this.ema },
    };
  }
  
  /**
   * Get latest snapshot.
   */
  latest(): RewardBreakdownSnapshot {
    const sorted = COMPONENT_KEYS
      .map(key => ({ name: key, value: this.current[key] }))
      .sort((a, b) => Math.abs(b.value) - Math.abs(a.value));
    
    return {
      tick: this._tick,
      components: { ...this.current },
      topContributors: sorted.slice(0, 3),
      ema: { ...this.ema },
    };
  }
}

/**
 * Diminishing returns multiplier for repeating the same verb on the same object.
 * utilityMultiplier = exp(-k * repeatsOnSameObjectInWindow)
 */
export function diminishingReturnMultiplier(repeats: number, k = 0.3): number {
  return Math.exp(-k * repeats);
}

/**
 * Tracks (verb, objectId) pairs within a rolling window to detect spam/loops.
 */
export class RepeatTracker {
  private readonly window = new Map<string, number[]>();
  private readonly windowSize: number;
  
  constructor(windowSize = 20) {
    this.windowSize = windowSize;
  }
  
  /**
   * Record an action and return the repeat count in the current window.
   */
  record(verb: string, objectId: number, tick: number): number {
    const key = `${verb}:${objectId}`;
    let ticks = this.window.get(key);
    if (!ticks) {
      ticks = [];
      this.window.set(key, ticks);
    }
    ticks.push(tick);
    
    // Trim to window
    const cutoff = tick - this.windowSize;
    while (ticks.length > 0 && ticks[0] < cutoff) ticks.shift();
    
    return ticks.length;
  }
  
  /**
   * Get repeat count for a specific verb+object pair.
   */
  getRepeats(verb: string, objectId: number, currentTick: number): number {
    const key = `${verb}:${objectId}`;
    const ticks = this.window.get(key);
    if (!ticks) return 0;
    const cutoff = currentTick - this.windowSize;
    return ticks.filter(t => t >= cutoff).length;
  }
  
  /**
   * Check if an action pair is considered a spam loop (> N repeats).
   */
  isSpamLoop(verb: string, objectId: number, currentTick: number, threshold = 5): boolean {
    return this.getRepeats(verb, objectId, currentTick) >= threshold;
  }
}
