/**
 * StallDetector — detects agent behaviour loops and triggers loop-breakers.
 *
 * Monitors per-agent rolling windows and flags stall when:
 *   - Δreward_total < eps  AND  uniqueTargetsTouched < k
 *   - OR repeatedSameAction > threshold
 */

export interface StallMetrics {
  stallEventsPerMin: number;
  timeInStallPct: number;
  isStalled: boolean;
}

interface StallWindow {
  rewards: number[];
  actions: string[];
  targetsTouched: Set<number>;
  objectsUsed: Set<number>;
  windowStart: number;
  stallEvents: number;
  stallTicks: number;
  totalTicks: number;
  /** ticks remaining in a forced-explore cool-down */
  forcedExploreTicks: number;
}

const DEFAULTS = {
  windowSize: 40,        // ticks to look back
  rewardEps: 0.005,      // min Δreward to not be stalled
  minTargets: 1,         // min unique targets in window
  repeatThreshold: 15,   // max same-action repeats
  forcedExploreDuration: 10, // ticks of forced exploration after stall
} as const;

export class StallDetector {
  private readonly windows = new Map<number, StallWindow>();
  private readonly cfg = { ...DEFAULTS };

  getOrCreate(agentId: number, nowTick: number): StallWindow {
    let w = this.windows.get(agentId);
    if (!w) {
      w = {
        rewards: [],
        actions: [],
        targetsTouched: new Set(),
        objectsUsed: new Set(),
        windowStart: nowTick,
        stallEvents: 0,
        stallTicks: 0,
        totalTicks: 0,
        forcedExploreTicks: 0,
      };
      this.windows.set(agentId, w);
    }
    return w;
  }

  /**
   * Record one tick of agent activity.
   * Returns true if a stall-break should be triggered this tick.
   */
  record(
    agentId: number,
    tick: number,
    action: string,
    reward: number,
    targetId?: number,
    objectId?: number,
  ): boolean {
    const w = this.getOrCreate(agentId, tick);
    w.totalTicks += 1;
    w.rewards.push(reward);
    w.actions.push(action);
    if (targetId !== undefined) w.targetsTouched.add(targetId);
    if (objectId !== undefined) w.objectsUsed.add(objectId);

    // Trim window
    while (w.rewards.length > this.cfg.windowSize) w.rewards.shift();
    while (w.actions.length > this.cfg.windowSize) w.actions.shift();
    if (w.rewards.length >= this.cfg.windowSize) {
      w.targetsTouched = new Set([...w.targetsTouched].slice(-this.cfg.windowSize));
      w.objectsUsed = new Set([...w.objectsUsed].slice(-this.cfg.windowSize));
    }

    // Decrement forced-explore counter
    if (w.forcedExploreTicks > 0) {
      w.forcedExploreTicks -= 1;
      return false; // don't re-trigger during cooldown
    }

    if (w.rewards.length < 5) return false; // not enough data

    const deltaReward = Math.abs(
      w.rewards[w.rewards.length - 1] - w.rewards[0],
    );
    const uniqueTargets = w.targetsTouched.size;
    const repeatedSame = (() => {
      let count = 0;
      const last = w.actions[w.actions.length - 1];
      for (let i = w.actions.length - 1; i >= 0; i--) {
        if (w.actions[i] === last) count++;
        else break;
      }
      return count;
    })();

    const stalled =
      (deltaReward < this.cfg.rewardEps && uniqueTargets < this.cfg.minTargets) ||
      repeatedSame >= this.cfg.repeatThreshold;

    if (stalled) {
      w.stallTicks += 1;
      w.stallEvents += 1;
      w.forcedExploreTicks = this.cfg.forcedExploreDuration;
      // Reset window for next cycle
      w.rewards.length = 0;
      w.actions.length = 0;
      w.targetsTouched.clear();
      w.objectsUsed.clear();
      return true;
    }
    return false;
  }

  isInForcedExplore(agentId: number): boolean {
    const w = this.windows.get(agentId);
    return w ? w.forcedExploreTicks > 0 : false;
  }

  metrics(elapsedMinutes: number): StallMetrics {
    let totalEvents = 0;
    let totalStallTicks = 0;
    let totalTicks = 0;
    for (const w of this.windows.values()) {
      totalEvents += w.stallEvents;
      totalStallTicks += w.stallTicks;
      totalTicks += w.totalTicks;
    }
    return {
      stallEventsPerMin: totalEvents / Math.max(1 / 60, elapsedMinutes),
      timeInStallPct: totalTicks > 0 ? (totalStallTicks / totalTicks) * 100 : 0,
      isStalled: [...this.windows.values()].some(w => w.forcedExploreTicks > 0),
    };
  }
}
