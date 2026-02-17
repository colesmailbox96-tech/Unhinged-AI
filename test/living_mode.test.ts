import { describe, it, expect } from 'vitest';
import { LiveModeEngine } from '../src/runner/live_mode';
import { tickNeeds, createDefaultNeeds, homeostasisReward, mostUrgentNeed } from '../src/sim/needs';
import { SkillTracker } from '../src/sim/skills';
import { RewardBreakdown, RepeatTracker, diminishingReturnMultiplier } from '../src/sim/reward_breakdown';

describe('Living Mode - Needs & Metabolism', () => {
  it('energy decreases over time with actions', () => {
    let needs = createDefaultNeeds();
    for (let i = 0; i < 20; i++) {
      const result = tickNeeds(needs, 'STRIKE_WITH');
      needs = result.needs;
    }
    expect(needs.energy).toBeLessThan(0.85);
    expect(needs.hydration).toBeLessThan(0.8);
  });

  it('REST recovers fatigue and energy', () => {
    let needs = createDefaultNeeds();
    // First tire the agent
    for (let i = 0; i < 10; i++) {
      needs = tickNeeds(needs, 'STRIKE_WITH').needs;
    }
    const tiredEnergy = needs.energy;
    const tiredFatigue = needs.fatigue;
    // Now rest
    for (let i = 0; i < 10; i++) {
      needs = tickNeeds(needs, 'REST').needs;
    }
    expect(needs.fatigue).toBeLessThan(tiredFatigue);
    expect(needs.energy).toBeGreaterThan(tiredEnergy);
  });

  it('SOAK increases hydration', () => {
    let needs = createDefaultNeeds();
    needs.hydration = 0.3;
    needs = tickNeeds(needs, 'SOAK').needs;
    expect(needs.hydration).toBeGreaterThan(0.3);
  });

  it('temperature drifts toward ambient (0.5)', () => {
    let needs = createDefaultNeeds();
    needs.temperature = 0.9;
    for (let i = 0; i < 50; i++) {
      needs = tickNeeds(needs, 'REST').needs;
    }
    expect(needs.temperature).toBeLessThan(0.9);
    expect(needs.temperature).toBeCloseTo(0.5, 0);
  });

  it('agent dies when energy depleted', () => {
    const needs = createDefaultNeeds();
    needs.energy = 0.04;
    const result = tickNeeds(needs, 'STRIKE_WITH');
    expect(result.alive).toBe(false);
  });

  it('homeostasisReward is positive for healthy state', () => {
    const healthy = createDefaultNeeds();
    expect(homeostasisReward(healthy)).toBeGreaterThan(0);
  });

  it('homeostasisReward is negative for depleted state', () => {
    const depleted = createDefaultNeeds();
    depleted.energy = 0.1;
    depleted.hydration = 0.1;
    expect(homeostasisReward(depleted)).toBeLessThan(homeostasisReward(createDefaultNeeds()));
  });

  it('mostUrgentNeed identifies low energy', () => {
    const needs = createDefaultNeeds();
    needs.energy = 0.15;
    const urgent = mostUrgentNeed(needs);
    expect(urgent.need).toBe('energy');
    expect(urgent.urgency).toBeGreaterThan(0);
  });
});

describe('Living Mode - Skill Discovery', () => {
  it('discovers skill after repeated successful transformations', () => {
    const tracker = new SkillTracker();
    // Simulate grinding increasing sharpness reliably
    for (let i = 0; i < 8; i++) {
      tracker.recordTransformation('GRIND', 'sharpness', 0.3, 0.35 + i * 0.01, i);
    }
    const owned = tracker.ownedSkills();
    expect(owned.length).toBeGreaterThanOrEqual(1);
    expect(owned[0].name).toContain('GRIND');
    expect(owned[0].name).toContain('sharpness');
  });

  it('does not discover skill with insufficient trials', () => {
    const tracker = new SkillTracker();
    tracker.recordTransformation('GRIND', 'sharpness', 0.3, 0.35, 0);
    tracker.recordTransformation('GRIND', 'sharpness', 0.35, 0.4, 1);
    expect(tracker.ownedSkills().length).toBe(0);
  });

  it('returns diminishing reward for subsequent discoveries', () => {
    const tracker = new SkillTracker();
    // First skill
    let reward1 = 0;
    for (let i = 0; i < 8; i++) {
      reward1 += tracker.recordTransformation('GRIND', 'sharpness', 0.3, 0.35 + i * 0.01, i);
    }
    // Second skill
    let reward2 = 0;
    for (let i = 0; i < 8; i++) {
      reward2 += tracker.recordTransformation('BIND_TO', 'mass', 0.3, 0.35 + i * 0.01, 10 + i);
    }
    expect(reward1).toBeGreaterThan(0);
    expect(reward2).toBeGreaterThan(0);
    // First discovery gives more reward due to diminishing returns
    expect(reward1).toBeGreaterThanOrEqual(reward2);
  });
});

describe('Living Mode - Anti-Degeneracy', () => {
  it('diminishing returns multiplier decreases with repeats', () => {
    expect(diminishingReturnMultiplier(0)).toBeCloseTo(1, 3);
    expect(diminishingReturnMultiplier(5)).toBeLessThan(0.3);
    expect(diminishingReturnMultiplier(10)).toBeLessThan(0.1);
  });

  it('RepeatTracker counts repeats in window', () => {
    const tracker = new RepeatTracker(10);
    for (let i = 0; i < 5; i++) {
      tracker.record('SOAK', 42, i);
    }
    expect(tracker.getRepeats('SOAK', 42, 5)).toBe(5);
    expect(tracker.isSpamLoop('SOAK', 42, 5)).toBe(true);
  });

  it('RepeatTracker window expires old entries', () => {
    const tracker = new RepeatTracker(5);
    for (let i = 0; i < 3; i++) {
      tracker.record('SOAK', 42, i);
    }
    // After window passes, old entries expire
    expect(tracker.getRepeats('SOAK', 42, 10)).toBe(0);
  });

  it('SOAK repeat on same object yields diminishing net-negative returns', () => {
    const engine = new LiveModeEngine({
      seed: 42,
      populationSize: 1,
      ticksPerSecond: 20,
      deterministic: true,
      rollingSeconds: 30,
      livingMode: true,
    });
    // Run several ticks to get some baseline
    for (let i = 0; i < 10; i++) {
      engine.tickOnce();
    }
    // After many repeat SOAKs tracked, penalty should accumulate
    for (let i = 0; i < 15; i++) {
      engine.repeatTracker.record('SOAK', 1, 10 + i);
    }
    const repeats = engine.repeatTracker.getRepeats('SOAK', 1, 25);
    expect(repeats).toBeGreaterThan(10);
    const multiplier = diminishingReturnMultiplier(repeats);
    expect(multiplier).toBeLessThan(0.1);
  });
});

describe('Living Mode - Reward Breakdown', () => {
  it('tracks components and computes top contributors', () => {
    const rb = new RewardBreakdown();
    const snapshot = rb.record(1, {
      survival: 0.1,
      foodIntake: 0.05,
      waterIntake: 0,
      craftingOutcome: 0,
      novelty: 0.03,
      predictionError: 0.02,
      empowerment: 0.01,
      skillDiscovery: 0.15,
      spamPenalty: 0,
      repeatPenalty: -0.08,
      idlePenalty: 0,
    });
    expect(snapshot.components.total).toBeCloseTo(0.28, 1);
    expect(snapshot.topContributors.length).toBe(3);
    // Skill discovery should be top contributor
    expect(snapshot.topContributors[0].name).toBe('skillDiscovery');
  });

  it('EMA smooths values over time', () => {
    const rb = new RewardBreakdown();
    // Record several ticks
    for (let i = 0; i < 20; i++) {
      rb.record(i, {
        survival: 0.1,
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
      });
    }
    const snapshot = rb.latest();
    // EMA of survival should approach 0.1
    expect(snapshot.ema.survival).toBeGreaterThan(0.05);
    expect(snapshot.ema.survival).toBeLessThanOrEqual(0.1);
  });
});

describe('Living Mode - Intent System', () => {
  it('agent switches intent based on needs', () => {
    const engine = new LiveModeEngine({
      seed: 42,
      populationSize: 1,
      ticksPerSecond: 20,
      deterministic: true,
      rollingSeconds: 30,
      livingMode: true,
    });
    // Run for enough ticks to see intent changes
    const intents = new Set<string>();
    for (let i = 0; i < 200; i++) {
      const result = engine.tickOnce();
      intents.add(result.agentIntent);
    }
    // Should have at least 2 different intents
    expect(intents.size).toBeGreaterThanOrEqual(2);
  });
});

describe('Living Mode - Living Loop', () => {
  it('agents do not immediately collapse to idle or infinite spam', () => {
    const engine = new LiveModeEngine({
      seed: 1337,
      populationSize: 1,
      ticksPerSecond: 20,
      deterministic: true,
      rollingSeconds: 30,
      livingMode: true,
    });
    let idleCount = 0;
    const actionCounts = new Map<string, number>();
    for (let i = 0; i < 300; i++) {
      const result = engine.tickOnce();
      if (result.action === 'REST') idleCount++;
      actionCounts.set(result.action, (actionCounts.get(result.action) ?? 0) + 1);
    }
    // Not all idle
    expect(idleCount).toBeLessThan(250);
    // Not single action spam
    expect(actionCounts.size).toBeGreaterThan(1);
  });

  it('deterministic mode produces identical results', () => {
    const cfg = {
      seed: 42,
      populationSize: 1,
      ticksPerSecond: 20,
      deterministic: true,
      rollingSeconds: 30,
      livingMode: true,
    } as const;
    const engine1 = new LiveModeEngine(cfg);
    const engine2 = new LiveModeEngine(cfg);
    for (let i = 0; i < 50; i++) {
      const r1 = engine1.tickOnce();
      const r2 = engine2.tickOnce();
      expect(r1.action).toBe(r2.action);
      expect(r1.woodPerMinute).toBeCloseTo(r2.woodPerMinute, 6);
    }
  });
});

describe('Living Mode - Property Axes', () => {
  it('new property axes exist on world objects', () => {
    const engine = new LiveModeEngine({
      seed: 42,
      populationSize: 1,
      ticksPerSecond: 20,
      deterministic: true,
      rollingSeconds: 30,
    });
    const obj = [...engine.world.objects.values()][0];
    expect(obj.props.conductivity).toBeDefined();
    expect(obj.props.malleability).toBeDefined();
    expect(obj.props.porosity).toBeDefined();
    expect(obj.props.conductivity).toBeGreaterThanOrEqual(0);
    expect(obj.props.conductivity).toBeLessThanOrEqual(1);
  });
});
