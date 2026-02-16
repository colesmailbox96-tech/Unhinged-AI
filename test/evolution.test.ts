import { describe, expect, test } from 'vitest';
import { StallDetector } from '../src/sim/stallDetector';
import { PopulationController } from '../src/sim/populationController';
import { TrainingScheduler } from '../src/sim/trainingScheduler';

describe('StallDetector', () => {
  test('does not trigger stall in first few ticks', () => {
    const detector = new StallDetector();
    const triggered = detector.record(1, 1, 'STRIKE_WITH', 0.1, 10, 20);
    expect(triggered).toBe(false);
    expect(detector.isInForcedExplore(1)).toBe(false);
  });

  test('triggers stall on repeated same action with no reward change', () => {
    const detector = new StallDetector();
    let triggered = false;
    for (let i = 0; i < 50; i++) {
      triggered = detector.record(1, i, 'STRIKE_WITH', 0.0, undefined, undefined);
      if (triggered) break;
    }
    expect(triggered).toBe(true);
    expect(detector.isInForcedExplore(1)).toBe(true);
  });

  test('does not trigger stall with diverse actions and reward changes', () => {
    const detector = new StallDetector();
    const actions = ['STRIKE_WITH', 'PICK_UP', 'MOVE_TO', 'GRIND', 'BIND_TO', 'DROP'];
    let triggered = false;
    for (let i = 0; i < 40; i++) {
      triggered = detector.record(1, i, actions[i % actions.length], i * 0.1, i % 5, i % 3);
      if (triggered) break;
    }
    expect(triggered).toBe(false);
  });

  test('forced explore cooldown expires after duration', () => {
    const detector = new StallDetector();
    // Force a stall
    for (let i = 0; i < 50; i++) {
      detector.record(1, i, 'STRIKE_WITH', 0.0, undefined, undefined);
    }
    expect(detector.isInForcedExplore(1)).toBe(true);
    // Tick through the forced explore duration with diverse actions and rewards
    const actions = ['PICK_UP', 'MOVE_TO', 'GRIND', 'BIND_TO', 'STRIKE_WITH', 'DROP'];
    for (let i = 50; i < 65; i++) {
      detector.record(1, i, actions[i % actions.length], 0.5 + i * 0.01, i, i);
    }
    expect(detector.isInForcedExplore(1)).toBe(false);
  });

  test('metrics report stall events and time correctly', () => {
    const detector = new StallDetector();
    // Trigger a stall
    for (let i = 0; i < 50; i++) {
      detector.record(1, i, 'STRIKE_WITH', 0.0, undefined, undefined);
    }
    const m = detector.metrics(1);
    expect(m.stallEventsPerMin).toBeGreaterThan(0);
    expect(m.isStalled).toBe(true);
  });
});

describe('PopulationController', () => {
  test('spawn throttle is high when counts are low', () => {
    const ctrl = new PopulationController({ minTargets: 5, maxTargets: 30, maxObjects: 200 });
    const result = ctrl.evaluate({ targetsAlive: 3, objectsTotal: 20, fragmentsTotal: 2 });
    expect(result.spawnProbability).toBe(1); // below minTargets
    expect(result.debrisToClean).toBe(0);
  });

  test('spawn throttle decreases as objects approach max', () => {
    const ctrl = new PopulationController({ maxObjects: 100, maxTargets: 40 });
    const result = ctrl.evaluate({ targetsAlive: 15, objectsTotal: 80, fragmentsTotal: 10 });
    expect(result.spawnProbability).toBeLessThan(0.5);
    expect(result.debrisToClean).toBe(0);
  });

  test('debris cleanup triggered when objects exceed max', () => {
    const ctrl = new PopulationController({ maxObjects: 50 });
    const result = ctrl.evaluate({ targetsAlive: 10, objectsTotal: 70, fragmentsTotal: 30 });
    expect(result.debrisToClean).toBeGreaterThan(0);
    expect(result.debrisToClean).toBeLessThanOrEqual(30);
  });

  test('debris cleanup triggered when fragment ratio exceeds max', () => {
    const ctrl = new PopulationController({ maxObjects: 200, maxFragmentRatio: 0.3 });
    const result = ctrl.evaluate({ targetsAlive: 15, objectsTotal: 100, fragmentsTotal: 50 });
    // 50/100=0.5 > 0.3, should cleanup excess
    expect(result.debrisToClean).toBeGreaterThan(0);
  });

  test('purgedNonDebrisPerMin is always 0', () => {
    const ctrl = new PopulationController();
    ctrl.evaluate({ targetsAlive: 10, objectsTotal: 500, fragmentsTotal: 400 });
    expect(ctrl.metrics().purgedNonDebrisPerMin).toBe(0);
  });
});

describe('TrainingScheduler', () => {
  test('starts in off state', () => {
    const scheduler = new TrainingScheduler();
    const m = scheduler.metrics(0, 0);
    expect(m.state).toBe('off');
    expect(m.trainingStepsTotal).toBe(0);
  });

  test('transitions to collecting after start', () => {
    const scheduler = new TrainingScheduler();
    scheduler.start();
    expect(scheduler.state).toBe('collecting');
  });

  test('records training steps and updates metrics', () => {
    const scheduler = new TrainingScheduler();
    scheduler.start();
    scheduler.recordStep(5, 0.5, 0.8, 10);
    scheduler.recordStep(3, 0.4, 0.7, 11);
    const m = scheduler.metrics(12, 100);
    expect(m.trainingStepsTotal).toBe(2);
    expect(m.trainingStepsLast60s).toBe(2);
    expect(m.state).toBe('training');
    expect(m.batchLoss).toBeGreaterThan(0);
    expect(m.replaySize).toBe(100);
  });

  test('rate limited state is set correctly', () => {
    const scheduler = new TrainingScheduler();
    scheduler.start();
    scheduler.recordStep(5, 0.5, 0.8, 10);
    scheduler.recordRateLimited();
    expect(scheduler.state).toBe('rate_limited');
  });

  test('error state is set correctly', () => {
    const scheduler = new TrainingScheduler();
    scheduler.start();
    scheduler.recordError('test error');
    expect(scheduler.state).toBe('error');
    expect(scheduler.lastError).toBe('test error');
  });

  test('windowed metrics expire old entries', () => {
    const scheduler = new TrainingScheduler();
    scheduler.start();
    scheduler.recordStep(5, 0.5, 0.8, 1);
    scheduler.recordStep(3, 0.4, 0.7, 100);
    const m = scheduler.metrics(100, 50);
    expect(m.trainingStepsLast60s).toBe(1); // first step at t=1 is older than 60s window
    expect(m.trainingStepsTotal).toBe(2);
  });
});
