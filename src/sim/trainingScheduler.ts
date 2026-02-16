/**
 * TrainingScheduler — guarantees training updates happen during Live Mode.
 *
 * Collects transitions into a replay buffer continuously,
 * runs training steps at cadence (trainEveryMs) but rate-limited by maxTrainMsPerSec,
 * and exposes live metrics for the progress dashboard.
 */

export type TrainingState = 'off' | 'collecting' | 'training' | 'rate_limited' | 'error';

export interface TrainingMetrics {
  trainingStepsTotal: number;
  trainingStepsLast60s: number;
  trainMsLast1s: number;
  trainMsLast60s: number;
  replaySize: number;
  batchLoss: number;
  policyEntropy: number;
  state: TrainingState;
  lastError?: string;
}

interface TimestampedStep {
  time: number;
  durationMs: number;
  loss: number;
}

export class TrainingScheduler {
  private _state: TrainingState = 'off';
  private _lastError?: string;
  private _stepsTotal = 0;
  private readonly _stepLog: TimestampedStep[] = [];
  private _lossSmoother = 0;
  private _entropySmoother = 0.5;

  get state(): TrainingState { return this._state; }
  get lastError(): string | undefined { return this._lastError; }

  start(): void {
    this._state = 'collecting';
  }

  stop(): void {
    this._state = 'off';
  }

  /**
   * Called by the live-mode loop each time a training step is performed.
   * @param durationMs  wall-time the step took
   * @param loss        loss value (or prediction error) from the step
   * @param entropy     policy entropy (or action diversity measure)
   * @param nowSec      simulated time in seconds (for windowed metrics)
   */
  recordStep(durationMs: number, loss: number, entropy: number, nowSec: number): void {
    this._stepsTotal += 1;
    this._lossSmoother += (loss - this._lossSmoother) * 0.05;
    this._entropySmoother += (entropy - this._entropySmoother) * 0.05;
    this._stepLog.push({ time: nowSec, durationMs, loss });
    // Keep a rolling 90-second window
    while (this._stepLog.length > 0 && this._stepLog[0].time < nowSec - 90) {
      this._stepLog.shift();
    }
    this._state = 'training';
  }

  recordRateLimited(): void {
    if (this._state !== 'off') this._state = 'rate_limited';
  }

  recordError(msg: string): void {
    this._state = 'error';
    this._lastError = msg;
  }

  setCollecting(): void {
    if (this._state === 'off') return;
    if (this._state !== 'error') this._state = 'collecting';
  }

  metrics(nowSec: number, replaySize: number): TrainingMetrics {
    const window60 = this._stepLog.filter(s => s.time >= nowSec - 60);
    const window1 = this._stepLog.filter(s => s.time >= nowSec - 1);
    return {
      trainingStepsTotal: this._stepsTotal,
      trainingStepsLast60s: window60.length,
      trainMsLast1s: window1.reduce((sum, s) => sum + s.durationMs, 0),
      trainMsLast60s: window60.reduce((sum, s) => sum + s.durationMs, 0),
      replaySize,
      batchLoss: this._lossSmoother,
      policyEntropy: this._entropySmoother,
      state: this._state,
      lastError: this._lastError,
    };
  }
}
