import { clamp } from '../sim/properties';

export type ModelActionVerb = 'BIND_TO' | 'STRIKE_WITH' | 'GRIND';

export interface ModelObjectFeatures {
  visual_features: number;
  mass_estimate: number;
  length_estimate: number;
  texture_proxy: number;
  interaction_feedback_history: number;
}

export interface WorldModelInput {
  objectA: ModelObjectFeatures;
  objectB: ModelObjectFeatures;
  geometry_features: number;
  action_verb: ModelActionVerb;
  relative_position: number;
}

export interface WorldModelPrediction {
  expected_damage: number;
  expected_tool_wear: number;
  expected_fragments: number;
  expected_property_changes: number;
}

export type WorldModelOutcome = WorldModelPrediction;

function actionOneHot(action: ModelActionVerb): [number, number, number] {
  if (action === 'STRIKE_WITH') return [1, 0, 0];
  if (action === 'GRIND') return [0, 1, 0];
  return [0, 0, 1];
}

export class WorldModel {
  private readonly weights: number[][];
  private readonly bias: number[];
  private readonly seen = new Map<string, number>();
  private runningPredictionError = 0;
  private updates = 0;
  private frozen = false;

  constructor() {
    const featureSize = 15;
    this.weights = Array.from({ length: 4 }, () => Array.from({ length: featureSize }, () => 0));
    this.bias = [0, 0, 0, 0];
  }

  private featurize(input: WorldModelInput): number[] {
    const a = input.objectA;
    const b = input.objectB;
    return [
      a.visual_features,
      a.mass_estimate,
      a.length_estimate,
      a.texture_proxy,
      a.interaction_feedback_history,
      b.visual_features,
      b.mass_estimate,
      b.length_estimate,
      b.texture_proxy,
      b.interaction_feedback_history,
      input.geometry_features,
      input.relative_position,
      ...actionOneHot(input.action_verb),
    ];
  }

  private keyOf(input: WorldModelInput): string {
    const q = (v: number): string => Math.round(clamp(v) * 6).toString(10);
    return `${input.action_verb}:${q(input.objectA.mass_estimate)}:${q(input.objectB.mass_estimate)}:${q(input.geometry_features)}`;
  }

  private project(x: number[], rowIndex: number): number {
    const raw = this.weights[rowIndex].reduce((acc, w, idx) => acc + w * x[idx], this.bias[rowIndex]);
    return Math.max(0, raw);
  }

  predict(input: WorldModelInput): WorldModelPrediction {
    const x = this.featurize(input);
    const out = this.weights.map((_, i) => this.project(x, i));
    return {
      expected_damage: out[0],
      expected_tool_wear: out[1],
      expected_fragments: out[2],
      expected_property_changes: out[3],
    };
  }

  novelty(input: WorldModelInput): number {
    const seenCount = this.seen.get(this.keyOf(input)) ?? 0;
    return 1 / Math.sqrt(seenCount + 1);
  }

  setFrozen(frozen: boolean): void {
    this.frozen = frozen;
  }

  update(input: WorldModelInput, actual: WorldModelOutcome, lr = 0.12): number {
    const x = this.featurize(input);
    const pred = this.predict(input);
    const predVec = [pred.expected_damage, pred.expected_tool_wear, pred.expected_fragments, pred.expected_property_changes];
    const target = [actual.expected_damage, actual.expected_tool_wear, actual.expected_fragments, actual.expected_property_changes];
    const absErrors = predVec.map((p, i) => Math.abs(target[i] - p));
    const predictionError = absErrors.reduce((a, b) => a + b, 0) / absErrors.length;
    const scaledLr = lr * (0.4 + this.novelty(input) * 0.6);

    if (!this.frozen) {
      for (let i = 0; i < this.weights.length; i++) {
        const diff = target[i] - predVec[i];
        for (let j = 0; j < x.length; j++) this.weights[i][j] += scaledLr * diff * x[j];
        this.bias[i] += scaledLr * diff;
      }

      const key = this.keyOf(input);
      this.seen.set(key, (this.seen.get(key) ?? 0) + 1);
    }

    this.updates += 1;
    this.runningPredictionError += (predictionError - this.runningPredictionError) / this.updates;
    return predictionError;
  }

  meanPredictionError(): number {
    return this.runningPredictionError;
  }
}
