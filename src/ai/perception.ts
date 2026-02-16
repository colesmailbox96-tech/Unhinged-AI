import type { WorldObject } from '../sim/object_model';
import type { PropertyVector } from '../sim/properties';
import { clamp } from '../sim/properties';
import { RNG } from '../sim/rng';

export interface Observation {
  observed_length: number;
  observed_mass_estimate: number;
  visual_symmetry: number;
  contact_area_estimate: number;
  texture_proxy: number;
  interaction_feedback_history: number;
}

export interface HiddenPrediction {
  hardness: number;
  brittleness: number;
  sharpness: number;
  uncertainty: number;
}

const targets: (keyof HiddenPrediction)[] = ['hardness', 'brittleness', 'sharpness'];

export class PerceptionHead {
  private readonly weights: number[][];
  private readonly bias: number[];
  private experience = 0;

  constructor(seed = 123) {
    const rng = new RNG(seed);
    this.weights = Array.from({ length: targets.length }, () => [
      rng.range(-0.1, 0.1),
      rng.range(-0.1, 0.1),
      rng.range(-0.1, 0.1),
      rng.range(-0.1, 0.1),
      rng.range(-0.1, 0.1),
      rng.range(-0.1, 0.1),
    ]);
    this.bias = [0, 0, 0];
  }

  observe(obj: WorldObject, rng: RNG): Observation {
    const noiseScale = Math.max(0.02, 0.15 / Math.sqrt(this.experience + 1));
    const maxOffset = Math.max(0.1, obj.length * 0.5 + obj.thickness * 0.5);
    const offsetMag = Math.hypot(obj.center_of_mass_offset.x, obj.center_of_mass_offset.y);
    const shapeAsymmetry = obj.shapeType === 'shard' ? 0.25 : obj.shapeType === 'rod' ? 0.1 : 0.05;
    return {
      observed_length: clamp(obj.length / 2.2 + rng.normal(0, noiseScale * 0.8)),
      observed_mass_estimate: clamp(obj.props.mass * 0.8 + obj.thickness * 0.2 + rng.normal(0, noiseScale)),
      visual_symmetry: clamp(1 - offsetMag / maxOffset - shapeAsymmetry + rng.normal(0, noiseScale * 0.7)),
      contact_area_estimate: clamp((obj.length * obj.thickness + Math.PI * obj.radius * obj.radius) / 3 + rng.normal(0, noiseScale)),
      texture_proxy: clamp(obj.props.roughness * 0.7 + obj.props.friction_coeff * 0.3 + rng.normal(0, noiseScale * 0.5)),
      interaction_feedback_history: clamp(this.experience / (this.experience + 12)),
    };
  }

  predict(obs: Observation): HiddenPrediction {
    const x = [
      obs.observed_length,
      obs.observed_mass_estimate,
      obs.visual_symmetry,
      obs.contact_area_estimate,
      obs.texture_proxy,
      obs.interaction_feedback_history,
    ];
    const outputs = this.weights.map((row, i) => clamp(row.reduce((acc, w, idx) => acc + w * x[idx], this.bias[i])));
    return {
      hardness: outputs[0],
      brittleness: outputs[1],
      sharpness: outputs[2],
      uncertainty: 1 / Math.sqrt(this.experience + 1),
    };
  }

  train(obs: Observation, truth: PropertyVector, outcomeSignal: number, lr = 0.08): void {
    const x = [
      obs.observed_length,
      obs.observed_mass_estimate,
      obs.visual_symmetry,
      obs.contact_area_estimate,
      obs.texture_proxy,
      obs.interaction_feedback_history,
    ];
    const y = [truth.hardness, truth.brittleness, truth.sharpness];
    const pred = this.predict(obs);
    const p = [pred.hardness, pred.brittleness, pred.sharpness];
    const scaledLr = lr * (0.3 + outcomeSignal * 0.7);

    for (let i = 0; i < this.weights.length; i++) {
      const error = p[i] - y[i];
      for (let j = 0; j < x.length; j++) {
        this.weights[i][j] -= scaledLr * error * x[j];
      }
      this.bias[i] -= scaledLr * error;
    }

    this.experience += 1;
  }

  hardnessError(objects: WorldObject[], rng: RNG): number {
    const errors = objects.map((obj) => {
      const obs = this.observe(obj, rng);
      const pred = this.predict(obs);
      return Math.abs(pred.hardness - obj.props.hardness);
    });
    return errors.reduce((a, b) => a + b, 0) / Math.max(1, errors.length);
  }

  snapshot(): {
    weights: number[][];
    bias: number[];
    experience: number;
  } {
    return {
      weights: this.weights.map((row) => [...row]),
      bias: [...this.bias],
      experience: this.experience,
    };
  }
}
