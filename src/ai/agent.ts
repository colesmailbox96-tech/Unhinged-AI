import { PerceptionHead, type Observation } from './perception';
import { ToolEmbedding } from './tool_embedding';
import { WorldModel, type WorldModelInput, type WorldModelPrediction } from './world_model';
import { World } from '../sim/world';
import type { ObjID, WorldObject } from '../sim/object_model';

export type Strategy = 'RANDOM_STRIKE' | 'BIND_THEN_STRIKE';

export interface EpisodeResult {
  woodGained: number;
  woodPerMinute: number;
  hardnessMaeBefore: number;
  hardnessMaeAfter: number;
  predictionErrorMean: number;
  noveltyCount: number;
  compositeDiscoveryRate: number;
  embeddingClusters: number;
  novelInteractionCount: number;
  predictionSnapshot?: PredictionSnapshot;
  embeddingSnapshot: EmbeddingSnapshot[];
  trace: string[];
  replaySignature: string;
  predictedStrikeArc?: ArcSnapshot;
  actualStrikeArc?: ArcSnapshot;
  logs: string[];
}

export interface PredictionSnapshot {
  action: WorldModelInput['action_verb'];
  predicted: Pick<WorldModelPrediction, 'expected_damage' | 'expected_tool_wear' | 'expected_fragments'>;
  actual: { damage: number; toolWear: number; fragments: number };
  error: { damage: number; toolWear: number; fragments: number };
}

export interface EmbeddingSnapshot {
  toolId: number;
  vector: [number, number, number, number];
  point: { x: number; y: number };
}

export interface RunEpisodeOptions {
  freezeWorldModel?: boolean;
  disablePredictionModel?: boolean;
  collectTrace?: boolean;
}

export interface ArcSnapshot {
  center: { x: number; y: number };
  radius: number;
  start: number;
  end: number;
}

interface CandidateAction {
  verb: 'PICK_UP' | 'BIND_TO' | 'STRIKE_WITH' | 'GRIND';
  objId?: ObjID;
  targetId?: ObjID;
  modelInput?: WorldModelInput;
  predicted?: WorldModelPrediction;
  score?: number;
}

const UTILITY_WEIGHTS = {
  damage: 0.4,
  fragments: 0.25,
  propertyChanges: 0.2,
  toolWear: 0.15,
} as const;

function chooseByPerception(world: World, perception: PerceptionHead, avoid: ObjID[] = []): WorldObject[] {
  const hidden = world.getNearbyObjectIds().filter((id) => !avoid.includes(id)).map((id) => world.objects.get(id)).filter((obj): obj is WorldObject => Boolean(obj));

  return hidden.sort((a, b) => {
    const oa = perception.observe(a, world.rng);
    const ob = perception.observe(b, world.rng);
    const pa = perception.predict(oa);
    const pb = perception.predict(ob);
    const sa = pa.hardness * 0.25 + oa.observed_length * oa.observed_mass_estimate * (1.2 - oa.visual_symmetry) + oa.contact_area_estimate * 0.1;
    const sb = pb.hardness * 0.25 + ob.observed_length * ob.observed_mass_estimate * (1.2 - ob.visual_symmetry) + ob.contact_area_estimate * 0.1;
    return sb - sa;
  });
}

function toModelFeatures(obs: Observation): WorldModelInput['objectA'] {
  return {
    visual_features: (obs.visual_symmetry + obs.contact_area_estimate) * 0.5,
    mass_estimate: obs.observed_mass_estimate,
    length_estimate: obs.observed_length,
    texture_proxy: obs.texture_proxy,
    interaction_feedback_history: obs.interaction_feedback_history,
  };
}

function buildModelInput(
  world: World,
  perception: PerceptionHead,
  action_verb: WorldModelInput['action_verb'],
  a: WorldObject,
  b: WorldObject,
): WorldModelInput {
  const oa = perception.observe(a, world.rng);
  const ob = perception.observe(b, world.rng);
  const dx = b.pos.x - a.pos.x;
  const dy = b.pos.y - a.pos.y;
  return {
    action_verb,
    objectA: toModelFeatures(oa),
    objectB: toModelFeatures(ob),
    geometry_features: Math.max(0, Math.min(1, (a.length / Math.max(0.1, a.thickness) + b.length / Math.max(0.1, b.thickness)) / 12)),
    relative_position: Math.max(0, Math.min(1, Math.hypot(dx, dy) / 3)),
  };
}

function chooseCandidate(
  world: World,
  strategy: Strategy,
  candidates: CandidateAction[],
  worldModel: WorldModel,
  disablePredictionModel = false,
): CandidateAction | undefined {
  if (!candidates.length) return undefined;
  if (strategy === 'RANDOM_STRIKE' || disablePredictionModel) return candidates[world.rng.int(0, candidates.length)];
  let best = candidates[0];
  for (const candidate of candidates) {
    if (!candidate.modelInput || !candidate.predicted) continue;
    const curiosity = worldModel.novelty(candidate.modelInput);
    const utility =
      candidate.predicted.expected_damage * UTILITY_WEIGHTS.damage +
      candidate.predicted.expected_fragments * UTILITY_WEIGHTS.fragments +
      candidate.predicted.expected_property_changes * UTILITY_WEIGHTS.propertyChanges -
      candidate.predicted.expected_tool_wear * UTILITY_WEIGHTS.toolWear;
    candidate.score = curiosity + utility;
    if ((candidate.score ?? -Infinity) > (best.score ?? -Infinity)) best = candidate;
  }
  return best;
}

function pca2D(vectors: [number, number, number, number][]): Array<{ x: number; y: number }> {
  if (!vectors.length) return [];
  const mean = [0, 0, 0, 0];
  for (const v of vectors) for (let i = 0; i < 4; i++) mean[i] += v[i];
  for (let i = 0; i < 4; i++) mean[i] /= vectors.length;
  const centered = vectors.map((v) => v.map((value, i) => value - mean[i]) as [number, number, number, number]);
  const cov = Array.from({ length: 4 }, () => [0, 0, 0, 0]);
  for (const v of centered) {
    for (let r = 0; r < 4; r++) for (let c = 0; c < 4; c++) cov[r][c] += v[r] * v[c];
  }
  const scale = 1 / Math.max(1, centered.length - 1);
  for (let r = 0; r < 4; r++) for (let c = 0; c < 4; c++) cov[r][c] *= scale;
  const projectAxis = (seedVec: [number, number, number, number], orthTo?: [number, number, number, number]): [number, number, number, number] => {
    let vec = [...seedVec] as [number, number, number, number];
    for (let iter = 0; iter < 6; iter++) {
      const next: [number, number, number, number] = [0, 0, 0, 0];
      for (let r = 0; r < 4; r++) for (let c = 0; c < 4; c++) next[r] += cov[r][c] * vec[c];
      if (orthTo) {
        const dot = next[0] * orthTo[0] + next[1] * orthTo[1] + next[2] * orthTo[2] + next[3] * orthTo[3];
        for (let i = 0; i < 4; i++) next[i] -= dot * orthTo[i];
      }
      const mag = Math.hypot(...next) || 1;
      vec = [next[0] / mag, next[1] / mag, next[2] / mag, next[3] / mag];
    }
    return vec;
  };
  const axisX = projectAxis([1, 1, 0.5, 0.25]);
  const axisY = projectAxis([0.25, -0.5, 1, -1], axisX);
  return centered.map((v) => ({
    x: v[0] * axisX[0] + v[1] * axisX[1] + v[2] * axisX[2] + v[3] * axisX[3],
    y: v[0] * axisY[0] + v[1] * axisY[1] + v[2] * axisY[2] + v[3] * axisY[3],
  }));
}

export function runEpisode(
  seed: number,
  strategy: Strategy,
  perception = new PerceptionHead(seed + 77),
  steps = 35,
  options: RunEpisodeOptions = {},
): EpisodeResult {
  const world = new World(seed);
  const worldModel = new WorldModel();
  worldModel.setFrozen(Boolean(options.freezeWorldModel));
  const embedding = new ToolEmbedding();
  const initialObjects = [...world.objects.values()];
  const hardnessMaeBefore = perception.hardnessError(initialObjects, world.rng.clone());
  const predictionErrors: number[] = [];
  let noveltyCount = 0;
  let compositeDiscoveries = 0;
  let predictionSnapshot: PredictionSnapshot | undefined;
  const trace: string[] = [];

  const pickables = chooseByPerception(world, perception);
  if (pickables[0]) world.apply({ type: 'PICK_UP', objId: pickables[0].id });

  for (let i = 0; i < steps; i++) {
    const held = world.agent.heldObjectId ? world.objects.get(world.agent.heldObjectId) : undefined;
    const targetId = world.getTargetId();
    const target = targetId ? world.objects.get(targetId) : undefined;
    const candidates: CandidateAction[] = [];

    if (!held) {
      for (const candidate of chooseByPerception(world, perception).slice(0, 3)) {
        candidates.push({ verb: 'PICK_UP', objId: candidate.id, score: 0.05 });
      }
    } else {
      if (target && target.id !== held.id) {
        const modelInput = buildModelInput(world, perception, 'STRIKE_WITH', held, target);
        const predicted = worldModel.predict(modelInput);
        const angle = Math.atan2(target.pos.y - held.pos.y, target.pos.x - held.pos.x);
        world.predictedStrikeArc = {
          center: { ...held.pos },
          radius: Math.max(0.4, held.length * 0.5),
          start: angle - 0.55,
          end: angle + 0.25,
          alpha: 0.45,
        };
        world.predictedStrikeDamage = predicted.expected_damage;
        candidates.push({ verb: 'STRIKE_WITH', targetId: target.id, modelInput, predicted });
      }

      for (const nearby of chooseByPerception(world, perception, [held.id]).slice(0, 2)) {
        const bindInput = buildModelInput(world, perception, 'BIND_TO', held, nearby);
        candidates.push({
          verb: 'BIND_TO',
          objId: nearby.id,
          modelInput: bindInput,
          predicted: worldModel.predict(bindInput),
        });

        const grindInput = buildModelInput(world, perception, 'GRIND', held, nearby);
        candidates.push({
          verb: 'GRIND',
          objId: nearby.id,
          modelInput: grindInput,
          predicted: worldModel.predict(grindInput),
        });
      }
    }

    const chosen = chooseCandidate(world, strategy, candidates, worldModel, Boolean(options.disablePredictionModel));
    if (!chosen) continue;
    if (options.collectTrace) trace.push(`step=${i} action=${chosen.verb}`);

    if (chosen.verb === 'PICK_UP' && chosen.objId) world.apply({ type: 'PICK_UP', objId: chosen.objId });
    if (chosen.verb === 'BIND_TO' && chosen.objId) {
      world.apply({ type: 'BIND_TO', objId: chosen.objId });
      compositeDiscoveries += 1;
    }
    if (chosen.verb === 'GRIND' && chosen.objId) world.apply({ type: 'GRIND', abrasiveId: chosen.objId });
    if (chosen.verb === 'STRIKE_WITH' && chosen.targetId) world.apply({ type: 'STRIKE_WITH', targetId: chosen.targetId });

    const outcome = world.lastInteractionOutcome;
    if (outcome?.toolId) {
      embedding.update(outcome.toolId, {
        damage: outcome.damage,
        toolWear: outcome.toolWear,
        fragments: outcome.fragments,
        propertyChanges: outcome.propertyChanges,
      });
    }

    if (chosen.modelInput && strategy !== 'RANDOM_STRIKE' && outcome) {
      predictionSnapshot = {
        action: chosen.modelInput.action_verb,
        predicted: {
          expected_damage: chosen.predicted?.expected_damage ?? 0,
          expected_tool_wear: chosen.predicted?.expected_tool_wear ?? 0,
          expected_fragments: chosen.predicted?.expected_fragments ?? 0,
        },
        actual: {
          damage: outcome.damage,
          toolWear: outcome.toolWear,
          fragments: outcome.fragments,
        },
        error: {
          damage: Math.abs((chosen.predicted?.expected_damage ?? 0) - outcome.damage),
          toolWear: Math.abs((chosen.predicted?.expected_tool_wear ?? 0) - outcome.toolWear),
          fragments: Math.abs((chosen.predicted?.expected_fragments ?? 0) - outcome.fragments),
        },
      };
      const predictionError = worldModel.update(chosen.modelInput, {
        expected_damage: outcome.damage,
        expected_tool_wear: outcome.toolWear,
        expected_fragments: outcome.fragments,
        expected_property_changes: outcome.propertyChanges,
      });
      predictionErrors.push(predictionError);
      if (predictionError > 0.08) noveltyCount += 1;
    }
    if (options.collectTrace && outcome) {
      trace.push(
        `step=${i} outcome=${outcome.action} dmg=${outcome.damage.toFixed(3)} wear=${outcome.toolWear.toFixed(3)} frags=${outcome.fragments}`,
      );
    }

    const heldAfter = world.agent.heldObjectId ? world.objects.get(world.agent.heldObjectId) : undefined;
    if (heldAfter) {
      const obs = perception.observe(heldAfter, world.rng);
      perception.train(obs, heldAfter.props, 1);
    }
  }

  const hardnessMaeAfter = perception.hardnessError([...world.objects.values()], world.rng.clone());
  const embeddingEntries = embedding.entries();
  const embeddingPoints = pca2D(embeddingEntries.map((entry) => entry.vector));
  const embeddingSnapshot = embeddingEntries.map((entry, idx) => ({ toolId: entry.toolId, vector: entry.vector, point: embeddingPoints[idx] ?? { x: 0, y: 0 } }));
  const replaySignature = trace.join('|');
  return {
    woodGained: world.woodGained,
    woodPerMinute: world.woodGained / (steps / 60),
    hardnessMaeBefore,
    hardnessMaeAfter,
    predictionErrorMean: predictionErrors.reduce((a, b) => a + b, 0) / Math.max(1, predictionErrors.length),
    noveltyCount,
    compositeDiscoveryRate: compositeDiscoveries / Math.max(1, steps),
    embeddingClusters: embedding.clusterCount(),
    novelInteractionCount: embedding.novelInteractionCount(),
    predictionSnapshot,
    embeddingSnapshot,
    trace,
    replaySignature,
    predictedStrikeArc: world.predictedStrikeArc
      ? { center: world.predictedStrikeArc.center, radius: world.predictedStrikeArc.radius, start: world.predictedStrikeArc.start, end: world.predictedStrikeArc.end }
      : undefined,
    actualStrikeArc: world.lastStrikeArc
      ? { center: world.lastStrikeArc.center, radius: world.lastStrikeArc.radius, start: world.lastStrikeArc.start, end: world.lastStrikeArc.end }
      : undefined,
    logs: world.logs.slice(0, 25),
  };
}
