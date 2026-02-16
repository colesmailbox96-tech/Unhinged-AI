import { PerceptionHead, type Observation } from '../ai/perception';
import { ReplayBuffer } from '../ai/replay_buffer';
import { ToolEmbedding } from '../ai/tool_embedding';
import { WorldModel, type WorldModelInput, type WorldModelOutcome, type WorldModelPrediction } from '../ai/world_model';
import type { InteractionOutcome } from '../sim/world';
import { World } from '../sim/world';
import type { ObjID, WorldObject } from '../sim/object_model';
import { MilestoneTracker, type MilestoneEvent } from '../sim/milestones';

type LiveVerb = 'PICK_UP' | 'BIND_TO' | 'STRIKE_WITH' | 'GRIND' | 'REST';

interface CandidateAction {
  verb: LiveVerb;
  objId?: ObjID;
  targetId?: ObjID;
  modelInput?: WorldModelInput;
  predicted?: WorldModelPrediction;
  score?: number;
}

interface ReplayTransition {
  input: WorldModelInput;
  outcome: WorldModelOutcome;
  toolId?: number;
}

export interface LiveModeConfig {
  seed: number;
  populationSize: number;
  ticksPerSecond: number;
  deterministic: boolean;
  rollingSeconds: number;
}

export interface LiveTrainingConfig {
  trainEveryMs: number;
  batchSize: number;
  maxTrainMsPerSecond: number;
  stepsPerTick: number;
}

export interface LiveAgentMemory {
  id: number;
  energy: number;
  observations: Observation[];
  actions: LiveVerb[];
  outcomes: number[];
  goodTools: number[];
  goodLocations: Array<{ x: number; y: number; score: number }>;
}

export interface LiveTickResult {
  tick: number;
  simTimeSeconds: number;
  activeAgentId: number;
  action: LiveVerb;
  predictionErrorMean: number;
  woodPerMinute: number;
  novelInteractionsPerMinute: number;
  compositeDiscoveryRate: number;
  embeddingClusters: number;
  milestones: MilestoneEvent[];
}

export interface SegmentFrame {
  tick: number;
  simTimeSeconds: number;
  action: LiveVerb;
  replaySignature: string;
  objectIds: number[];
}

export interface LiveBookmark {
  id: string;
  createdAtTick: number;
  createdAtSeconds: number;
  frames: SegmentFrame[];
}

export interface LiveSnapshot {
  seed: number;
  tick: number;
  simTimeSeconds: number;
  metrics: {
    woodPerMinute: number;
    predictionErrorMean: number;
    novelInteractionsPerMinute: number;
    compositeDiscoveryRate: number;
    embeddingClusters: number;
  };
  milestones: MilestoneEvent[];
  agents: LiveAgentMemory[];
  world: {
    woodGained: number;
    objects: WorldObject[];
  };
  modelState: {
    worldModel: ReturnType<WorldModel['snapshot']>;
    perception: ReturnType<PerceptionHead['snapshot']>;
    embedding: ReturnType<ToolEmbedding['snapshot']>;
  };
}

const UTILITY_WEIGHTS = {
  damage: 0.4,
  fragments: 0.25,
  propertyChanges: 0.2,
  toolWear: 0.15,
} as const;

function chooseByPerception(world: World, perception: PerceptionHead, avoid: ObjID[] = []): WorldObject[] {
  const hidden = world
    .getNearbyObjectIds()
    .filter((id) => !avoid.includes(id))
    .map((id) => world.objects.get(id))
    .filter((obj): obj is WorldObject => Boolean(obj));
  return hidden.sort((a, b) => {
    const oa = perception.observe(a, world.rng);
    const ob = perception.observe(b, world.rng);
    const pa = perception.predict(oa);
    const pb = perception.predict(ob);
    const sa =
      pa.hardness * 0.25 + oa.observed_length * oa.observed_mass_estimate * (1.2 - oa.visual_symmetry) + oa.contact_area_estimate * 0.1;
    const sb =
      pb.hardness * 0.25 + ob.observed_length * ob.observed_mass_estimate * (1.2 - ob.visual_symmetry) + ob.contact_area_estimate * 0.1;
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
  actionVerb: WorldModelInput['action_verb'],
  a: WorldObject,
  b: WorldObject,
): WorldModelInput {
  const oa = perception.observe(a, world.rng);
  const ob = perception.observe(b, world.rng);
  const dx = b.pos.x - a.pos.x;
  const dy = b.pos.y - a.pos.y;
  return {
    action_verb: actionVerb,
    objectA: toModelFeatures(oa),
    objectB: toModelFeatures(ob),
    geometry_features: Math.max(0, Math.min(1, (a.length / Math.max(0.1, a.thickness) + b.length / Math.max(0.1, b.thickness)) / 12)),
    relative_position: Math.max(0, Math.min(1, Math.hypot(dx, dy) / 3)),
  };
}

function chooseCandidate(candidates: CandidateAction[], worldModel: WorldModel): CandidateAction | undefined {
  if (!candidates.length) return undefined;
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

export class LiveModeEngine {
  readonly seed: number;
  readonly world: World;
  readonly memories: LiveAgentMemory[];
  readonly milestones = new MilestoneTracker();
  readonly replayBuffer = new ReplayBuffer<ReplayTransition, LiveVerb>(20_000);
  readonly bookmarks: LiveBookmark[] = [];
  readonly rollingFrames: SegmentFrame[] = [];
  readonly worldModel: WorldModel;
  readonly embedding: ToolEmbedding;
  readonly perception: PerceptionHead;
  tick = 0;
  simTimeSeconds = 0;
  compositeCount = 0;
  trainMsThisSecond = 0;
  private readonly config: LiveModeConfig;

  constructor(config: LiveModeConfig) {
    this.config = config;
    this.seed = config.seed;
    this.world = new World(config.seed);
    this.worldModel = new WorldModel();
    this.embedding = new ToolEmbedding();
    this.perception = new PerceptionHead(config.seed + 97);
    this.memories = Array.from({ length: Math.max(1, config.populationSize) }, (_, idx) => ({
      id: idx + 1,
      energy: 1,
      observations: [],
      actions: [],
      outcomes: [],
      goodTools: [],
      goodLocations: [],
    }));
  }

  private pushFrame(frame: SegmentFrame): void {
    this.rollingFrames.push(frame);
    const maxFrames = Math.max(1, Math.round(this.config.rollingSeconds * Math.max(1, this.config.ticksPerSecond)));
    while (this.rollingFrames.length > maxFrames) this.rollingFrames.shift();
  }

  private ingestMemory(memory: LiveAgentMemory, action: LiveVerb, score: number, held?: WorldObject): void {
    memory.actions.push(action);
    memory.outcomes.push(score);
    while (memory.actions.length > 32) memory.actions.shift();
    while (memory.outcomes.length > 32) memory.outcomes.shift();
    if (held) {
      const observed = this.perception.observe(held, this.world.rng);
      memory.observations.push(observed);
      while (memory.observations.length > 32) memory.observations.shift();
      if (score > 0.25) {
        memory.goodTools.unshift(held.id);
        memory.goodTools = [...new Set(memory.goodTools)].slice(0, 8);
        memory.goodLocations.unshift({ x: held.pos.x, y: held.pos.y, score });
        memory.goodLocations = memory.goodLocations.slice(0, 8);
      }
    }
  }

  private createCandidates(held: WorldObject | undefined): CandidateAction[] {
    const targetId = this.world.getTargetId();
    const target = targetId ? this.world.objects.get(targetId) : undefined;
    const candidates: CandidateAction[] = [];
    if (!held) {
      for (const candidate of chooseByPerception(this.world, this.perception).slice(0, 3)) {
        candidates.push({ verb: 'PICK_UP', objId: candidate.id, score: 0.05 });
      }
      return candidates;
    }
    if (target && target.id !== held.id) {
      const strikeInput = buildModelInput(this.world, this.perception, 'STRIKE_WITH', held, target);
      candidates.push({ verb: 'STRIKE_WITH', targetId: target.id, modelInput: strikeInput, predicted: this.worldModel.predict(strikeInput) });
    }
    for (const nearby of chooseByPerception(this.world, this.perception, [held.id]).slice(0, 2)) {
      const bindInput = buildModelInput(this.world, this.perception, 'BIND_TO', held, nearby);
      candidates.push({ verb: 'BIND_TO', objId: nearby.id, modelInput: bindInput, predicted: this.worldModel.predict(bindInput) });
      const grindInput = buildModelInput(this.world, this.perception, 'GRIND', held, nearby);
      candidates.push({ verb: 'GRIND', objId: nearby.id, modelInput: grindInput, predicted: this.worldModel.predict(grindInput) });
    }
    return candidates;
  }

  private applyCandidate(chosen: CandidateAction): InteractionOutcome | undefined {
    if (chosen.verb === 'PICK_UP' && chosen.objId) this.world.apply({ type: 'PICK_UP', objId: chosen.objId });
    if (chosen.verb === 'BIND_TO' && chosen.objId) this.world.apply({ type: 'BIND_TO', objId: chosen.objId });
    if (chosen.verb === 'GRIND' && chosen.objId) this.world.apply({ type: 'GRIND', abrasiveId: chosen.objId });
    if (chosen.verb === 'STRIKE_WITH' && chosen.targetId) this.world.apply({ type: 'STRIKE_WITH', targetId: chosen.targetId });
    return this.world.lastInteractionOutcome;
  }

  private ensureEcologyPressure(): void {
    if (this.world.rng.float() < 0.01) this.world.spawnTarget();
  }

  tickOnce(bookmarkId?: string): LiveTickResult {
    this.tick += 1;
    this.simTimeSeconds = this.tick / Math.max(1, this.config.ticksPerSecond);
    this.ensureEcologyPressure();
    const memoryIndex = this.config.deterministic
      ? this.tick % this.memories.length
      : Math.floor(Math.random() * this.memories.length);
    const memory = this.memories[memoryIndex];
    memory.energy = Math.min(1, memory.energy + 0.015);
    const held = this.world.agent.heldObjectId ? this.world.objects.get(this.world.agent.heldObjectId) : undefined;
    let chosen: CandidateAction = { verb: 'REST', score: 0 };
    if (memory.energy >= 0.08) {
      const candidates = this.createCandidates(held);
      const selected = chooseCandidate(candidates, this.worldModel);
      if (selected) chosen = selected;
    }
    const outcome = chosen.verb === 'REST' ? undefined : this.applyCandidate(chosen);
    if (chosen.verb !== 'REST') memory.energy = Math.max(0, memory.energy - 0.07);
    if (chosen.verb === 'BIND_TO') this.compositeCount += 1;
    const effectiveness = outcome ? outcome.damage + outcome.fragments * 0.4 - outcome.toolWear * 0.3 : 0;
    let predictionError = 0;
    if (chosen.modelInput && outcome) {
      const predicted = chosen.predicted ?? this.worldModel.predict(chosen.modelInput);
      predictionError =
        (Math.abs(predicted.expected_damage - outcome.damage) +
          Math.abs(predicted.expected_tool_wear - outcome.toolWear) +
          Math.abs(predicted.expected_fragments - outcome.fragments)) /
        3;
      this.replayBuffer.push({
        state: {
          input: chosen.modelInput,
          outcome: {
            expected_damage: outcome.damage,
            expected_tool_wear: outcome.toolWear,
            expected_fragments: outcome.fragments,
            expected_property_changes: outcome.propertyChanges,
          },
          toolId: outcome.toolId,
        },
        action: chosen.verb,
        reward: effectiveness,
      });
    }
    const heldAfter = this.world.agent.heldObjectId ? this.world.objects.get(this.world.agent.heldObjectId) : undefined;
    this.ingestMemory(memory, chosen.verb, effectiveness, heldAfter);
    const replaySignature = `${this.tick}:${chosen.verb}:${outcome?.action ?? 'none'}:${(outcome?.damage ?? 0).toFixed(4)}`;
    const objects = [outcome?.toolId ?? 0, outcome?.targetId ?? 0].filter((id) => id > 0);
    this.pushFrame({
      tick: this.tick,
      simTimeSeconds: this.simTimeSeconds,
      action: chosen.verb,
      replaySignature,
      objectIds: objects,
    });
    const milestoneEvents = this.milestones.ingest(
      {
        timestamp: this.simTimeSeconds,
        agentId: memory.id,
        action: chosen.verb,
        objectIds: objects,
        compositeKey: chosen.verb === 'BIND_TO' ? `${objects.join(':')}` : undefined,
        predictionError,
        effectiveness,
      },
      bookmarkId,
    );
    const elapsedMinutes = Math.max(1 / 60, this.simTimeSeconds / 60);
    return {
      tick: this.tick,
      simTimeSeconds: this.simTimeSeconds,
      activeAgentId: memory.id,
      action: chosen.verb,
      predictionErrorMean: this.worldModel.meanPredictionError(),
      woodPerMinute: this.world.woodGained / elapsedMinutes,
      novelInteractionsPerMinute: this.embedding.novelInteractionCount() / elapsedMinutes,
      compositeDiscoveryRate: this.compositeCount / Math.max(1, this.tick),
      embeddingClusters: this.embedding.clusterCount(),
      milestones: milestoneEvents,
    };
  }

  trainChunk(config: LiveTrainingConfig, nowMs = performance.now()): number {
    const secondMark = Math.floor(this.simTimeSeconds);
    if (secondMark !== Math.floor((this.tick - 1) / Math.max(1, this.config.ticksPerSecond))) this.trainMsThisSecond = 0;
    let steps = 0;
    const maxSteps = Math.max(1, config.stepsPerTick);
    const budget = Math.max(1, config.maxTrainMsPerSecond);
    while (steps < maxSteps && this.trainMsThisSecond < budget) {
      const started = performance.now();
      const samples = this.replayBuffer.sampleLast(Math.max(1, config.batchSize));
      if (!samples.length) break;
      for (const sample of samples) {
        this.worldModel.update(sample.state.input, sample.state.outcome);
        if (sample.state.toolId) {
          this.embedding.update(sample.state.toolId, {
            damage: sample.state.outcome.expected_damage,
            toolWear: sample.state.outcome.expected_tool_wear,
            fragments: sample.state.outcome.expected_fragments,
            propertyChanges: sample.state.outcome.expected_property_changes,
          });
        }
      }
      steps += 1;
      this.trainMsThisSecond += performance.now() - started;
      if (performance.now() - nowMs > budget) break;
    }
    return steps;
  }

  bookmark(idPrefix = 'bookmark'): LiveBookmark {
    const bookmark: LiveBookmark = {
      id: `${idPrefix}-${this.tick}`,
      createdAtTick: this.tick,
      createdAtSeconds: this.simTimeSeconds,
      frames: this.rollingFrames.map((frame) => ({ ...frame, objectIds: [...frame.objectIds] })),
    };
    this.bookmarks.unshift(bookmark);
    this.bookmarks.splice(16);
    return bookmark;
  }

  replayBookmark(id: string): SegmentFrame[] {
    const bookmark = this.bookmarks.find((entry) => entry.id === id);
    return bookmark ? bookmark.frames.map((frame) => ({ ...frame, objectIds: [...frame.objectIds] })) : [];
  }

  createSnapshot(): LiveSnapshot {
    const elapsedMinutes = Math.max(1 / 60, this.simTimeSeconds / 60);
    return {
      seed: this.seed,
      tick: this.tick,
      simTimeSeconds: this.simTimeSeconds,
      metrics: {
        woodPerMinute: this.world.woodGained / elapsedMinutes,
        predictionErrorMean: this.worldModel.meanPredictionError(),
        novelInteractionsPerMinute: this.embedding.novelInteractionCount() / elapsedMinutes,
        compositeDiscoveryRate: this.compositeCount / Math.max(1, this.tick),
        embeddingClusters: this.embedding.clusterCount(),
      },
      milestones: this.milestones.all(),
      agents: this.memories.map((memory) => ({
        id: memory.id,
        energy: memory.energy,
        observations: [...memory.observations],
        actions: [...memory.actions],
        outcomes: [...memory.outcomes],
        goodTools: [...memory.goodTools],
        goodLocations: memory.goodLocations.map((entry) => ({ ...entry })),
      })),
      world: {
        woodGained: this.world.woodGained,
        objects: [...this.world.objects.values()].map((object) => ({ ...object, pos: { ...object.pos }, vel: { ...object.vel } })),
      },
      modelState: {
        worldModel: this.worldModel.snapshot(),
        perception: this.perception.snapshot(),
        embedding: this.embedding.snapshot(),
      },
    };
  }
}
