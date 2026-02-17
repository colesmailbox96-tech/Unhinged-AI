import { PerceptionHead, type Observation } from '../ai/perception';
import { ReplayBuffer } from '../ai/replay_buffer';
import { ToolEmbedding } from '../ai/tool_embedding';
import { WorldModel, type WorldModelInput, type WorldModelOutcome, type WorldModelPrediction } from '../ai/world_model';
import { ClosedLoopController, type ControllerRuntimeState } from '../ai/controller';
import type { InteractionOutcome } from '../sim/world';
import { World } from '../sim/world';
import type { ObjID, WorldObject } from '../sim/object_model';
import { MilestoneTracker, type MilestoneEvent } from '../sim/milestones';
import type { MeasurementResult } from '../sim/metrology';
import {
  avgDistanceToStation,
  createEmptyWorkset,
  DEFAULT_WORKSET_CONFIG,
  refreshWorkset,
  worksetAtStationFraction,
  type WorksetState,
} from '../sim/workset';
import { TrainingScheduler, type TrainingMetrics } from '../sim/trainingScheduler';
import { StallDetector, type StallMetrics } from '../sim/stallDetector';
import { PopulationController, type PopulationMetrics } from '../sim/populationController';
import { type AgentNeeds, createDefaultNeeds, tickNeeds, homeostasisReward, mostUrgentNeed } from '../sim/needs';
import { SkillTracker, type SkillDiscoveryMetrics } from '../sim/skills';
import { RewardBreakdown, RepeatTracker, diminishingReturnMultiplier, type RewardBreakdownSnapshot } from '../sim/reward_breakdown';
import { PROPERTY_KEYS } from '../sim/properties';

type LiveVerb = 'MOVE_TO' | 'PICK_UP' | 'DROP' | 'BIND_TO' | 'STRIKE_WITH' | 'GRIND' | 'HEAT' | 'SOAK' | 'COOL' | 'ANCHOR' | 'CONTROL' | 'REST';
export type LiveRegime = 'explore' | 'exploit' | 'manufacture';
type LiveIntent = 'SEEK_WATER' | 'SEEK_FOOD' | 'HARVEST' | 'CRAFT' | 'BUILD' | 'MAINTAIN' | 'EXPLORE' | 'REST';
type LiveRole = 'forager' | 'builder' | 'maintainer';

interface CandidateAction {
  verb: LiveVerb;
  objId?: ObjID;
  targetId?: ObjID;
  moveTarget?: { x: number; y: number };
  modelInput?: WorldModelInput;
  predicted?: WorldModelPrediction;
  score?: number;
  intensity?: number;
}

interface AgentTraits {
  strength: number;
  precision: number;
  endurance: number;
  curiosity: number;
  builderBias: number;
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
  livingMode?: boolean;
  livingPreset?: 'default' | 'living-v1-ecology';
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
  needs: AgentNeeds;
  traits: AgentTraits;
  roleBias: LiveRole;
  roleScores: Record<LiveRole, number>;
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
  repeatabilityScore: number;
  precisionScore: number;
  controllerTarget?: { metric: string; target: number; achieved: number };
  measurements: MeasurementResult[];
  stationQuality: number;
  regime: LiveRegime;
  timeInRegime: number;
  regimeChangeReason?: string;
  biomassAvg: number;
  biomassMin: number;
  avgTargetYield: number;
  targetsAlive: number;
  spawnedPerMin: number;
  destroyedPerMin: number;
  objectsTotal: number;
  fragmentsTotal: number;
  despawnedPerMin: number;
  measurementUsefulRate: number;
  measurementUseful: boolean;
  measurementTotal: number;
  measurementSpamPenalty: number;
  avgEnergy: number;
  idleFraction: number;
  actionsPerMin: number;
  controllerStepsPerMin: number;
  controllerState: ControllerRuntimeState;
  controllerStepsLast60s: number;
  controllerEvaluationsLast60s: number;
  lastControllerTarget?: string;
  lastControllerOutcomeDelta?: number;
  embeddingsInWindow: number;
  dutyCycleLab: number;
  dutyCycleWorld: number;
  worksetSize: number;
  worksetIds: number[];
  worksetHomeStationId?: number;
  worksetAgeSec: number;
  worksetAtStationFraction: number;
  haulTripsPerMin: number;
  avgDistanceToStation: number;
  purgedNonDebrisPerMin: number;
  despawnedTargetsPerMin: number;
  spawnSuccessPerMin: number;
  despawnByReason: Record<string, number>;
  activeStationId?: number;
  activeStationQuality: number;
  distanceToActiveStation?: number;
  stationQualities: Array<{ id: number; quality: number }>;
  dutyMode: 'lab' | 'world';
  manufacturingImprovements: number;
  milestones: MilestoneEvent[];
  trainingMetrics: TrainingMetrics;
  stallMetrics: StallMetrics;
  populationMetrics: PopulationMetrics;
  agentIntent: string;
  agentIntentDrivers?: string[];
  agentIntentScores?: Array<{ intent: string; score: number; breakdown: string[] }>;
  agentRole?: LiveRole;
  roleDistribution?: Record<LiveRole, number>;
  // Living Mode fields
  livingMode: boolean;
  agentNeeds?: AgentNeeds;
  rewardBreakdown?: RewardBreakdownSnapshot;
  skillMetrics?: SkillDiscoveryMetrics;
  waterSources: number;
  repeatPenalty: number;
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
  manufacturing: {
    repeatabilityScore: number;
    precisionScore: number;
    stationCount: number;
    regime: LiveRegime;
    timeInRegime: number;
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

const REGIME_THRESHOLDS = {
  EXPLORE_TO_EXPLOIT_MIN_SECONDS: 25,
  EXPLORE_TO_EXPLOIT_MAX_ERROR: 0.22,
  MANUFACTURE_WINDOW_SECONDS: 30,
  MANUFACTURE_MIN_REGIME_SECONDS: 20,
  MANUFACTURE_MAX_NOVELTY_PER_MIN: 8,
  MANUFACTURE_MAX_PREDICTION_ERROR: 0.2,
  MANUFACTURE_MAX_WOOD_VARIANCE: 120,
} as const;

const CONTROL_THRESHOLDS = {
  MIN_PLANARITY_FOR_CONTROL: 0.65,
  MIN_MICROSTRUCTURE_FOR_CONTROL: 0.65,
} as const;

const MANUFACTURE_FORCE_INTERVALS = {
  ANCHOR_TICKS: 1,
  CONTROL_TICKS: 3,
} as const;

const MANUFACTURE_DUTY = {
  LAB: 0.7,
  WORLD: 0.3,
} as const;

const ECOLOGY_LIMITS = {
  MAX_OBJECTS: 120,
} as const;

const MEASUREMENT_SPAM_POLICY = {
  FREE_REPEATS: 3,
  EARLY_REWARD: 0.02,
  PENALTY_STEP: 0.01,
  MAX_PENALTY: 0.08,
} as const;

const ACTION_ENERGY_COST: Partial<Record<LiveVerb, number>> = {
  MOVE_TO: 0.02,
  STRIKE_WITH: 0.16,
  GRIND: 0.1,
  BIND_TO: 0.09,
  HEAT: 0.08,
  SOAK: 0.08,
  COOL: 0.07,
  ANCHOR: 0.09,
  CONTROL: 0.11,
  PICK_UP: 0.04,
  DROP: 0.02,
};

function actionCost(action: LiveVerb): number {
  return ACTION_ENERGY_COST[action] ?? 0.03;
}

function variance(values: number[]): number {
  if (values.length < 2) return 0;
  const mean = values.reduce((sum, value) => sum + value, 0) / values.length;
  return values.reduce((sum, value) => sum + (value - mean) ** 2, 0) / values.length;
}

function clamp01(v: number): number {
  return Math.max(0, Math.min(1, v));
}

function roleForIndex(index: number): LiveRole {
  if (index % 3 === 0) return 'forager';
  if (index % 3 === 1) return 'builder';
  return 'maintainer';
}

function traitsForRole(role: LiveRole): AgentTraits {
  if (role === 'forager') return { strength: 0.74, precision: 0.42, endurance: 0.69, curiosity: 0.58, builderBias: 0.24 };
  if (role === 'builder') return { strength: 0.45, precision: 0.76, endurance: 0.48, curiosity: 0.44, builderBias: 0.82 };
  return { strength: 0.4, precision: 0.63, endurance: 0.73, curiosity: 0.38, builderBias: 0.52 };
}

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

function chooseCandidate(
  candidates: CandidateAction[],
  worldModel: WorldModel,
  noveltyWindow: Map<string, number>,
  regime: LiveRegime,
): CandidateAction | undefined {
  if (!candidates.length) return undefined;
  let best = candidates[0];
  for (const candidate of candidates) {
    if (!candidate.modelInput || !candidate.predicted) continue;
    const noveltyKey = `${candidate.modelInput.action_verb}:${candidate.objId ?? candidate.targetId ?? 0}`;
    const seen = noveltyWindow.get(noveltyKey) ?? 0;
    const curiosity = worldModel.novelty(candidate.modelInput) * Math.max(0.2, 1 - seen * 0.15);
    const utility =
      candidate.predicted.expected_damage * UTILITY_WEIGHTS.damage +
      candidate.predicted.expected_fragments * UTILITY_WEIGHTS.fragments +
      candidate.predicted.expected_property_changes * UTILITY_WEIGHTS.propertyChanges -
      candidate.predicted.expected_tool_wear * UTILITY_WEIGHTS.toolWear;
    const regimeUtilityScale = regime === 'manufacture' && candidate.verb === 'STRIKE_WITH' ? 0.45 : 1;
    candidate.score = curiosity + utility * regimeUtilityScale;
    if ((candidate.score ?? -Infinity) > (best.score ?? -Infinity)) best = candidate;
  }
  if (best.modelInput) {
    const noveltyKey = `${best.modelInput.action_verb}:${best.objId ?? best.targetId ?? 0}`;
    noveltyWindow.set(noveltyKey, (noveltyWindow.get(noveltyKey) ?? 0) + 1);
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
  readonly controller = new ClosedLoopController();
  readonly trainingScheduler = new TrainingScheduler();
  readonly stallDetector = new StallDetector();
  readonly populationCtrl = new PopulationController({ maxObjects: ECOLOGY_LIMITS.MAX_OBJECTS });
  tick = 0;
  simTimeSeconds = 0;
  compositeCount = 0;
  trainMsThisSecond = 0;
  repeatabilityScore = 0;
  precisionScore = 0;
  regime: LiveRegime = 'explore';
  regimeSinceSeconds = 0;
  lastRegimeChangeReason = '';
  spawnedTargets = 0;
  destroyedTargets = 0;
  despawnedObjects = 0;
  purgedNonDebris = 0;
  despawnedTargets = 0;
  spawnSuccess = 0;
  totalActions = 0;
  idleTicks = 0;
  controllerSteps = 0;
  controllerState: ControllerRuntimeState = 'idle';
  manufacturingImprovements = 0;
  haulTrips = 0;
  measurementTotal = 0;
  measurementUsefulCount = 0;
  measurementSpamPenalty = 0;
  lastControllerTarget?: string;
  lastControllerOutcomeDelta?: number;
  private _agentIntent = 'idle';
  private _agentIntentDrivers: string[] = [];
  private _agentIntentScores: Array<{ intent: LiveIntent; score: number; breakdown: string[] }> = [];
  private readonly recentStrikeDamage: number[] = [];
  private readonly recentWoodRates: number[] = [];
  private readonly recentPredictionErrors: number[] = [];
  private readonly recentNoveltyRates: number[] = [];
  private readonly processChain: LiveVerb[] = [];
  private readonly objectLastTransformedTick = new Map<number, number>();
  private readonly objectLastMeasuredTick = new Map<number, number>();
  private readonly measurementRepeatCount = new Map<number, number>();
  private readonly noveltyWindow = new Map<string, number>();
  private readonly controllerStepTimestamps: number[] = [];
  private readonly controllerEvaluationTimestamps: number[] = [];
  private readonly despawnByReasonCounts: Record<string, number> = {};
  private workset: WorksetState = createEmptyWorkset();
  private readonly config: LiveModeConfig;
  private readonly ecologyEnabled: boolean;
  // Living Mode subsystems
  readonly skillTracker = new SkillTracker();
  readonly rewardBreakdown = new RewardBreakdown();
  readonly repeatTracker = new RepeatTracker();
  private _lastRepeatPenalty = 0;

  constructor(config: LiveModeConfig) {
    this.config = config;
    this.ecologyEnabled = config.livingPreset === 'living-v1-ecology';
    this.seed = config.seed;
    this.world = new World(config.seed);
    this.worldModel = new WorldModel();
    this.embedding = new ToolEmbedding();
    this.perception = new PerceptionHead(config.seed + 97);
    this.memories = Array.from({ length: Math.max(1, config.populationSize) }, (_, idx) => {
      const roleBias = roleForIndex(idx);
      return {
        id: idx + 1,
        energy: 1,
        observations: [],
        actions: [],
        outcomes: [],
        goodTools: [],
        goodLocations: [],
        needs: createDefaultNeeds(),
        traits: traitsForRole(roleBias),
        roleBias,
        roleScores: { forager: 0, builder: 0, maintainer: 0 },
      };
    });
    this.spawnedTargets = 1;
    this.spawnSuccess = 1;
    this.trainingScheduler.start();
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

  private shiftRegime(next: LiveRegime, reason: string): void {
    if (this.regime === next) return;
    this.regime = next;
    this.regimeSinceSeconds = this.simTimeSeconds;
    this.lastRegimeChangeReason = `t=${this.simTimeSeconds.toFixed(1)} ${reason}`;
    this.world.logs.unshift(`REGIME ${this.regime.toUpperCase()} ${this.lastRegimeChangeReason}`);
  }

  private maybeUpdateRegime(): void {
    if (
      this.regime === 'explore' &&
      this.simTimeSeconds >= REGIME_THRESHOLDS.EXPLORE_TO_EXPLOIT_MIN_SECONDS &&
      this.worldModel.meanPredictionError() <= REGIME_THRESHOLDS.EXPLORE_TO_EXPLOIT_MAX_ERROR
    ) {
      this.shiftRegime('exploit', 'prediction error stabilized');
    }
    const windowSeconds = REGIME_THRESHOLDS.MANUFACTURE_WINDOW_SECONDS;
    const windowSize = Math.max(10, Math.round(windowSeconds * Math.max(1, this.config.ticksPerSecond)));
    const noveltyWindow = this.recentNoveltyRates.slice(-windowSize);
    const predictionWindow = this.recentPredictionErrors.slice(-windowSize);
    const woodWindow = this.recentWoodRates.slice(-windowSize);
    const noveltyAvg = noveltyWindow.reduce((sum, value) => sum + value, 0) / Math.max(1, noveltyWindow.length);
    const predictionAvg = predictionWindow.reduce((sum, value) => sum + value, 0) / Math.max(1, predictionWindow.length);
    const woodVariance = variance(woodWindow);
    if (
      this.regime !== 'manufacture' &&
      this.simTimeSeconds - this.regimeSinceSeconds > REGIME_THRESHOLDS.MANUFACTURE_MIN_REGIME_SECONDS &&
      noveltyWindow.length >= windowSize &&
      noveltyAvg < REGIME_THRESHOLDS.MANUFACTURE_MAX_NOVELTY_PER_MIN &&
      predictionAvg < REGIME_THRESHOLDS.MANUFACTURE_MAX_PREDICTION_ERROR &&
      woodVariance < REGIME_THRESHOLDS.MANUFACTURE_MAX_WOOD_VARIANCE
    ) {
      this.shiftRegime('manufacture', `novel/min=${noveltyAvg.toFixed(2)} pred=${predictionAvg.toFixed(3)} woodVar=${woodVariance.toFixed(2)}`);
    }
  }

  private processChainLength(): number {
    return new Set(this.processChain.slice(-6)).size;
  }

  private measurementRewardFor(objId: number, useful: boolean): { reward: number; penalty: number; repeats: number } {
    const repeats = (this.measurementRepeatCount.get(objId) ?? 0) + 1;
    this.measurementRepeatCount.set(objId, repeats);
    if (useful) return { reward: 0.12, penalty: 0, repeats };
    if (repeats <= MEASUREMENT_SPAM_POLICY.FREE_REPEATS) return { reward: MEASUREMENT_SPAM_POLICY.EARLY_REWARD, penalty: 0, repeats };
    const penalty = Math.min(MEASUREMENT_SPAM_POLICY.MAX_PENALTY, (repeats - MEASUREMENT_SPAM_POLICY.FREE_REPEATS) * MEASUREMENT_SPAM_POLICY.PENALTY_STEP);
    return { reward: 0, penalty, repeats };
  }

  private controllerTargetFor(held: WorldObject): { metric: 'surface_planarity' | 'microstructure_order' | 'impurity_level'; target: number } {
    if (held.latentPrecision.surface_planarity < 0.78) return { metric: 'surface_planarity', target: 0.82 };
    if (held.latentPrecision.microstructure_order < 0.75) return { metric: 'microstructure_order', target: 0.8 };
    return { metric: 'impurity_level', target: 0.25 };
  }

  setPinWorkset(pinned: boolean): void {
    this.workset = { ...this.workset, pinned };
  }

  private isLabDutyTick(): boolean {
    const cycle = 10;
    const labTicks = Math.round(MANUFACTURE_DUTY.LAB * cycle);
    return this.tick % cycle < labTicks;
  }

  private dominantRole(memory: LiveAgentMemory): LiveRole {
    const sorted = (Object.keys(memory.roleScores) as LiveRole[])
      .map((role) => ({ role, score: memory.roleScores[role] }))
      .sort((a, b) => b.score - a.score);
    return sorted[0]?.role ?? memory.roleBias;
  }

  private deriveIntent(memory: LiveAgentMemory): LiveIntent {
    if (!this.ecologyEnabled) {
      const urgent = mostUrgentNeed(memory.needs);
      if (urgent.need === 'hydration' && urgent.urgency > 0.1) {
        this._agentIntentDrivers = [`hydration=${memory.needs.hydration.toFixed(2)}`];
        return 'SEEK_WATER';
      }
      if (urgent.need === 'energy' && urgent.urgency > 0.1) {
        this._agentIntentDrivers = [`energy=${memory.needs.energy.toFixed(2)}`];
        return 'SEEK_FOOD';
      }
      if (urgent.need === 'fatigue' && urgent.urgency > 0.1) {
        this._agentIntentDrivers = [`fatigue=${memory.needs.fatigue.toFixed(2)}`];
        return 'REST';
      }
      this._agentIntentDrivers = [this.regime === 'manufacture' ? 'manufacture' : 'explore'];
      return this.regime === 'manufacture' ? 'CRAFT' : 'EXPLORE';
    }
    if (this.stallDetector.isInForcedExplore(memory.id)) {
      this._agentIntentScores = [{ intent: 'EXPLORE', score: 1, breakdown: ['stall=1.00', 'forced explore'] }];
      this._agentIntentDrivers = ['stall detected', 'forced explore'];
      return 'EXPLORE';
    }
    const pos = this.world.agent.pos;
    const hydrationNeed = Math.max(0, 0.62 - memory.needs.hydration) * 1.9;
    const energyNeed = Math.max(0, 0.58 - memory.needs.energy) * 1.45;
    const fatigueNeed = Math.max(0, memory.needs.fatigue - 0.64) * 1.5;
    const waterTravel = this.world.nearestWaterDistance(pos.x, pos.y) / Math.max(1, this.world.width);
    const bestBiomass = this.world.richestBiomassCellCenter();
    const harvestTravel = Math.hypot(bestBiomass.x - pos.x, bestBiomass.y - pos.y) / Math.max(1, this.world.width);
    const localBiomass = this.world.biomassAt(pos.x, pos.y);
    const localMoisture = this.world.moistureAt(pos.x, pos.y);
    const maintenancePressure = this.world.stationMaintenancePressure();
    const lowStationCount = this.world.stations.size < 3 ? 0.16 : 0;
    const buildCost = this.world.stations.size * 0.05 - this.world.structureSupportLogistics();
    const holdBonus = this.world.agent.heldObjectId ? 0.09 : 0;
    const scores: Array<{ intent: LiveIntent; score: number; breakdown: string[] }> = [
      {
        intent: 'SEEK_WATER',
        score: hydrationNeed + localMoisture * 0.55 + memory.traits.endurance * 0.08 - waterTravel * 0.4,
        breakdown: [`hydrationNeed=${hydrationNeed.toFixed(2)}`, `moisture=${localMoisture.toFixed(2)}`, `travel=-${(waterTravel * 0.4).toFixed(2)}`],
      },
      {
        intent: 'SEEK_FOOD',
        score: energyNeed + localBiomass * 0.45 + memory.traits.strength * 0.12 - harvestTravel * 0.35,
        breakdown: [`energyNeed=${energyNeed.toFixed(2)}`, `biomass=${localBiomass.toFixed(2)}`, `travel=-${(harvestTravel * 0.35).toFixed(2)}`],
      },
      {
        intent: 'HARVEST',
        score: energyNeed * 0.8 + localBiomass * 0.6 + memory.traits.strength * 0.14 + holdBonus,
        breakdown: [`energyNeed=${(energyNeed * 0.8).toFixed(2)}`, `yield=${(localBiomass * 0.6).toFixed(2)}`, `hold=${holdBonus.toFixed(2)}`],
      },
      {
        intent: 'CRAFT',
        score: memory.traits.precision * 0.35 + holdBonus + (this.regime === 'manufacture' ? 0.24 : 0.08),
        breakdown: [`precision=${(memory.traits.precision * 0.35).toFixed(2)}`, `hold=${holdBonus.toFixed(2)}`, `regime=${this.regime}`],
      },
      {
        intent: 'BUILD',
        score: memory.traits.builderBias * 0.42 + lowStationCount + (this.regime !== 'explore' ? 0.11 : 0.02) - buildCost,
        breakdown: [`builderBias=${(memory.traits.builderBias * 0.42).toFixed(2)}`, `stationNeed=${lowStationCount.toFixed(2)}`, `footprintCost=-${buildCost.toFixed(2)}`],
      },
      {
        intent: 'MAINTAIN',
        score: maintenancePressure * 0.9 + memory.traits.endurance * 0.12 + (this.world.stations.size > 0 ? 0.08 : 0),
        breakdown: [`maintenance=${(maintenancePressure * 0.9).toFixed(2)}`, `endurance=${(memory.traits.endurance * 0.12).toFixed(2)}`, `stations=${this.world.stations.size}`],
      },
      {
        intent: 'EXPLORE',
        score: memory.traits.curiosity * 0.45 + (1 - localBiomass) * 0.2 + (1 - localMoisture) * 0.2,
        breakdown: [`curiosity=${(memory.traits.curiosity * 0.45).toFixed(2)}`, `scarcity=${(((1 - localBiomass) + (1 - localMoisture)) * 0.2).toFixed(2)}`],
      },
      {
        intent: 'REST',
        score: fatigueNeed + Math.max(0, 0.42 - memory.needs.energy) * 0.8,
        breakdown: [`fatigue=${fatigueNeed.toFixed(2)}`, `energyDeficit=${(Math.max(0, 0.42 - memory.needs.energy) * 0.8).toFixed(2)}`],
      },
    ];
    this._agentIntentScores = scores.sort((a, b) => b.score - a.score);
    this._agentIntentDrivers = this._agentIntentScores[0]?.breakdown ?? [];
    return this._agentIntentScores[0]?.intent ?? 'EXPLORE';
  }

  private trackDespawn(reason: string, obj: WorldObject): void {
    this.despawnByReasonCounts[reason] = (this.despawnByReasonCounts[reason] ?? 0) + 1;
    if (obj.debugFamily === 'target-visual') this.despawnedTargets += 1;
  }

  private cleanupDebrisOnly(): void {
    const fragmentsTotal = [...this.world.objects.values()].filter(
      (entry) => (entry.constituents?.length ?? 1) <= 1 && entry.debugFamily === 'fragment',
    ).length;
    this.populationCtrl.evaluate({
      targetsAlive: this.world.targetYieldStats().targetsAlive,
      objectsTotal: this.world.objects.size,
      fragmentsTotal,
    });
    const over = this.world.objects.size - ECOLOGY_LIMITS.MAX_OBJECTS;
    if (over <= 0) return;
    const removable = [...this.world.objects.values()]
      .filter((entry) => entry.debugFamily === 'fragment' || entry.debugFamily === 'dustCandidate')
      .sort((a, b) => a.integrity - b.integrity);
    for (const entry of removable.slice(0, over)) {
      if (this.world.agent.heldObjectId === entry.id || this.workset.ids.includes(entry.id)) continue;
      this.world.objects.delete(entry.id);
      this.trackDespawn('debris-cap', entry);
    }
  }

  private createCandidates(held: WorldObject | undefined): CandidateAction[] {
    const targetId = this.world.getTargetId();
    const target = targetId ? this.world.objects.get(targetId) : undefined;
    const candidates: CandidateAction[] = [];
    const labDuty = this.regime === 'manufacture' && this.isLabDutyTick();
    const dropZone = this.workset.dropZone;
    const worksetObjects = this.workset.ids.map((id) => this.world.objects.get(id)).filter((obj): obj is WorldObject => Boolean(obj));
    const moveHeldToDropZone =
      held && dropZone && Math.hypot(held.pos.x - dropZone.x, held.pos.y - dropZone.y) > DEFAULT_WORKSET_CONFIG.stationRadius;
    if (!held) {
      if (labDuty && dropZone && worksetObjects.length) {
        const notAtStation = worksetObjects
          .filter((obj) => Math.hypot(obj.pos.x - dropZone.x, obj.pos.y - dropZone.y) > DEFAULT_WORKSET_CONFIG.stationRadius)
          .sort(
            (a, b) =>
              Math.hypot(a.pos.x - this.world.agent.pos.x, a.pos.y - this.world.agent.pos.y) -
              Math.hypot(b.pos.x - this.world.agent.pos.x, b.pos.y - this.world.agent.pos.y),
          );
        const pickup = notAtStation[0] ?? worksetObjects[0];
        if (pickup) {
          const dist = Math.hypot(pickup.pos.x - this.world.agent.pos.x, pickup.pos.y - this.world.agent.pos.y);
          if (dist <= DEFAULT_WORKSET_CONFIG.stationRadius) {
            candidates.push({ verb: 'PICK_UP', objId: pickup.id, score: 0.8 });
            candidates.push({ verb: 'MOVE_TO', score: 0.75, intensity: 1, objId: pickup.id });
          } else {
            candidates.push({ verb: 'MOVE_TO', score: 0.75, intensity: 1, objId: pickup.id });
            candidates.push({ verb: 'PICK_UP', objId: pickup.id, score: 0.7 });
          }
          return candidates;
        }
      }
      for (const candidate of chooseByPerception(this.world, this.perception).slice(0, 3)) {
        candidates.push({ verb: 'PICK_UP', objId: candidate.id, score: 0.05 });
      }
      return candidates;
    }
    if (labDuty && dropZone && moveHeldToDropZone) {
      candidates.push({ verb: 'MOVE_TO', score: 0.9, intensity: 1 });
      return candidates;
    }
    if (labDuty && dropZone && held && Math.hypot(held.pos.x - dropZone.x, held.pos.y - dropZone.y) <= DEFAULT_WORKSET_CONFIG.stationRadius) {
      const stationFraction = worksetAtStationFraction(this.world, this.workset, DEFAULT_WORKSET_CONFIG.stationRadius);
      const shouldControl =
        held.latentPrecision.surface_planarity < CONTROL_THRESHOLDS.MIN_PLANARITY_FOR_CONTROL ||
        held.latentPrecision.microstructure_order < CONTROL_THRESHOLDS.MIN_MICROSTRUCTURE_FOR_CONTROL;
      if (stationFraction < 0.7) {
        candidates.push({ verb: 'DROP', score: 0.84 });
        if (shouldControl) candidates.push({ verb: 'CONTROL', score: 0.6 });
      } else {
        if (shouldControl || this.regime === 'manufacture') candidates.push({ verb: 'CONTROL', score: 0.82 });
        candidates.push({ verb: 'DROP', score: 0.58 });
      }
    }
    if (target && target.id !== held.id) {
      const strikeInput = buildModelInput(this.world, this.perception, 'STRIKE_WITH', held, target);
      if (!labDuty) candidates.push({ verb: 'STRIKE_WITH', targetId: target.id, modelInput: strikeInput, predicted: this.worldModel.predict(strikeInput) });
    }
    const nearbyPool = labDuty && worksetObjects.length ? worksetObjects.filter((entry) => entry.id !== held.id) : chooseByPerception(this.world, this.perception, [held.id]).slice(0, 2);
    for (const nearby of nearbyPool.slice(0, 2)) {
      const bindInput = buildModelInput(this.world, this.perception, 'BIND_TO', held, nearby);
      candidates.push({ verb: 'BIND_TO', objId: nearby.id, modelInput: bindInput, predicted: this.worldModel.predict(bindInput) });
      const grindInput = buildModelInput(this.world, this.perception, 'GRIND', held, nearby);
      candidates.push({ verb: 'GRIND', objId: nearby.id, modelInput: grindInput, predicted: this.worldModel.predict(grindInput) });
    }
    if (
      this.regime === 'manufacture' ||
      held.latentPrecision.surface_planarity < CONTROL_THRESHOLDS.MIN_PLANARITY_FOR_CONTROL ||
      held.latentPrecision.microstructure_order < CONTROL_THRESHOLDS.MIN_MICROSTRUCTURE_FOR_CONTROL
    ) {
      candidates.push({ verb: 'CONTROL', score: 0.55 + (1 - held.latentPrecision.surface_planarity) * 0.2 });
    }
    if (!held.anchored && this.regime !== 'explore') {
      const anchorScore = this.regime === 'manufacture' ? 0.72 : held.constituents && held.constituents.length > 1 ? 0.35 : 0.18;
      candidates.push({ verb: 'ANCHOR', score: anchorScore });
    }
    return candidates;
  }

  private applyCandidate(chosen: CandidateAction): {
    outcome?: InteractionOutcome;
    measurement?: MeasurementResult;
    controllerAchieved?: number;
    controllerTarget?: { metric: string; target: number };
    transformedObjectIds: number[];
    controllerOutcomeDelta?: number;
    controllerApplied?: boolean;
    controllerState?: ControllerRuntimeState;
  } {
    const transformedObjectIds: number[] = [];
    if (chosen.verb === 'MOVE_TO') {
      const destination =
        chosen.moveTarget
          ? chosen.moveTarget
          :
        chosen.objId && this.world.objects.has(chosen.objId)
          ? this.world.objects.get(chosen.objId)?.pos
          : this.workset.dropZone && this.regime === 'manufacture'
            ? this.workset.dropZone
            : undefined;
      if (destination) this.world.apply({ type: 'MOVE_TO', x: destination.x, y: destination.y });
    }
    if (chosen.verb === 'PICK_UP' && chosen.objId) this.world.apply({ type: 'PICK_UP', objId: chosen.objId });
    if (chosen.verb === 'DROP') {
      const before = this.world.agent.heldObjectId;
      this.world.apply({ type: 'DROP' });
      if (before) transformedObjectIds.push(before);
    }
    if (chosen.verb === 'BIND_TO' && chosen.objId) {
      const heldBefore = this.world.agent.heldObjectId;
      this.world.apply({ type: 'BIND_TO', objId: chosen.objId });
      if (heldBefore) transformedObjectIds.push(heldBefore, chosen.objId, this.world.agent.heldObjectId ?? 0);
    }
    if (chosen.verb === 'GRIND' && chosen.objId) {
      this.world.apply({ type: 'GRIND', abrasiveId: chosen.objId, intensity: chosen.intensity });
      if (this.world.agent.heldObjectId) transformedObjectIds.push(this.world.agent.heldObjectId);
    }
    if (chosen.verb === 'STRIKE_WITH' && chosen.targetId) {
      this.world.apply({ type: 'STRIKE_WITH', targetId: chosen.targetId });
      transformedObjectIds.push(chosen.targetId);
    }
    if (chosen.verb === 'HEAT') {
      this.world.apply({ type: 'HEAT', intensity: chosen.intensity ?? 0.5 });
      if (this.world.agent.heldObjectId) transformedObjectIds.push(this.world.agent.heldObjectId);
    }
    if (chosen.verb === 'SOAK') {
      this.world.apply({ type: 'SOAK', intensity: chosen.intensity ?? 0.5 });
      if (this.world.agent.heldObjectId) transformedObjectIds.push(this.world.agent.heldObjectId);
    }
    if (chosen.verb === 'COOL') {
      this.world.apply({ type: 'COOL', intensity: chosen.intensity ?? 0.5 });
      if (this.world.agent.heldObjectId) transformedObjectIds.push(this.world.agent.heldObjectId);
    }
    if (chosen.verb === 'ANCHOR') {
      const heldBefore = this.world.agent.heldObjectId;
      this.world.apply({ type: 'ANCHOR' });
      if (heldBefore) transformedObjectIds.push(heldBefore);
    }
    if (chosen.verb === 'CONTROL' && this.world.agent.heldObjectId) {
      const held = this.world.objects.get(this.world.agent.heldObjectId);
      if (held) {
        const target = this.controllerTargetFor(held);
        const before =
          target.metric === 'surface_planarity'
            ? held.latentPrecision.surface_planarity
            : target.metric === 'microstructure_order'
              ? held.latentPrecision.microstructure_order
              : held.latentPrecision.impurity_level;
        const step = this.controller.step(this.world, held, target);
        const after =
          target.metric === 'surface_planarity'
            ? step.achieved
            : target.metric === 'microstructure_order'
              ? step.achieved
              : 1 - step.achieved;
        return {
          outcome: this.world.lastInteractionOutcome,
          measurement: step.measured,
          controllerAchieved: step.achieved,
          controllerTarget: step.applied ? target : undefined,
          transformedObjectIds: [held.id],
          controllerOutcomeDelta: after - before,
          controllerApplied: step.applied,
          controllerState: step.state,
        };
      }
    }
    if (chosen.verb === 'CONTROL') {
      this.world.maintainNearestStation(this.world.agent.pos.x, this.world.agent.pos.y, 0.04);
    }
    return { outcome: this.world.lastInteractionOutcome, transformedObjectIds };
  }

  private ensureEcologyPressure(): void {
    this.world.regrowBiomass(0.0035);
    if (this.ecologyEnabled) {
      this.world.regrowMoisture(0.004);
      this.world.tickStructureDecay();
    }
    if (this.world.objects.size < 18) this.world.spawnLooseObject();
    if (this.world.targetYieldStats().targetsAlive === 0) {
      this.world.spawnTarget();
      this.spawnedTargets += 1;
      this.spawnSuccess += 1;
    }
    if (this.world.rng.float() < 0.01) {
      this.world.spawnTarget();
      this.spawnedTargets += 1;
      this.spawnSuccess += 1;
    }
    this.cleanupDebrisOnly();
  }

  tickOnce(): LiveTickResult {
    this.tick += 1;
    this.simTimeSeconds = this.tick / Math.max(1, this.config.ticksPerSecond);
    this.ensureEcologyPressure();
    this.maybeUpdateRegime();
    this.world.recomputeStations();
    this.workset = refreshWorkset(this.world, this.workset, DEFAULT_WORKSET_CONFIG, 1 / Math.max(1, this.config.ticksPerSecond));
    this.world.worksetDebugIds = [...this.workset.ids];
    this.world.worksetDropZone = this.workset.dropZone ? { ...this.workset.dropZone } : undefined;
    const memoryIndex = this.config.deterministic ? this.tick % this.memories.length : this.world.rng.int(0, this.memories.length);
    const memory = this.memories[memoryIndex];
    const objectCountBeforeAction = this.world.objects.size;
    memory.energy = Math.min(1, memory.energy + 0.018);
    this.world.agent.energy = memory.energy;
    const held = this.world.agent.heldObjectId ? this.world.objects.get(this.world.agent.heldObjectId) : undefined;
    let chosen: CandidateAction = { verb: 'REST', score: 0 };
    const dutyMode: 'lab' | 'world' = this.regime === 'manufacture' && this.isLabDutyTick() ? 'lab' : 'world';
    if (memory.energy >= 0.05) {
      const candidates = this.createCandidates(held);
      const selected = chooseCandidate(candidates, this.worldModel, this.noveltyWindow, this.regime);
      if (selected) chosen = selected;
      if (this.regime === 'manufacture' && this.world.stations.size === 0) {
        const anchor = candidates.find((entry) => entry.verb === 'ANCHOR');
        if (anchor && this.tick % MANUFACTURE_FORCE_INTERVALS.ANCHOR_TICKS === 0) chosen = anchor;
      }
      if (this.regime === 'manufacture' && held && this.world.stations.size > 0) {
        const control = candidates.find((entry) => entry.verb === 'CONTROL');
        if (control && (this.tick % MANUFACTURE_FORCE_INTERVALS.CONTROL_TICKS === 0 || selected?.verb === 'STRIKE_WITH')) chosen = control;
      }
      if (dutyMode === 'world' && this.regime === 'manufacture' && chosen.verb === 'CONTROL') {
        const worldFallback = candidates.find((entry) => entry.verb === 'STRIKE_WITH' || entry.verb === 'PICK_UP');
        if (worldFallback) chosen = worldFallback;
      }
      // Stall detection: if forced explore, override action (skip during manufacture to avoid disrupting controller)
      if (this.regime !== 'manufacture' && this.stallDetector.isInForcedExplore(memory.id)) {
        // Force exploration: drop current object and pick up something new, or move
        const exploreCandidates = candidates.filter((entry) => entry.verb === 'PICK_UP' || entry.verb === 'MOVE_TO' || entry.verb === 'DROP');
        if (exploreCandidates.length > 0) {
          chosen = exploreCandidates[this.world.rng.int(0, exploreCandidates.length)];
        } else if (candidates.length > 0) {
          chosen = candidates[this.world.rng.int(0, candidates.length)]; // random action
        }
      }
    }
    if (this.config.livingMode && this.ecologyEnabled && memory.energy >= 0.05) {
      const intent = this.deriveIntent(memory);
      this._agentIntent = intent;
      const candidates = this.createCandidates(held);
      const pick = (verbs: LiveVerb[]): CandidateAction | undefined => candidates.find((entry) => verbs.includes(entry.verb));
      if (intent === 'SEEK_WATER') {
        chosen = held ? pick(['SOAK', 'MOVE_TO']) ?? chosen : { verb: 'MOVE_TO', moveTarget: this.world.nearestWaterSource(this.world.agent.pos.x, this.world.agent.pos.y), score: 0.7 };
      } else if (intent === 'SEEK_FOOD') {
        chosen = held ? pick(['STRIKE_WITH', 'GRIND', 'MOVE_TO']) ?? chosen : pick(['PICK_UP', 'MOVE_TO']) ?? { verb: 'MOVE_TO', moveTarget: this.world.richestBiomassCellCenter(), score: 0.68 };
      } else if (intent === 'HARVEST') {
        chosen = pick(['STRIKE_WITH', 'GRIND', 'PICK_UP', 'MOVE_TO']) ?? chosen;
      } else if (intent === 'BUILD') {
        chosen = pick(['ANCHOR', 'BIND_TO', 'MOVE_TO']) ?? chosen;
      } else if (intent === 'MAINTAIN') {
        chosen = pick(['CONTROL', 'MOVE_TO', 'DROP']) ?? chosen;
      } else if (intent === 'CRAFT') {
        chosen = pick(['CONTROL', 'BIND_TO', 'GRIND']) ?? chosen;
      } else if (intent === 'REST') {
        chosen = { verb: 'REST', score: 0 };
      } else {
        chosen = pick(['MOVE_TO', 'PICK_UP', 'DROP']) ?? chosen;
      }
    } else {
      this._agentIntent = this.deriveIntent(memory);
    }
    const baseCost = actionCost(chosen.verb);
    const buildLoadScale = chosen.verb === 'ANCHOR'
      ? Math.max(1, 1 + this.world.stations.size * 0.1 - this.world.structureSupportLogistics())
      : 1;
    const cost = baseCost * buildLoadScale;
    if (chosen.verb !== 'REST' && memory.energy < cost) chosen = { verb: 'REST', score: 0 };
    const applyResult = chosen.verb === 'REST' ? { transformedObjectIds: [] } : this.applyCandidate(chosen);
    const outcome = applyResult.outcome;
    if (chosen.verb !== 'REST') {
      memory.energy = Math.max(0, memory.energy - cost);
      this.totalActions += 1;
    } else {
      this.idleTicks += 1;
    }
    this.world.agent.energy = memory.energy;
    // Living Mode: tick needs + repeat tracking
    let repeatPenalty = 0;
    let skillReward = 0;
    const heldBeforeProps = held ? { ...held.props } : undefined;
    if (this.config.livingMode) {
      const needsResult = tickNeeds(memory.needs, chosen.verb);
      memory.needs = needsResult.needs;
      if (chosen.verb === 'SOAK') {
        const moistureBoost = this.world.moistureAt(this.world.agent.pos.x, this.world.agent.pos.y) * 0.04;
        const purifierBoost = [...this.world.stations.values()].some((station) =>
          station.functionType === 'purifier' && Math.hypot(station.worldPos.x - this.world.agent.pos.x, station.worldPos.y - this.world.agent.pos.y) <= 2.2)
          ? 0.02
          : 0;
        memory.needs.hydration = clamp01(memory.needs.hydration + moistureBoost + purifierBoost);
      }
      memory.energy = memory.needs.energy;
      this.world.agent.energy = memory.energy;
      // Track repeat penalties
      const actionObjId = chosen.objId ?? chosen.targetId ?? 0;
      if (actionObjId > 0) {
        const repeats = this.repeatTracker.record(chosen.verb, actionObjId, this.tick);
        repeatPenalty = repeats > 2 ? (1 - diminishingReturnMultiplier(repeats)) * 0.1 : 0;
        this._lastRepeatPenalty = repeatPenalty;
      }
    }
    const transformedObjectIds = applyResult.transformedObjectIds.filter((id) => id > 0);
    for (const id of transformedObjectIds) {
      this.objectLastTransformedTick.set(id, this.tick);
      this.measurementRepeatCount.set(id, 0);
    }
    if (chosen.verb === 'BIND_TO') this.compositeCount += 1;
    if (chosen.verb === 'CONTROL') {
      this.controllerEvaluationTimestamps.push(this.simTimeSeconds);
      const stepApplied = Boolean(applyResult.controllerApplied);
      if (stepApplied) {
        this.controllerSteps += 1;
        this.controllerStepTimestamps.push(this.simTimeSeconds);
      }
      this.lastControllerTarget = applyResult.controllerTarget?.metric;
      this.lastControllerOutcomeDelta = applyResult.controllerOutcomeDelta;
      if ((applyResult.controllerOutcomeDelta ?? 0) > 0) this.manufacturingImprovements += 1;
    }
    if (chosen.verb === 'DROP' && this.workset.dropZone) this.haulTrips += 1;
    this.controllerState = chosen.verb === 'CONTROL' ? (applyResult.controllerState ?? this.controller.currentState()) : 'idle';
    if (chosen.verb !== 'CONTROL') this.controller.setIdle();
    if (chosen.verb === 'STRIKE_WITH' && outcome) {
      this.recentStrikeDamage.push(outcome.damage);
      while (this.recentStrikeDamage.length > 10) this.recentStrikeDamage.shift();
      if (this.world.lastInteractionOutcome?.targetId && !this.world.objects.has(this.world.lastInteractionOutcome.targetId)) {
        this.destroyedTargets += 1;
        this.spawnedTargets += 1;
        this.spawnSuccess += 1;
      }
    }
    const repeatabilityVariance = (() => {
      if (this.recentStrikeDamage.length < 2) return 0;
      const mean = this.recentStrikeDamage.reduce((sum, value) => sum + value, 0) / this.recentStrikeDamage.length;
      return this.recentStrikeDamage.reduce((sum, value) => sum + (value - mean) ** 2, 0) / this.recentStrikeDamage.length;
    })();
    this.repeatabilityScore = 1 / (1 + repeatabilityVariance);
    const heldAfterProcess = this.world.agent.heldObjectId ? this.world.objects.get(this.world.agent.heldObjectId) : undefined;
    const precisionNow = heldAfterProcess
      ? (heldAfterProcess.latentPrecision.surface_planarity +
          heldAfterProcess.latentPrecision.microstructure_order +
          (1 - heldAfterProcess.latentPrecision.impurity_level)) /
        3
      : 0;
    this.precisionScore = precisionNow;
    this.processChain.push(chosen.verb);
    while (this.processChain.length > 12) this.processChain.shift();
    const effectivenessBase = outcome ? outcome.damage + outcome.fragments * 0.4 - outcome.toolWear * 0.3 : 0;
    const woodWeight = this.regime === 'manufacture' ? 0.15 : this.regime === 'exploit' ? 0.45 : 0.8;
    const qualityWeight = this.regime === 'manufacture' ? 0.65 : 0.35;
    let effectiveness = effectivenessBase * woodWeight + this.repeatabilityScore * qualityWeight * 0.3 + this.precisionScore * qualityWeight * 0.35;
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
    this.recentPredictionErrors.push(this.worldModel.meanPredictionError());
    while (this.recentPredictionErrors.length > 1024) this.recentPredictionErrors.shift();
    const heldAfter = this.world.agent.heldObjectId ? this.world.objects.get(this.world.agent.heldObjectId) : undefined;
    const measurements = heldAfter ? this.world.measureObject(heldAfter.id, heldAfter.id) : [];
    let measurementUseful = false;
    if (heldAfter && measurements.length) {
      this.measurementTotal += 1;
      const lastMeasurementTick = this.objectLastMeasuredTick.get(heldAfter.id) ?? -1;
      const transformedSinceLast = (this.objectLastTransformedTick.get(heldAfter.id) ?? -1) > lastMeasurementTick;
      measurementUseful = Boolean(applyResult.controllerTarget) || transformedSinceLast || this.processChainLength() >= 3;
      const measurementReward = this.measurementRewardFor(heldAfter.id, measurementUseful);
      effectiveness += measurementReward.reward;
      this.measurementSpamPenalty = measurementReward.penalty;
      effectiveness = Math.max(-0.2, effectiveness - measurementReward.penalty);
      if (measurementUseful) this.measurementUsefulCount += 1;
      this.objectLastMeasuredTick.set(heldAfter.id, this.tick);
    } else {
      this.measurementSpamPenalty = 0;
    }
    const stationQuality =
      heldAfter?.id !== undefined
        ? this.world.stations.get(heldAfter.id)?.quality ?? 0
        : [...this.world.stations.values()].reduce((best, station) => Math.max(best, station.quality), 0);
    // Living Mode: skill tracking + reward breakdown
    if (this.config.livingMode) {
      // Track property transformations for skill discovery
      if (heldAfter && heldBeforeProps) {
        for (const key of PROPERTY_KEYS) {
          if (Math.abs(heldAfter.props[key] - heldBeforeProps[key]) > 0.005) {
            skillReward += this.skillTracker.recordTransformation(
              chosen.verb, key, heldBeforeProps[key], heldAfter.props[key], this.tick,
            );
          }
        }
      }
      // Apply living mode reward adjustments
      const homeostasis = homeostasisReward(memory.needs);
      const roleReward = {
        forager: (chosen.verb === 'STRIKE_WITH' || chosen.verb === 'GRIND' ? 0.06 : 0) + Math.max(0, this.world.biomassAt(this.world.agent.pos.x, this.world.agent.pos.y) - 0.3) * 0.03,
        builder: (chosen.verb === 'ANCHOR' || chosen.verb === 'BIND_TO' ? 0.06 : 0) + (chosen.verb === 'CONTROL' ? 0.03 : 0) + this.world.stations.size * 0.005,
        maintainer: (chosen.verb === 'CONTROL' ? 0.07 : 0) + this.world.stationMaintenancePressure() * 0.03 + (this.world.targetYieldStats().targetsAlive > 0 ? 0.005 : 0),
      };
      const roleBiasScale = memory.roleBias === 'forager' ? roleReward.forager : memory.roleBias === 'builder' ? roleReward.builder : roleReward.maintainer;
      memory.roleScores.forager += roleReward.forager;
      memory.roleScores.builder += roleReward.builder;
      memory.roleScores.maintainer += roleReward.maintainer;
      effectiveness += homeostasis + skillReward - repeatPenalty + roleBiasScale;
      // Record reward breakdown
      this.rewardBreakdown.record(this.tick, {
        survival: homeostasis,
        foodIntake: chosen.verb === 'STRIKE_WITH' ? effectivenessBase * 0.3 : 0,
        waterIntake: chosen.verb === 'SOAK' ? (0.04 + this.world.moistureAt(this.world.agent.pos.x, this.world.agent.pos.y) * 0.06) : 0,
        craftingOutcome: chosen.verb === 'BIND_TO' || chosen.verb === 'CONTROL' || chosen.verb === 'ANCHOR' ? this.precisionScore * 0.1 : 0,
        novelty: predictionError > 0.08 ? 0.03 : 0,
        predictionError: predictionError * 0.1,
        empowerment: this.repeatabilityScore * 0.02,
        skillDiscovery: skillReward,
        spamPenalty: -this.measurementSpamPenalty,
        repeatPenalty: -repeatPenalty - this.world.stationMaintenancePressure() * 0.01,
        idlePenalty: chosen.verb === 'REST' ? -0.01 : 0,
      });
    }
    this.ingestMemory(memory, chosen.verb, effectiveness, heldAfter);
    // Record action in stall detector
    this.stallDetector.record(
      memory.id,
      this.tick,
      chosen.verb,
      effectiveness,
      outcome?.targetId,
      outcome?.toolId,
    );
    if (this.world.objects.size < objectCountBeforeAction) this.despawnedObjects += objectCountBeforeAction - this.world.objects.size;
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
          stationQuality,
          measurementSigma: measurements[0]?.sigma,
          measurementSigmaBaseline: measurements[0]?.sampleCount ? measurements[0].sigma * Math.sqrt(measurements[0].sampleCount) : undefined,
          controllerTarget: applyResult.controllerTarget?.target,
          controllerAchieved: applyResult.controllerAchieved,
          processChainAction: chosen.verb,
          processChainLength: this.processChainLength(),
          controllerSteps: this.controllerSteps,
          stations: this.world.stations.size,
          measurementUseful,
        },
        undefined,
      );
    const elapsedMinutes = Math.max(1 / 60, this.simTimeSeconds / 60);
    const woodPerMinute = this.world.woodGained / elapsedMinutes;
    this.recentWoodRates.push(woodPerMinute);
    while (this.recentWoodRates.length > 1024) this.recentWoodRates.shift();
    const novelPerMinute = this.embedding.novelInteractionCount() / elapsedMinutes;
    this.recentNoveltyRates.push(novelPerMinute);
    while (this.recentNoveltyRates.length > 1024) this.recentNoveltyRates.shift();
    const biomassStats = this.world.biomassStats();
    const targetYieldStats = this.world.targetYieldStats();
    const avgEnergy = this.memories.reduce((sum, entry) => sum + entry.energy, 0) / Math.max(1, this.memories.length);
    const measurementUsefulRate = this.measurementUsefulCount / Math.max(1, this.measurementTotal);
    while (this.controllerStepTimestamps.length && this.controllerStepTimestamps[0] < this.simTimeSeconds - 60) this.controllerStepTimestamps.shift();
    while (this.controllerEvaluationTimestamps.length && this.controllerEvaluationTimestamps[0] < this.simTimeSeconds - 60)
      this.controllerEvaluationTimestamps.shift();
    const stationEntries = [...this.world.stations.values()].sort((a, b) => b.quality - a.quality);
    const activeStation = stationEntries[0];
    const worksetFraction = worksetAtStationFraction(this.world, this.workset, DEFAULT_WORKSET_CONFIG.stationRadius);
    const avgDist = avgDistanceToStation(this.world, this.workset);
    const roleDistribution = this.memories.reduce<Record<LiveRole, number>>(
      (acc, entry) => {
        acc[this.dominantRole(entry)] += 1;
        return acc;
      },
      { forager: 0, builder: 0, maintainer: 0 },
    );
    return {
      tick: this.tick,
      simTimeSeconds: this.simTimeSeconds,
      activeAgentId: memory.id,
      action: chosen.verb,
      predictionErrorMean: this.worldModel.meanPredictionError(),
      woodPerMinute,
      novelInteractionsPerMinute: novelPerMinute,
      compositeDiscoveryRate: this.compositeCount / Math.max(1, this.tick),
      embeddingClusters: this.embedding.clusterCount(),
      repeatabilityScore: this.repeatabilityScore,
      precisionScore: this.precisionScore,
      controllerTarget: applyResult.controllerTarget
        ? {
            metric: applyResult.controllerTarget.metric,
            target: applyResult.controllerTarget.target,
            achieved: applyResult.controllerAchieved ?? 0,
          }
        : undefined,
      measurements,
      stationQuality,
      regime: this.regime,
      timeInRegime: Math.max(0, this.simTimeSeconds - this.regimeSinceSeconds),
      regimeChangeReason: this.lastRegimeChangeReason || undefined,
      biomassAvg: biomassStats.avg,
      biomassMin: biomassStats.min,
      avgTargetYield: targetYieldStats.avgTargetYield,
      targetsAlive: targetYieldStats.targetsAlive,
      spawnedPerMin: this.spawnedTargets / elapsedMinutes,
      destroyedPerMin: this.destroyedTargets / elapsedMinutes,
      objectsTotal: this.world.objects.size,
      fragmentsTotal: [...this.world.objects.values()].filter((entry) => (entry.constituents?.length ?? 1) <= 1 && entry.debugFamily === 'fragment').length,
      despawnedPerMin: this.despawnedObjects / elapsedMinutes,
      measurementUsefulRate,
      measurementUseful,
      measurementTotal: this.measurementTotal,
      measurementSpamPenalty: this.measurementSpamPenalty,
      avgEnergy,
      idleFraction: this.idleTicks / Math.max(1, this.tick),
      actionsPerMin: this.totalActions / elapsedMinutes,
      controllerStepsPerMin: this.controllerSteps / elapsedMinutes,
      controllerState: this.controllerState,
      controllerStepsLast60s: this.controllerStepTimestamps.length,
      controllerEvaluationsLast60s: this.controllerEvaluationTimestamps.length,
      lastControllerTarget: this.lastControllerTarget,
      lastControllerOutcomeDelta: this.lastControllerOutcomeDelta,
      embeddingsInWindow: this.embedding.entries().length,
      dutyCycleLab: MANUFACTURE_DUTY.LAB,
      dutyCycleWorld: MANUFACTURE_DUTY.WORLD,
      worksetSize: this.workset.ids.length,
      worksetIds: [...this.workset.ids],
      worksetHomeStationId: this.workset.homeStationId,
      worksetAgeSec: this.workset.ageSec,
      worksetAtStationFraction: worksetFraction,
      haulTripsPerMin: this.haulTrips / elapsedMinutes,
      avgDistanceToStation: avgDist,
      purgedNonDebrisPerMin: this.purgedNonDebris / elapsedMinutes,
      despawnedTargetsPerMin: this.despawnedTargets / elapsedMinutes,
      spawnSuccessPerMin: this.spawnSuccess / elapsedMinutes,
      despawnByReason: { ...this.despawnByReasonCounts },
      activeStationId: activeStation?.objectId,
      activeStationQuality: activeStation?.quality ?? 0,
      distanceToActiveStation: activeStation
        ? Math.hypot(this.world.agent.pos.x - activeStation.worldPos.x, this.world.agent.pos.y - activeStation.worldPos.y)
        : undefined,
      stationQualities: stationEntries.map((station) => ({ id: station.objectId, quality: station.quality })),
      dutyMode,
      manufacturingImprovements: this.manufacturingImprovements,
      milestones: milestoneEvents,
      trainingMetrics: this.trainingScheduler.metrics(this.simTimeSeconds, this.replayBuffer.size),
      stallMetrics: this.stallDetector.metrics(elapsedMinutes),
      populationMetrics: this.populationCtrl.metrics(),
      agentIntent: this._agentIntent,
      agentIntentDrivers: [...this._agentIntentDrivers],
      agentIntentScores: this._agentIntentScores.slice(0, 3).map((entry) => ({ intent: entry.intent, score: entry.score, breakdown: [...entry.breakdown] })),
      agentRole: this.dominantRole(memory),
      roleDistribution,
      // Living Mode fields
      livingMode: Boolean(this.config.livingMode),
      agentNeeds: this.config.livingMode ? { ...memory.needs } : undefined,
      rewardBreakdown: this.config.livingMode ? this.rewardBreakdown.latest() : undefined,
      skillMetrics: this.config.livingMode ? this.skillTracker.metrics() : undefined,
      waterSources: this.config.livingMode ? this.world.waterSources.length : 0,
      repeatPenalty: this._lastRepeatPenalty,
    };
  }

  trainChunk(config: LiveTrainingConfig, nowMs = performance.now()): number {
    const secondMark = Math.floor(this.simTimeSeconds);
    if (secondMark !== Math.floor((this.tick - 1) / Math.max(1, this.config.ticksPerSecond))) this.trainMsThisSecond = 0;
    let steps = 0;
    const maxSteps = Math.max(1, config.stepsPerTick);
    const budget = Math.max(1, config.maxTrainMsPerSecond);
    if (this.replayBuffer.size === 0) {
      this.trainingScheduler.setCollecting();
      return 0;
    }
    while (steps < maxSteps && this.trainMsThisSecond < budget) {
      const started = performance.now();
      const samples = this.replayBuffer.sampleLast(Math.max(1, config.batchSize));
      if (!samples.length) break;
      let batchLoss = 0;
      for (const sample of samples) {
        const err = this.worldModel.update(sample.state.input, sample.state.outcome);
        batchLoss += err;
        if (sample.state.toolId) {
          this.embedding.update(sample.state.toolId, {
            damage: sample.state.outcome.expected_damage,
            toolWear: sample.state.outcome.expected_tool_wear,
            fragments: sample.state.outcome.expected_fragments,
            propertyChanges: sample.state.outcome.expected_property_changes,
          });
        }
      }
      const stepDuration = performance.now() - started;
      const avgLoss = batchLoss / Math.max(1, samples.length);
      const actionDiversity = new Set(samples.map(s => s.action)).size / Math.max(1, samples.length);
      this.trainingScheduler.recordStep(stepDuration, avgLoss, actionDiversity, this.simTimeSeconds);
      steps += 1;
      this.trainMsThisSecond += stepDuration;
      if (performance.now() - nowMs > budget) {
        this.trainingScheduler.recordRateLimited();
        break;
      }
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

  focusOnObject(objectId: number): boolean {
    const object = this.world.objects.get(objectId);
    if (!object) return false;
    this.world.agent.pos = { ...object.pos };
    return true;
  }

  measurementSpamSeries(objectId: number, repeats: number): number[] {
    const series: number[] = [];
    for (let i = 0; i < repeats; i++) {
      const result = this.measurementRewardFor(objectId, false);
      series.push(result.reward - result.penalty);
    }
    return series;
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
        needs: { ...memory.needs },
        traits: { ...memory.traits },
        roleBias: memory.roleBias,
        roleScores: { ...memory.roleScores },
      })),
      world: {
        woodGained: this.world.woodGained,
        objects: [...this.world.objects.values()].map((object) => ({ ...object, pos: { ...object.pos }, vel: { ...object.vel } })),
      },
      manufacturing: {
        repeatabilityScore: this.repeatabilityScore,
        precisionScore: this.precisionScore,
        stationCount: this.world.stations.size,
        regime: this.regime,
        timeInRegime: Math.max(0, this.simTimeSeconds - this.regimeSinceSeconds),
      },
      modelState: {
        worldModel: this.worldModel.snapshot(),
        perception: this.perception.snapshot(),
        embedding: this.embedding.snapshot(),
      },
    };
  }
}
