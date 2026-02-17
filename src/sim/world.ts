import { deriveGripScore, type ObjID, type ShapeType, type WorldObject } from './object_model';
import { bindObjects, cool, grind, heat, soak, strike } from './interactions';
import { fibrousTargetProps, materialDistributions, sampleProperties } from './material_distributions';
import { fibrousTargetScore } from './properties';
import { RNG } from './rng';
import { type MeasurementResult, measureConductivity, measureGeometry, measureHardness, measureMass, measureOptical } from './metrology';
import { anchorObject, stationFromAnchoredObject, type AnchoredStation, type StationFunction } from './stations';
import { SpatialGrid } from './spatial_grid';

export type PrimitiveVerb =
  | { type: 'MOVE_TO'; x: number; y: number }
  | { type: 'PICK_UP'; objId: ObjID }
  | { type: 'DROP' }
  | { type: 'BIND_TO'; objId: ObjID }
  | { type: 'STRIKE_WITH'; targetId: ObjID }
  | { type: 'GRIND'; abrasiveId: ObjID; intensity?: number }
  | { type: 'HEAT'; intensity: number }
  | { type: 'SOAK'; intensity: number }
  | { type: 'COOL'; intensity: number }
  | { type: 'ANCHOR'; worldPos?: { x: number; y: number } };

export interface AgentState {
  id: number;
  pos: { x: number; y: number };
  heldObjectId?: ObjID;
  energy: number;
}

interface StrikeArc {
  center: { x: number; y: number };
  radius: number;
  start: number;
  end: number;
  alpha: number;
}

export interface InteractionOutcome {
  action: 'BIND_TO' | 'STRIKE_WITH' | 'GRIND';
  toolId?: ObjID;
  targetId?: ObjID;
  damage: number;
  toolWear: number;
  fragments: number;
  propertyChanges: number;
}

export interface PredictionRealityOverlay {
  predicted: { damage: number; toolWear: number; fragments: number };
  actual: { damage: number; toolWear: number; fragments: number };
  error: { damage: number; toolWear: number; fragments: number };
}

export class World {
  private static readonly MAX_AGENT_STEP_DISTANCE = 0.65;
  private static readonly MAX_PICKUP_DISTANCE = 2.5;
  private static readonly MAX_STRUCTURE_TILES = 14;
  private static readonly MAX_LOG_ENTRIES = 200;
  readonly width = 10;
  readonly height = 10;
  readonly biomassResolution = 5;
  readonly rng: RNG;
  readonly objects = new Map<ObjID, WorldObject>();
  readonly logs: string[] = [];
  readonly spatialGrid = new SpatialGrid<WorldObject>(2.5);
  lastStrikeArc?: StrikeArc;
  predictedStrikeArc?: StrikeArc;
  predictedStrikeDamage?: number;
  actualStrikeDamage?: number;
  predictionRealityOverlay?: PredictionRealityOverlay;
  lastInteractionOutcome?: InteractionOutcome;
  readonly agent: AgentState = { id: 1, pos: { x: 5, y: 5 }, energy: 1 };
  nextObjectId = 1;
  woodGained = 0;
  readonly stations = new Map<ObjID, AnchoredStation>();
  readonly stationDurability = new Map<ObjID, number>();
  readonly stationFootprint = new Map<ObjID, number>();
  readonly lastMeasurements = new Map<ObjID, MeasurementResult[]>();
  readonly biomass: number[][];
  readonly biomassCapacity: number[][];
  readonly biomassCooldown: number[][];
  readonly harvestPressure: number[][];
  readonly moisture: number[][];
  readonly moistureCapacity: number[][];
  readonly waterSources: Array<{ x: number; y: number }> = [
    { x: 1.6, y: 1.8 },
    { x: 8.1, y: 2.4 },
    { x: 7.2, y: 8.2 },
  ];
  worksetDebugIds: ObjID[] = [];
  worksetDropZone?: { x: number; y: number };

  constructor(seed: number) {
    this.rng = new RNG(seed);
    this.biomass = Array.from({ length: this.biomassResolution }, () => Array.from({ length: this.biomassResolution }, () => 1));
    this.biomassCapacity = Array.from({ length: this.biomassResolution }, () => Array.from({ length: this.biomassResolution }, () => 1));
    this.biomassCooldown = Array.from({ length: this.biomassResolution }, () => Array.from({ length: this.biomassResolution }, () => 0));
    this.harvestPressure = Array.from({ length: this.biomassResolution }, () => Array.from({ length: this.biomassResolution }, () => 0));
    this.moisture = Array.from({ length: this.biomassResolution }, () => Array.from({ length: this.biomassResolution }, () => 0.2));
    this.moistureCapacity = Array.from({ length: this.biomassResolution }, () => Array.from({ length: this.biomassResolution }, () => 1));
    this.initialize();
  }

  initialize(): void {
    this.objects.clear();
    this.spatialGrid.clear();
    this.logs.length = 0;
    this.woodGained = 0;
    this.lastStrikeArc = undefined;
    this.predictedStrikeArc = undefined;
    this.predictedStrikeDamage = undefined;
    this.actualStrikeDamage = undefined;
    this.predictionRealityOverlay = undefined;
    this.lastInteractionOutcome = undefined;
    this.nextObjectId = 1;
    this.stations.clear();
    this.stationDurability.clear();
    this.stationFootprint.clear();
    this.lastMeasurements.clear();
    this.agent.pos = { x: 5, y: 5 };
    this.agent.heldObjectId = undefined;
    this.agent.energy = 1;
    for (let gy = 0; gy < this.biomassResolution; gy++) {
      for (let gx = 0; gx < this.biomassResolution; gx++) {
        const cx = (gx + 0.5) / this.biomassResolution;
        const cy = (gy + 0.5) / this.biomassResolution;
        const centerFalloff = Math.min(1, Math.hypot(cx - 0.5, cy - 0.5) * 1.4);
        const carryingCap = 0.58 + centerFalloff * 0.55;
        this.biomassCapacity[gy][gx] = carryingCap;
        this.biomass[gy][gx] = carryingCap;
        this.biomassCooldown[gy][gx] = 0;
        this.harvestPressure[gy][gx] = 0;
      }
    }
    this.seedMoistureField();

    for (let i = 0; i < 10; i++) {
      const d = materialDistributions[i % materialDistributions.length];
      this.addObject({
        pos: { x: this.rng.range(3.5, 6.5), y: this.rng.range(3.5, 6.5) },
        shapeType: this.sampleShapeType(),
        radius: this.rng.range(0.12, 0.34),
        length: this.rng.range(0.2, 2),
        thickness: this.rng.range(0.12, 0.45),
        center_of_mass_offset: { x: this.rng.normal(0, 0.08), y: this.rng.normal(0, 0.04) },
        integrity: this.rng.range(0.55, 1),
        props: sampleProperties(d, this.rng),
        debugFamily: d.debugFamily,
      });
    }

    this.addObject({
      pos: { x: this.rng.range(4.2, 5.8), y: this.rng.range(4.2, 5.8) },
      shapeType: 'rod',
      radius: 0.22,
      length: 1.9,
      thickness: 0.22,
      center_of_mass_offset: { x: 0.08, y: 0 },
      integrity: 0.94,
      props: sampleProperties(materialDistributions[1], this.rng),
      debugFamily: 'long-fiber',
    });

    this.spawnTarget();
  }

  spawnTarget(): ObjID {
    return this.addObject({
      pos: { x: this.rng.range(4.4, 5.6), y: this.rng.range(4.4, 5.6) },
      shapeType: 'plate',
      radius: 0.42,
      length: this.rng.range(1.2, 1.8),
      thickness: this.rng.range(0.3, 0.55),
      center_of_mass_offset: { x: this.rng.normal(0, 0.04), y: this.rng.normal(0, 0.04) },
      integrity: 1,
      props: fibrousTargetProps(this.rng),
      debugFamily: 'target-visual',
    });
  }

  spawnLooseObject(): ObjID {
    const d = materialDistributions[this.rng.int(0, materialDistributions.length)];
    return this.addObject({
      pos: { x: this.rng.range(3.6, 6.4), y: this.rng.range(3.6, 6.4) },
      shapeType: this.sampleShapeType(),
      radius: this.rng.range(0.1, 0.32),
      length: this.rng.range(0.2, 1.6),
      thickness: this.rng.range(0.1, 0.4),
      center_of_mass_offset: { x: this.rng.normal(0, 0.06), y: this.rng.normal(0, 0.04) },
      integrity: this.rng.range(0.5, 1),
      props: sampleProperties(d, this.rng),
      debugFamily: d.debugFamily,
    });
  }

  private sampleShapeType(): ShapeType {
    const roll = this.rng.float();
    if (roll < 0.28) return 'sphere';
    if (roll < 0.62) return 'rod';
    if (roll < 0.82) return 'shard';
    return 'plate';
  }

  private addObject(
    input: Omit<WorldObject, 'id' | 'vel' | 'grip_score' | 'latentPrecision' | 'processHistory'> &
      Partial<Pick<WorldObject, 'latentPrecision' | 'processHistory'>>,
  ): ObjID {
    const id = this.nextObjectId++;
    const obj: WorldObject = {
      ...input,
      id,
      vel: { x: 0, y: 0 },
      grip_score: deriveGripScore(input.shapeType, input.length, input.thickness, input.props.roughness),
      latentPrecision: input.latentPrecision ?? {
        surface_planarity: this.rng.range(0.25, 0.7),
        impurity_level: this.rng.range(0.2, 0.75),
        microstructure_order: this.rng.range(0.2, 0.7),
        internal_stress: this.rng.range(0.2, 0.65),
        feature_resolution_limit: this.rng.range(0.2, 0.65),
      },
      processHistory: input.processHistory ?? {
        grind_passes: 0,
        thermal_cycles: 0,
        soak_cycles: 0,
      },
    };
    this.objects.set(id, obj);
    this.spatialGrid.upsert(obj);
    return id;
  }

  private getObject(id: ObjID): WorldObject | undefined {
    return this.objects.get(id);
  }

  getTargetId(): ObjID | undefined {
    let best: { id: ObjID; score: number } | undefined;
    for (const obj of this.objects.values()) {
      const score = fibrousTargetScore(obj.props) * obj.integrity;
      if (!best || score > best.score) best = { id: obj.id, score };
    }
    return best?.id;
  }

  getNearbyObjectIds(radius = 2.5): ObjID[] {
    return this.spatialGrid
      .queryRadius(this.agent.pos.x, this.agent.pos.y, radius)
      .map(obj => obj.id);
  }

  apply(action: PrimitiveVerb): void {
    this.lastInteractionOutcome = undefined;
    this.trimLogs();
    if (action.type === 'MOVE_TO') {
      if (!Number.isFinite(action.x) || !Number.isFinite(action.y)) return;
      const tx = Math.max(0, Math.min(this.width, action.x));
      const ty = Math.max(0, Math.min(this.height, action.y));
      const dx = tx - this.agent.pos.x;
      const dy = ty - this.agent.pos.y;
      const dist = Math.hypot(dx, dy);
      const maxStep = World.MAX_AGENT_STEP_DISTANCE;
      const scale = dist > maxStep ? maxStep / Math.max(0.0001, dist) : 1;
      this.agent.pos.x += dx * scale;
      this.agent.pos.y += dy * scale;
      if (this.agent.heldObjectId) {
        const held = this.getObject(this.agent.heldObjectId);
        if (held) {
          held.pos = { ...this.agent.pos };
          this.spatialGrid.upsert(held);
        }
      }
      return;
    }

    if (action.type === 'PICK_UP') {
      const obj = this.getObject(action.objId);
      if (!obj) return;
      if (Math.hypot(obj.pos.x - this.agent.pos.x, obj.pos.y - this.agent.pos.y) > World.MAX_PICKUP_DISTANCE) return;
      this.agent.heldObjectId = obj.id;
      obj.heldBy = this.agent.id;
      obj.pos = { ...this.agent.pos };
      return;
    }

    if (action.type === 'DROP') {
      if (!this.agent.heldObjectId) return;
      const held = this.getObject(this.agent.heldObjectId);
      if (held) {
        held.heldBy = undefined;
        held.pos = { ...this.agent.pos };
      }
      this.agent.heldObjectId = undefined;
      return;
    }

    if (action.type === 'BIND_TO') {
      if (!this.agent.heldObjectId) return;
      const held = this.getObject(this.agent.heldObjectId);
      const other = this.getObject(action.objId);
      if (!held || !other || held.id === other.id) return;
      const result = bindObjects(held, other, this.nextObjectId++, this.rng);
      result.composite.heldBy = this.agent.id;
      this.objects.delete(held.id);
      this.spatialGrid.remove(held.id);
      this.objects.delete(other.id);
      this.spatialGrid.remove(other.id);
      this.objects.set(result.composite.id, result.composite);
      this.spatialGrid.upsert(result.composite);
      this.agent.heldObjectId = result.composite.id;
      this.logs.unshift(`BIND ${held.id}+${other.id} -> ${result.composite.id} (q=${result.bindingQuality.toFixed(2)})`);
      this.lastInteractionOutcome = {
        action: 'BIND_TO',
        toolId: result.composite.id,
        targetId: other.id,
        damage: 0,
        toolWear: Math.max(0, 1 - result.composite.integrity),
        fragments: 0,
        propertyChanges: result.bindingQuality,
      };
      return;
    }

    if (action.type === 'STRIKE_WITH') {
      if (!this.agent.heldObjectId) return;
      const tool = this.getObject(this.agent.heldObjectId);
      const target = this.getObject(action.targetId);
      if (!tool || !target || tool.id === target.id) return;
      const targetIntegrityBefore = target.integrity;
      const toolIntegrityBefore = tool.integrity;
      const result = strike(tool, target, this.rng, () => this.nextObjectId++);
      const angle = Math.atan2(target.pos.y - tool.pos.y, target.pos.x - tool.pos.x);
      this.lastStrikeArc = {
        center: { ...tool.pos },
        radius: Math.max(0.4, tool.length * 0.5),
        start: angle - 0.55,
        end: angle + 0.25,
        alpha: Math.min(1, 0.45 + result.damage),
      };
      if (result.fractured) {
        const localBiomass = this.consumeBiomassAt(target.pos.x, target.pos.y, 0.22);
        for (const fragment of result.fragments) {
          this.objects.set(fragment.id, fragment);
          this.spatialGrid.upsert(fragment);
        }
        const distanceFromCenter = Math.hypot(target.pos.x - this.width * 0.5, target.pos.y - this.height * 0.5) / Math.max(1, this.width * 0.7);
        const travelYieldBoost = 1 + Math.max(0, distanceFromCenter) * 0.25;
        const woodYield = Math.round(localBiomass * travelYieldBoost * 1000) / 1000;
        this.woodGained += woodYield;
        this.logs.unshift(`STRIKE ${tool.id} -> target ${target.id} dmg=${result.damage.toFixed(2)} wood+=${woodYield.toFixed(2)}`);
        this.objects.delete(target.id);
        this.spatialGrid.remove(target.id);
        this.spawnTarget();
      } else {
        this.logs.unshift(`STRIKE ${tool.id} -> target ${target.id} dmg=${result.damage.toFixed(2)}`);
      }
      this.actualStrikeDamage = result.damage;
      this.lastInteractionOutcome = {
        action: 'STRIKE_WITH',
        toolId: tool.id,
        targetId: target.id,
        damage: result.damage,
        toolWear: Math.max(0, toolIntegrityBefore - tool.integrity),
        fragments: result.fragments.length,
        propertyChanges: Math.abs(targetIntegrityBefore - target.integrity),
      };
      return;
    }

    if (action.type === 'GRIND') {
      if (!this.agent.heldObjectId) return;
      const held = this.getObject(this.agent.heldObjectId);
      const abrasive = this.getObject(action.abrasiveId);
      if (!held || !abrasive || held.id === abrasive.id) return;
      const beforeSharpness = held.props.sharpness;
      const station = this.stations.get(held.id);
      const stationBoost = station ? station.bonuses.maxPlanarityGain * 0.4 : 0;
      const result = grind(held, abrasive, action.intensity ?? 0.75 + stationBoost);
      result.newObject.heldBy = this.agent.id;
      this.objects.set(held.id, result.newObject);
      this.logs.unshift(`GRIND ${held.id} with ${abrasive.id} wear=${result.wear.toFixed(2)}`);
      this.lastInteractionOutcome = {
        action: 'GRIND',
        toolId: held.id,
        targetId: abrasive.id,
        damage: 0,
        toolWear: result.wear,
        fragments: 0,
        propertyChanges: Math.abs(result.newObject.props.sharpness - beforeSharpness),
      };
      return;
    }

    if (action.type === 'HEAT') {
      if (!this.agent.heldObjectId) return;
      if (!Number.isFinite(action.intensity)) return;
      const held = this.getObject(this.agent.heldObjectId);
      if (!held) return;
      const updated = heat(held, action.intensity);
      updated.heldBy = this.agent.id;
      this.objects.set(held.id, updated);
      this.logs.unshift(`HEAT ${held.id} i=${action.intensity.toFixed(2)}`);
      return;
    }

    if (action.type === 'SOAK') {
      if (!this.agent.heldObjectId) return;
      if (!Number.isFinite(action.intensity)) return;
      const held = this.getObject(this.agent.heldObjectId);
      if (!held) return;
      const updated = soak(held, action.intensity);
      updated.heldBy = this.agent.id;
      this.objects.set(held.id, updated);
      this.logs.unshift(`SOAK ${held.id} i=${action.intensity.toFixed(2)}`);
      return;
    }

    if (action.type === 'COOL') {
      if (!this.agent.heldObjectId) return;
      if (!Number.isFinite(action.intensity)) return;
      const held = this.getObject(this.agent.heldObjectId);
      if (!held) return;
      const updated = cool(held, action.intensity);
      updated.heldBy = this.agent.id;
      this.objects.set(held.id, updated);
      this.logs.unshift(`COOL ${held.id} i=${action.intensity.toFixed(2)}`);
      return;
    }

    if (action.type === 'ANCHOR') {
      if (!this.agent.heldObjectId) return;
      const held = this.getObject(this.agent.heldObjectId);
      if (!held) return;
      const worldPos = action.worldPos ?? { ...this.agent.pos };
      if (this.stationFootprint.size >= World.MAX_STRUCTURE_TILES) {
        this.logs.unshift('ANCHOR blocked: structure budget exceeded');
        return;
      }
      const isAdjacent = [...this.stations.values()].some((station) => Math.hypot(station.worldPos.x - worldPos.x, station.worldPos.y - worldPos.y) <= 2.35);
      const localResourceSupport = this.biomassAt(worldPos.x, worldPos.y) > 0.72 || this.moistureAt(worldPos.x, worldPos.y) > 0.58;
      if (this.stations.size > 0 && !isAdjacent && !localResourceSupport) {
        this.logs.unshift('ANCHOR blocked: must connect to logistics/resource zone');
        return;
      }
      const anchored = anchorObject(held, worldPos);
      this.objects.set(held.id, anchored);
      this.agent.heldObjectId = undefined;
      anchored.heldBy = undefined;
      const station = stationFromAnchoredObject(anchored);
      station.functionType = this.chooseStationFunction(worldPos);
      this.stations.set(anchored.id, station);
      this.stationDurability.set(anchored.id, 1);
      this.stationFootprint.set(anchored.id, 1 + this.stations.size * 0.08);
      this.logs.unshift(`ANCHOR ${held.id} q=${station.quality.toFixed(2)} fn=${station.functionType}`);
      return;
    }
  }

  private trimLogs(): void {
    if (this.logs.length > World.MAX_LOG_ENTRIES) {
      this.logs.length = World.MAX_LOG_ENTRIES;
    }
  }

  private biomassCellFor(x: number, y: number): { gx: number; gy: number } {
    return {
      gx: Math.min(this.biomassResolution - 1, Math.max(0, Math.floor((x / Math.max(0.001, this.width)) * this.biomassResolution))),
      gy: Math.min(this.biomassResolution - 1, Math.max(0, Math.floor((y / Math.max(0.001, this.height)) * this.biomassResolution))),
    };
  }

  biomassAt(x: number, y: number): number {
    const { gx, gy } = this.biomassCellFor(x, y);
    return this.biomass[gy]?.[gx] ?? 1;
  }

  consumeBiomassAt(x: number, y: number, amount: number): number {
    const { gx, gy } = this.biomassCellFor(x, y);
    const current = this.biomass[gy]?.[gx] ?? 1;
    const pressure = this.harvestPressure[gy]?.[gx] ?? 0;
    const requested = Math.max(0, amount);
    const overfarmPenalty = Math.max(0.3, 1 - pressure * 0.7);
    const harvested = Math.min(current, requested) * overfarmPenalty;
    const next = Math.max(0, current - harvested);
    if (this.biomass[gy]) this.biomass[gy][gx] = next;
    if (this.harvestPressure[gy]) this.harvestPressure[gy][gx] = Math.min(1, pressure + requested * 0.65);
    if (this.biomassCooldown[gy]) this.biomassCooldown[gy][gx] = Math.max(this.biomassCooldown[gy][gx], 28);
    return harvested;
  }

  regrowBiomass(amount: number): void {
    const rate = Math.max(0, amount);
    for (let y = 0; y < this.biomassResolution; y++) {
      for (let x = 0; x < this.biomassResolution; x++) {
        const cooldown = this.biomassCooldown[y][x];
        if (cooldown > 0) {
          this.biomassCooldown[y][x] = cooldown - 1;
        } else {
          const pressure = this.harvestPressure[y][x];
          const carry = this.biomassCapacity[y][x];
          const regenRate = rate * Math.max(0.2, 1 - pressure * 0.5);
          this.biomass[y][x] = Math.min(carry, this.biomass[y][x] + regenRate);
        }
        this.harvestPressure[y][x] = Math.max(0, this.harvestPressure[y][x] - rate * 0.55);
      }
    }
  }

  moistureAt(x: number, y: number): number {
    const { gx, gy } = this.biomassCellFor(x, y);
    return this.moisture[gy]?.[gx] ?? 0.2;
  }

  regrowMoisture(amount: number): void {
    const rate = Math.max(0, amount);
    for (let y = 0; y < this.biomassResolution; y++) {
      for (let x = 0; x < this.biomassResolution; x++) {
        const cap = this.moistureCapacity[y][x];
        this.moisture[y][x] = Math.min(cap, this.moisture[y][x] + rate * 0.4);
      }
    }
    this.seedMoistureField(0.015);
  }

  nearestWaterDistance(x: number, y: number): number {
    return this.waterSources.reduce((best, src) => Math.min(best, Math.hypot(src.x - x, src.y - y)), Number.POSITIVE_INFINITY);
  }

  nearestWaterSource(x: number, y: number): { x: number; y: number } {
    let best = this.waterSources[0];
    let bestDist = Number.POSITIVE_INFINITY;
    for (const source of this.waterSources) {
      const dist = Math.hypot(source.x - x, source.y - y);
      if (dist < bestDist) {
        bestDist = dist;
        best = source;
      }
    }
    return { ...best };
  }

  richestBiomassCellCenter(): { x: number; y: number } {
    let best = { gx: 0, gy: 0, value: -1 };
    for (let gy = 0; gy < this.biomassResolution; gy++) {
      for (let gx = 0; gx < this.biomassResolution; gx++) {
        const value = (this.biomass[gy]?.[gx] ?? 0) - (this.harvestPressure[gy]?.[gx] ?? 0) * 0.3;
        if (value > best.value) best = { gx, gy, value };
      }
    }
    return {
      x: ((best.gx + 0.5) / this.biomassResolution) * this.width,
      y: ((best.gy + 0.5) / this.biomassResolution) * this.height,
    };
  }

  biomassStats(): { avg: number; min: number } {
    const values = this.biomass.flat();
    const avg = values.reduce((sum, value) => sum + value, 0) / Math.max(1, values.length);
    const min = values.reduce((best, value) => Math.min(best, value), 1);
    return { avg, min };
  }

  targetYieldStats(): { targetsAlive: number; avgTargetYield: number } {
    const targets = [...this.objects.values()].filter((entry) => entry.debugFamily === 'target-visual');
    const avgTargetYield =
      targets.reduce((sum, target) => {
        const { gx, gy } = this.biomassCellFor(target.pos.x, target.pos.y);
        const pressure = this.harvestPressure[gy]?.[gx] ?? 0;
        return sum + this.biomassAt(target.pos.x, target.pos.y) * Math.max(0.35, 1 - pressure * 0.6);
      }, 0) / Math.max(1, targets.length);
    return { targetsAlive: targets.length, avgTargetYield };
  }

  measureObject(
    objId: ObjID,
    instrumentId?: ObjID,
  ): MeasurementResult[] {
    const obj = this.getObject(objId);
    if (!obj) return [];
    const instrument = instrumentId ? this.getObject(instrumentId) : undefined;
    const station = [...this.stations.values()].find((entry) => Math.hypot(entry.worldPos.x - obj.pos.x, entry.worldPos.y - obj.pos.y) <= 1.2);
    const out = [
      measureMass(obj, this.rng, instrument, station),
      measureGeometry(obj, this.rng, instrument, station),
      measureHardness(obj, this.rng, instrument, station),
      measureConductivity(obj, this.rng, instrument, station),
      measureOptical(obj, this.rng, instrument, station),
    ];
    this.lastMeasurements.set(objId, out);
    return out;
  }

  recomputeStations(): void {
    for (const stationId of [...this.stations.keys()]) {
      const anchored = this.objects.get(stationId);
      if (!anchored || !anchored.anchored) {
        this.stations.delete(stationId);
        this.stationDurability.delete(stationId);
        this.stationFootprint.delete(stationId);
        continue;
      }
      const previous = this.stations.get(stationId);
      const durability = this.stationDurability.get(stationId) ?? 1;
      const updated = stationFromAnchoredObject(anchored);
      updated.functionType = previous?.functionType ?? updated.functionType;
      const debrisPenalty = this.localDebrisPenalty(updated.worldPos.x, updated.worldPos.y);
      updated.quality *= Math.max(0.15, durability - debrisPenalty * 0.2);
      updated.bonuses.maxPlanarityGain *= Math.max(0.2, durability);
      this.stations.set(stationId, updated);
    }
  }

  tickStructureDecay(): void {
    for (const station of this.stations.values()) {
      const durability = this.stationDurability.get(station.objectId) ?? 1;
      const debrisPenalty = this.localDebrisPenalty(station.worldPos.x, station.worldPos.y);
      const decay = 0.0018 + debrisPenalty * 0.0014;
      this.stationDurability.set(station.objectId, Math.max(0.1, durability - decay));
    }
  }

  maintainNearestStation(x: number, y: number, amount = 0.04): number {
    let best: { id: number; dist: number } | undefined;
    for (const station of this.stations.values()) {
      const dist = Math.hypot(station.worldPos.x - x, station.worldPos.y - y);
      if (!best || dist < best.dist) best = { id: station.objectId, dist };
    }
    if (!best || best.dist > 2.2) return 0;
    const current = this.stationDurability.get(best.id) ?? 1;
    const next = Math.min(1, current + Math.max(0, amount));
    this.stationDurability.set(best.id, next);
    return next - current;
  }

  stationMaintenancePressure(): number {
    if (!this.stations.size) return 0;
    const avgDurability = [...this.stationDurability.values()].reduce((sum, value) => sum + value, 0) / Math.max(1, this.stationDurability.size);
    return Math.max(0, 1 - avgDurability);
  }

  structureSupportLogistics(): number {
    const storageCount = [...this.stations.values()].filter((entry) => entry.functionType === 'storage').length;
    return Math.min(0.5, storageCount * 0.08);
  }

  /**
   * Apply slow weathering to all non-anchored objects.
   * Objects gradually lose integrity and surface quality over time,
   * simulating environmental wear. Anchored objects and held objects
   * weather at a reduced rate.
   */
  weatherObjects(tickDelta = 1): void {
    const baseDecay = 0.0002 * tickDelta;
    for (const obj of this.objects.values()) {
      obj.ageTicks = (obj.ageTicks ?? 0) + tickDelta;
      if (obj.anchored || obj.heldBy !== undefined) continue; // sheltered objects don't weather
      const moistureEffect = this.moistureAt(obj.pos.x, obj.pos.y);
      const weatherRate = baseDecay * (1 + moistureEffect * 0.5 + obj.props.porosity * 0.3);
      obj.integrity = Math.max(0.01, obj.integrity - weatherRate);
      obj.latentPrecision.surface_planarity = Math.max(0, obj.latentPrecision.surface_planarity - weatherRate * 0.3);
    }
  }

  private chooseStationFunction(worldPos: { x: number; y: number }): StationFunction {
    if (this.moistureAt(worldPos.x, worldPos.y) > 0.62) return 'purifier';
    if (this.biomassAt(worldPos.x, worldPos.y) > 0.9) return 'storage';
    if (this.stations.size % 3 === 0) return 'beacon';
    return 'workshop';
  }

  private localDebrisPenalty(x: number, y: number): number {
    const nearbyDebris = [...this.objects.values()].filter((entry) => entry.debugFamily === 'fragment' && Math.hypot(entry.pos.x - x, entry.pos.y - y) <= 1.8).length;
    return Math.min(0.5, nearbyDebris * 0.04);
  }

  private seedMoistureField(boost = 0.03): void {
    for (let gy = 0; gy < this.biomassResolution; gy++) {
      for (let gx = 0; gx < this.biomassResolution; gx++) {
        const x = ((gx + 0.5) / this.biomassResolution) * this.width;
        const y = ((gy + 0.5) / this.biomassResolution) * this.height;
        const nearest = this.nearestWaterDistance(x, y);
        const sourceInfluence = Math.max(0, 1 - nearest / 5.5);
        const cap = 0.12 + sourceInfluence * 0.88;
        this.moistureCapacity[gy][gx] = cap;
        this.moisture[gy][gx] = Math.min(cap, this.moisture[gy][gx] + boost * sourceInfluence);
      }
    }
  }
}
