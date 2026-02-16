import { deriveGripScore, type ObjID, type ShapeType, type WorldObject } from './object_model';
import { bindObjects, cool, grind, heat, soak, strike } from './interactions';
import { fibrousTargetProps, materialDistributions, sampleProperties } from './material_distributions';
import { fibrousTargetScore } from './properties';
import { RNG } from './rng';
import { type MeasurementResult, measureConductivity, measureGeometry, measureHardness, measureMass, measureOptical } from './metrology';
import { anchorObject, stationFromAnchoredObject, type AnchoredStation } from './stations';

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
  readonly width = 10;
  readonly height = 10;
  readonly biomassResolution = 5;
  readonly rng: RNG;
  readonly objects = new Map<ObjID, WorldObject>();
  readonly logs: string[] = [];
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
  readonly lastMeasurements = new Map<ObjID, MeasurementResult[]>();
  readonly biomass: number[][];
  worksetDebugIds: ObjID[] = [];
  worksetDropZone?: { x: number; y: number };

  constructor(seed: number) {
    this.rng = new RNG(seed);
    this.biomass = Array.from({ length: this.biomassResolution }, () => Array.from({ length: this.biomassResolution }, () => 1));
    this.initialize();
  }

  initialize(): void {
    this.objects.clear();
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
    this.lastMeasurements.clear();
    this.agent.pos = { x: 5, y: 5 };
    this.agent.heldObjectId = undefined;
    this.agent.energy = 1;
    for (const row of this.biomass) row.fill(1);

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
    this.objects.set(id, {
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
    });
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
    const out: ObjID[] = [];
    for (const obj of this.objects.values()) {
      const dx = obj.pos.x - this.agent.pos.x;
      const dy = obj.pos.y - this.agent.pos.y;
      if (Math.hypot(dx, dy) <= radius) out.push(obj.id);
    }
    return out;
  }

  apply(action: PrimitiveVerb): void {
    this.lastInteractionOutcome = undefined;
    if (action.type === 'MOVE_TO') {
      const tx = Math.max(0, Math.min(this.width, action.x));
      const ty = Math.max(0, Math.min(this.height, action.y));
      const dx = tx - this.agent.pos.x;
      const dy = ty - this.agent.pos.y;
      const dist = Math.hypot(dx, dy);
      const maxStep = 0.65;
      const scale = dist > maxStep ? maxStep / Math.max(0.0001, dist) : 1;
      this.agent.pos.x += dx * scale;
      this.agent.pos.y += dy * scale;
      if (this.agent.heldObjectId) {
        const held = this.getObject(this.agent.heldObjectId);
        if (held) held.pos = { ...this.agent.pos };
      }
      return;
    }

    if (action.type === 'PICK_UP') {
      const obj = this.getObject(action.objId);
      if (!obj) return;
      if (Math.hypot(obj.pos.x - this.agent.pos.x, obj.pos.y - this.agent.pos.y) > 2.5) return;
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
      this.objects.delete(other.id);
      this.objects.set(result.composite.id, result.composite);
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
        for (const fragment of result.fragments) this.objects.set(fragment.id, fragment);
        const woodYield = Math.round(localBiomass * 1000) / 1000;
        this.woodGained += woodYield;
        this.logs.unshift(`STRIKE ${tool.id} -> target ${target.id} dmg=${result.damage.toFixed(2)} wood+${woodYield.toFixed(2)}`);
        this.objects.delete(target.id);
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
      const anchored = anchorObject(held, worldPos);
      this.objects.set(held.id, anchored);
      this.agent.heldObjectId = undefined;
      anchored.heldBy = undefined;
      const station = stationFromAnchoredObject(anchored);
      this.stations.set(anchored.id, station);
      this.logs.unshift(`ANCHOR ${held.id} q=${station.quality.toFixed(2)}`);
      return;
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
    const next = Math.max(0, current - Math.max(0, amount));
    if (this.biomass[gy]) this.biomass[gy][gx] = next;
    return next;
  }

  regrowBiomass(amount: number): void {
    const rate = Math.max(0, amount);
    for (let y = 0; y < this.biomassResolution; y++) {
      for (let x = 0; x < this.biomassResolution; x++) {
        this.biomass[y][x] = Math.min(1, this.biomass[y][x] + rate);
      }
    }
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
      targets.reduce((sum, target) => sum + this.biomassAt(target.pos.x, target.pos.y), 0) / Math.max(1, targets.length);
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
        continue;
      }
      this.stations.set(stationId, stationFromAnchoredObject(anchored));
    }
  }
}
