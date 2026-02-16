import { deriveGripScore, type ObjID, type ShapeType, type WorldObject } from './object_model';
import { bindObjects, grind, strike } from './interactions';
import { fibrousTargetProps, materialDistributions, sampleProperties } from './material_distributions';
import { fibrousTargetScore } from './properties';
import { RNG } from './rng';

export type PrimitiveVerb =
  | { type: 'MOVE_TO'; x: number; y: number }
  | { type: 'PICK_UP'; objId: ObjID }
  | { type: 'DROP' }
  | { type: 'BIND_TO'; objId: ObjID }
  | { type: 'STRIKE_WITH'; targetId: ObjID }
  | { type: 'GRIND'; abrasiveId: ObjID }
  | { type: 'HEAT'; intensity: number }
  | { type: 'SOAK'; intensity: number };

export interface AgentState {
  id: number;
  pos: { x: number; y: number };
  heldObjectId?: ObjID;
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

export class World {
  readonly width = 10;
  readonly height = 10;
  readonly rng: RNG;
  readonly objects = new Map<ObjID, WorldObject>();
  readonly logs: string[] = [];
  lastStrikeArc?: StrikeArc;
  predictedStrikeArc?: StrikeArc;
  predictedStrikeDamage?: number;
  actualStrikeDamage?: number;
  lastInteractionOutcome?: InteractionOutcome;
  readonly agent: AgentState = { id: 1, pos: { x: 5, y: 5 } };
  nextObjectId = 1;
  woodGained = 0;

  constructor(seed: number) {
    this.rng = new RNG(seed);
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
    this.lastInteractionOutcome = undefined;
    this.nextObjectId = 1;
    this.agent.pos = { x: 5, y: 5 };
    this.agent.heldObjectId = undefined;

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

  private sampleShapeType(): ShapeType {
    const roll = this.rng.float();
    if (roll < 0.28) return 'sphere';
    if (roll < 0.62) return 'rod';
    if (roll < 0.82) return 'shard';
    return 'plate';
  }

  private addObject(input: Omit<WorldObject, 'id' | 'vel' | 'grip_score'>): ObjID {
    const id = this.nextObjectId++;
    this.objects.set(id, {
      ...input,
      id,
      vel: { x: 0, y: 0 },
      grip_score: deriveGripScore(input.shapeType, input.length, input.thickness, input.props.roughness),
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
      this.agent.pos.x = Math.max(0, Math.min(this.width, action.x));
      this.agent.pos.y = Math.max(0, Math.min(this.height, action.y));
      return;
    }

    if (action.type === 'PICK_UP') {
      const obj = this.getObject(action.objId);
      if (!obj) return;
      this.agent.heldObjectId = obj.id;
      obj.heldBy = this.agent.id;
      return;
    }

    if (action.type === 'DROP') {
      if (!this.agent.heldObjectId) return;
      const held = this.getObject(this.agent.heldObjectId);
      if (held) held.heldBy = undefined;
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
        for (const fragment of result.fragments) this.objects.set(fragment.id, fragment);
        this.woodGained += 1;
        this.logs.unshift(`STRIKE ${tool.id} -> target ${target.id} dmg=${result.damage.toFixed(2)} wood+1`);
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
      const result = grind(held, abrasive);
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
  }
}
