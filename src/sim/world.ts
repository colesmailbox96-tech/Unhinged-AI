import type { ObjID, WorldObject } from './object_model';
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

export class World {
  readonly width = 10;
  readonly height = 10;
  readonly rng: RNG;
  readonly objects = new Map<ObjID, WorldObject>();
  readonly logs: string[] = [];
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
    this.nextObjectId = 1;
    this.agent.pos = { x: 5, y: 5 };
    this.agent.heldObjectId = undefined;

    for (let i = 0; i < 10; i++) {
      const d = materialDistributions[i % materialDistributions.length];
      this.addObject({
        pos: { x: this.rng.range(3.5, 6.5), y: this.rng.range(3.5, 6.5) },
        radius: this.rng.range(0.12, 0.34),
        length: this.rng.range(0.2, 2),
        integrity: this.rng.range(0.55, 1),
        props: sampleProperties(d, this.rng),
        debugFamily: d.debugFamily,
      });
    }

    this.addObject({
      pos: { x: this.rng.range(4.2, 5.8), y: this.rng.range(4.2, 5.8) },
      radius: 0.22,
      length: 1.9,
      integrity: 0.94,
      props: sampleProperties(materialDistributions[1], this.rng),
      debugFamily: 'long-fiber',
    });

    this.spawnTarget();
  }

  spawnTarget(): ObjID {
    return this.addObject({
      pos: { x: this.rng.range(4.4, 5.6), y: this.rng.range(4.4, 5.6) },
      radius: 0.42,
      length: this.rng.range(1.2, 1.8),
      integrity: 1,
      props: fibrousTargetProps(this.rng),
      debugFamily: 'target-visual',
    });
  }

  private addObject(input: Omit<WorldObject, 'id' | 'vel'>): ObjID {
    const id = this.nextObjectId++;
    this.objects.set(id, { ...input, id, vel: { x: 0, y: 0 } });
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
      return;
    }

    if (action.type === 'STRIKE_WITH') {
      if (!this.agent.heldObjectId) return;
      const tool = this.getObject(this.agent.heldObjectId);
      const target = this.getObject(action.targetId);
      if (!tool || !target || tool.id === target.id) return;
      const result = strike(tool, target, this.rng, () => this.nextObjectId++);
      if (result.fractured) {
        for (const fragment of result.fragments) this.objects.set(fragment.id, fragment);
        this.woodGained += 1;
        this.logs.unshift(`STRIKE ${tool.id} -> target ${target.id} dmg=${result.damage.toFixed(2)} wood+1`);
        this.objects.delete(target.id);
        this.spawnTarget();
      } else {
        this.logs.unshift(`STRIKE ${tool.id} -> target ${target.id} dmg=${result.damage.toFixed(2)}`);
      }
      return;
    }

    if (action.type === 'GRIND') {
      if (!this.agent.heldObjectId) return;
      const held = this.getObject(this.agent.heldObjectId);
      const abrasive = this.getObject(action.abrasiveId);
      if (!held || !abrasive || held.id === abrasive.id) return;
      const result = grind(held, abrasive);
      result.newObject.heldBy = this.agent.id;
      this.objects.set(held.id, result.newObject);
      this.logs.unshift(`GRIND ${held.id} with ${abrasive.id} wear=${result.wear.toFixed(2)}`);
      return;
    }
  }
}
