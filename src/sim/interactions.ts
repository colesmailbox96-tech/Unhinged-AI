import { deriveGripScore, estimateContactArea, type ObjID, type WorldObject } from './object_model';
import {
  EPSILON,
  clamp,
  combineVectors,
  cuttingPotential,
  impactPotential,
  mutate,
} from './properties';
import { RNG } from './rng';

export interface BindResult {
  composite: WorldObject;
  bindingQuality: number;
}

export interface StrikeResult {
  damage: number;
  fractured: boolean;
  fragments: WorldObject[];
}

export interface GrindResult {
  wear: number;
  newObject: WorldObject;
}

function latentOf(obj: WorldObject): WorldObject['latentPrecision'] {
  return (
    obj.latentPrecision ?? {
      surface_planarity: 0.5,
      impurity_level: 0.5,
      microstructure_order: 0.5,
      internal_stress: 0.5,
      feature_resolution_limit: 0.5,
    }
  );
}

function historyOf(obj: WorldObject): WorldObject['processHistory'] {
  return (
    obj.processHistory ?? {
      grind_passes: 0,
      thermal_cycles: 0,
      soak_cycles: 0,
    }
  );
}

function contactProxy(a: WorldObject, b: WorldObject): number {
  const area = Math.min(
    estimateContactArea(a.shapeType, a.length, a.thickness, a.radius),
    estimateContactArea(b.shapeType, b.length, b.thickness, b.radius),
  );
  return clamp((Math.sqrt(area) + Math.min(a.length, b.length) * 0.15) / 1.2);
}

export function bindObjects(a: WorldObject, b: WorldObject, id: ObjID, rng: RNG): BindResult {
  const latentA = latentOf(a);
  const latentB = latentOf(b);
  const historyA = historyOf(a);
  const historyB = historyOf(b);
  const contact = contactProxy(a, b);
  const bindingSurface = clamp((a.props.roughness + b.props.roughness) * 0.3 + (a.props.friction_coeff + b.props.friction_coeff) * 0.25 + contact * 0.45);
  const bindingQuality = clamp(
    bindingSurface * 0.35 +
      ((a.props.tensile_strength + b.props.tensile_strength) * 0.5) * 0.25 +
      ((a.props.elasticity + b.props.elasticity) * 0.5) * 0.2 +
      ((a.props.friction_coeff + b.props.friction_coeff) * 0.5) * 0.2,
  );

  const massWeight = clamp((a.props.mass + EPSILON) / (a.props.mass + b.props.mass + EPSILON));
  const props = mutate(combineVectors(a.props, b.props, { mass: massWeight, hardness: massWeight }), (1 - bindingQuality) * 0.01, rng);
  const totalMass = a.props.mass + b.props.mass + EPSILON;
  const center_of_mass_offset = {
    x:
      (a.center_of_mass_offset.x * a.props.mass +
        b.center_of_mass_offset.x * b.props.mass +
        (b.props.mass - a.props.mass) * 0.18) /
      totalMass,
    y: (a.center_of_mass_offset.y * a.props.mass + b.center_of_mass_offset.y * b.props.mass) / totalMass,
  };
  const length = Math.max(a.length, b.length);
  const gripPoint = { x: -length * 0.5, y: 0 };
  const leverArm = Math.max(
    0.2,
    Math.hypot(center_of_mass_offset.x - gripPoint.x, center_of_mass_offset.y - gripPoint.y),
  );
  const impactMultiplier = clamp(leverArm / 1.2, 0.5, 2.2);
  props.mass = clamp((a.props.mass + b.props.mass * 0.65) * impactMultiplier);
  props.hardness = clamp(Math.max(a.props.hardness, b.props.hardness) + bindingQuality * 0.08);

  const longBoost = Math.max(a.length, b.length) / (Math.min(a.length, b.length) + 0.2);
  props.sharpness = clamp(props.sharpness + clamp(longBoost - 1, 0, 0.25) * 0.1);

  const composite: WorldObject = {
    id,
    pos: { x: (a.pos.x + b.pos.x) / 2, y: (a.pos.y + b.pos.y) / 2 },
    vel: { x: 0, y: 0 },
    shapeType: a.shapeType === 'rod' || b.shapeType === 'rod' ? 'rod' : a.shapeType,
    radius: Math.max(a.radius, b.radius),
    length,
    thickness: Math.max(a.thickness, b.thickness),
    center_of_mass_offset,
    grip_score: deriveGripScore('rod', length, Math.max(a.thickness, b.thickness), props.roughness),
    props,
    integrity: clamp(Math.min(a.integrity, b.integrity) * (0.6 + bindingQuality * 0.5)),
    constituents: [a.id, b.id],
    debugFamily: 'bound-composite',
    anchored: false,
    latentPrecision: {
      surface_planarity: clamp((latentA.surface_planarity + latentB.surface_planarity) * 0.5 + bindingQuality * 0.04),
      impurity_level: clamp((latentA.impurity_level + latentB.impurity_level) * 0.5),
      microstructure_order: clamp((latentA.microstructure_order + latentB.microstructure_order) * 0.5 + bindingQuality * 0.03),
      internal_stress: clamp((latentA.internal_stress + latentB.internal_stress) * 0.5 + (1 - bindingQuality) * 0.05),
      feature_resolution_limit: clamp((latentA.feature_resolution_limit + latentB.feature_resolution_limit) * 0.5 + bindingQuality * 0.03),
    },
    processHistory: {
      grind_passes: historyA.grind_passes + historyB.grind_passes,
      thermal_cycles: historyA.thermal_cycles + historyB.thermal_cycles,
      soak_cycles: historyA.soak_cycles + historyB.soak_cycles,
    },
  };

  return { composite, bindingQuality };
}

function fragmentFrom(target: WorldObject, id: ObjID, rng: RNG): WorldObject {
  const latent = latentOf(target);
  return {
    id,
    pos: { x: target.pos.x + rng.normal(0, 0.12), y: target.pos.y + rng.normal(0, 0.12) },
    vel: { x: rng.normal(0, 0.05), y: rng.normal(0, 0.05) },
    shapeType: target.shapeType,
    radius: Math.max(0.05, target.radius * 0.52),
    length: Math.max(0.05, target.length * 0.5),
    thickness: Math.max(0.05, target.thickness * 0.6),
    center_of_mass_offset: {
      x: target.center_of_mass_offset.x * 0.5 + rng.normal(0, 0.02),
      y: target.center_of_mass_offset.y * 0.5 + rng.normal(0, 0.02),
    },
    grip_score: target.grip_score,
    props: mutate(target.props, 0.03, rng),
    integrity: clamp(target.integrity * 0.45),
    debugFamily: target.debugFamily,
    anchored: false,
    latentPrecision: {
      surface_planarity: clamp(latent.surface_planarity * 0.92 + rng.normal(0, 0.02)),
      impurity_level: clamp(latent.impurity_level + rng.normal(0, 0.02)),
      microstructure_order: clamp(latent.microstructure_order * 0.95 + rng.normal(0, 0.02)),
      internal_stress: clamp(latent.internal_stress * 0.8),
      feature_resolution_limit: clamp(latent.feature_resolution_limit * 0.9),
    },
    processHistory: { ...historyOf(target) },
  };
}

export function strike(tool: WorldObject, target: WorldObject, rng: RNG, nextId: () => ObjID): StrikeResult {
  const gripPoint = { x: -tool.length * 0.5, y: 0 };
  const leverArm = Math.max(0.15, Math.hypot(tool.center_of_mass_offset.x - gripPoint.x, tool.center_of_mass_offset.y - gripPoint.y));
  const impactForce = impactPotential(tool.props, tool.length) * leverArm;
  const torque = impactForce * leverArm;
  const inertiaShapeFactor = tool.shapeType === 'sphere' ? 0.4 : tool.shapeType === 'plate' ? 0.1 : tool.shapeType === 'shard' ? 0.12 : 1 / 12;
  const inertia = Math.max(EPSILON, tool.props.mass * (tool.length * tool.length + tool.thickness * tool.thickness) * inertiaShapeFactor);
  const angularVelocity = Math.max(0, Math.min(6, (tool.angularVelocity ?? 0) + torque / inertia));
  tool.angularVelocity = angularVelocity;
  const effectiveCuttingPotential = clamp(cuttingPotential(tool.props) + tool.props.sharpness * 0.2 + tool.grip_score * 0.1);
  const damage = Math.max(0, angularVelocity * effectiveCuttingPotential * 0.35);
  const threshold = 0.22 + target.integrity * 0.42 + target.props.elasticity * 0.18;
  const fractured = damage > threshold;

  if (!fractured) {
    target.integrity = clamp(target.integrity - damage * 0.08);
    tool.integrity = clamp(tool.integrity - target.props.hardness * 0.02);
    return { damage, fractured, fragments: [] };
  }

  target.integrity = clamp(target.integrity * 0.4);
  tool.integrity = clamp(tool.integrity - target.props.hardness * 0.04);
  return {
    damage,
    fractured,
    fragments: [fragmentFrom(target, nextId(), rng), fragmentFrom(target, nextId(), rng)],
  };
}

export function grind(obj: WorldObject, abrasive: WorldObject, intensity = 1): GrindResult {
  const latent = latentOf(obj);
  const history = historyOf(obj);
  const scaled = clamp(intensity);
  const abrasiveFactor = clamp(abrasive.props.roughness * 0.44 + abrasive.props.friction_coeff * 0.56);
  const sharpGain = (1 - obj.props.sharpness) * abrasiveFactor * 0.2 * scaled;
  const stressGain = obj.props.brittleness * abrasiveFactor * scaled * 0.1;
  const wear = (obj.props.brittleness * 0.6 + (1 - obj.props.elasticity) * 0.4 + stressGain * 0.5) * abrasiveFactor * 0.12 * scaled;

  const newObject: WorldObject = {
    ...obj,
    props: {
      ...obj.props,
      sharpness: clamp(obj.props.sharpness + sharpGain),
    },
    integrity: clamp(obj.integrity - wear),
    latentPrecision: {
      ...latent,
      surface_planarity: clamp(latent.surface_planarity + abrasiveFactor * (1 - latent.surface_planarity) * 0.2 * scaled),
      internal_stress: clamp(latent.internal_stress + stressGain),
      feature_resolution_limit: clamp(
        latent.feature_resolution_limit + abrasiveFactor * 0.1 * (1 - latent.internal_stress) * scaled,
      ),
    },
    processHistory: {
      ...history,
      grind_passes: history.grind_passes + 1,
    },
  };

  return { wear, newObject };
}

export function heat(obj: WorldObject, intensity: number): WorldObject {
  const latent = latentOf(obj);
  const history = historyOf(obj);
  const scaled = clamp(intensity);
  const combustLoss = scaled * obj.props.combustibility * 0.4;
  const orderGain = scaled * (1 - latent.impurity_level) * 0.16;
  const stressShift = scaled > 0.65 ? scaled * 0.1 : -scaled * 0.08;
  return {
    ...obj,
    props: {
      ...obj.props,
      brittleness: clamp(obj.props.brittleness + scaled * 0.12),
      elasticity: clamp(obj.props.elasticity - scaled * 0.12),
    },
    integrity: clamp(obj.integrity - combustLoss),
    latentPrecision: {
      ...latent,
      microstructure_order: clamp(latent.microstructure_order + orderGain),
      internal_stress: clamp(latent.internal_stress + stressShift),
      feature_resolution_limit: clamp(latent.feature_resolution_limit + orderGain * 0.35),
    },
    processHistory: {
      ...history,
      thermal_cycles: history.thermal_cycles + 1,
    },
  };
}

export function soak(obj: WorldObject, intensity: number): WorldObject {
  const latent = latentOf(obj);
  const history = historyOf(obj);
  const scaled = clamp(intensity);
  const porous = clamp(obj.props.roughness * 0.55 + (1 - obj.props.density) * 0.45);
  return {
    ...obj,
    props: {
      ...obj.props,
      mass: clamp(obj.props.mass + scaled * 0.1),
      elasticity: clamp(obj.props.elasticity + scaled * 0.08),
    },
    integrity: clamp(obj.integrity - obj.props.brittleness * scaled * 0.08),
    latentPrecision: {
      ...latent,
      impurity_level: clamp(latent.impurity_level - porous * scaled * 0.12 + (1 - porous) * scaled * 0.03),
      internal_stress: clamp(latent.internal_stress - porous * scaled * 0.08),
    },
    processHistory: {
      ...history,
      soak_cycles: history.soak_cycles + 1,
    },
  };
}

export function cool(obj: WorldObject, intensity: number): WorldObject {
  const latent = latentOf(obj);
  const history = historyOf(obj);
  const scaled = clamp(intensity);
  const stressRelief = scaled * (0.12 + latent.microstructure_order * 0.12);
  return {
    ...obj,
    latentPrecision: {
      ...latent,
      internal_stress: clamp(latent.internal_stress - stressRelief),
      microstructure_order: clamp(latent.microstructure_order + scaled * 0.05),
    },
    processHistory: {
      ...history,
      thermal_cycles: history.thermal_cycles + 1,
    },
  };
}
