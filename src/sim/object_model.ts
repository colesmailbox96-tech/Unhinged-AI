import type { PropertyVector } from './properties';

export type ObjID = number;
export type ShapeType = 'sphere' | 'rod' | 'shard' | 'plate';

export interface Vec2 {
  x: number;
  y: number;
}

export interface WorldObject {
  id: ObjID;
  pos: Vec2;
  vel: Vec2;
  shapeType: ShapeType;
  radius: number;
  length: number;
  thickness: number;
  center_of_mass_offset: Vec2;
  grip_score: number;
  props: PropertyVector;
  integrity: number;
  angularVelocity?: number;
  heldBy?: number;
  constituents?: ObjID[];
  debugFamily?: string;
}

export function deriveGripScore(shapeType: ShapeType, length: number, thickness: number, roughness: number): number {
  const slenderness = length / Math.max(0.1, thickness);
  const shapeBonus = shapeType === 'rod' ? 0.24 : shapeType === 'shard' ? 0.16 : shapeType === 'plate' ? 0.12 : 0.08;
  const geometryScore = Math.max(0, Math.min(1, slenderness / 8));
  return Math.max(0, Math.min(1, roughness * 0.5 + geometryScore * 0.4 + shapeBonus));
}

export function estimateContactArea(shapeType: ShapeType, length: number, thickness: number, radius: number): number {
  if (shapeType === 'sphere') return Math.PI * radius * radius;
  if (shapeType === 'rod') return length * thickness;
  if (shapeType === 'plate') return length * thickness * 1.2;
  return (length * thickness) / 2;
}
